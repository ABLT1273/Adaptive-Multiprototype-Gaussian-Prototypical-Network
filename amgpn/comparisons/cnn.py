from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from amgpn.data.episodes import *
from amgpn.data.feature_maps import *
import gc
from amgpn.data.preprocessing import crop_and_rescale_symmetric


def count_parameters(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total:,} ({total/1e3:.1f}K)")
    print(f"Trainable parameters: {trainable:,}")

    return total


class EuclideanDistance(nn.Module):
    """欧氏距离度量"""
    def __init__(self):
        super().__init__()

    def forward(self, query_features, support_features):
        """
        计算欧氏距离

        Args:
            query_features: [n_query, feature_dim]
            support_features: [n_support, feature_dim]

        Returns:
            distances: [n_query, n_support]
        """
        # 扩展维度以进行广播
        # query: [n_query, <configured>, feature_dim]
        # support: [<configured>, n_support, feature_dim]
        query_expanded = query_features.unsqueeze(1)
        support_expanded = support_features.unsqueeze(0)

        # 计算欧氏距离
        distances = torch.sqrt(((query_expanded - support_expanded) ** 2).sum(dim=2))

        return distances


class CNNBaselineTrainer:
    """
    CNN Baseline训练器

    使用ResNeXt提取特征 + 欧氏距离做最近邻分类
    不使用原型机制，直接用support set样本
    """
    def __init__(self, model, device, save_middle=None, mlr=None,
                 crop_ratio_h=None, crop_ratio_l=None, resize=None):
        """
        Args:
            model: CNN骨干网络（只输出特征，不需要v和s分离）
            device: 训练设备
            save_middle: 是否中途保存最优模型
            mlr: 模型学习率
            crop_ratio_h: 高频裁剪比例
            crop_ratio_l: 低频裁剪比例
            resize: 是否resize
        """
        save_middle = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.__init__.save_middle', save_middle)
        mlr = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.__init__.mlr', mlr)
        crop_ratio_h = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.__init__.crop_ratio_h', crop_ratio_h)
        crop_ratio_l = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.__init__.crop_ratio_l', crop_ratio_l)
        resize = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.__init__.resize', resize)
        self.device = device
        self.crop_ratio_h = crop_ratio_h
        self.crop_ratio_l = crop_ratio_l
        self.resize = resize
        self.save_middle = save_middle

        # 初始化模型
        self.model = model
        self.model.to(device)

        # 初始化欧氏距离度量
        self.distance_metric = EuclideanDistance()

        # 优化器
        self.mlr = mlr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mlr)

        # 训练统计
        self.current_epoch = 0

        print("CNN Baseline模式: 使用欧氏距离的最近邻分类")

    def save_model(self, save_path=None):
        """保存模型"""
        save_path = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.save_model.save_path', save_path)
        filtered_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if not ("total_ops" in k or "total_params" in k)
        }
        checkpoint = {
            'model_state_dict': filtered_state_dict,
            'current_epoch': self.current_epoch,
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")

    def _single_task_forward(self, support_signals, support_labels, query_signals, query_labels):
        """
        单个任务的前向传播（CNN baseline版本）

        不使用原型，直接计算query到所有support样本的距离

        Returns:
            loss: 损失值（tensor）
            accuracy: 准确率（float）
            output: 输出字典
        """
        # 移动到设备
        support_signals = support_signals.to(self.device).float().unsqueeze(1)
        support_labels = support_labels.to(self.device)
        query_signals = query_signals.to(self.device).float().unsqueeze(1)
        query_labels = query_labels.to(self.device)

        # 数据增强（与GPN保持一致）
        if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
            processed_support_list = []
            for signal in support_signals.unbind(0):
                processed_support_list.append(
                    crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                )
            support_signals = torch.stack(processed_support_list, dim=0)

            processed_query_list = []
            for signal in query_signals.unbind(0):
                processed_query_list.append(
                    crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                )
            query_signals = torch.stack(processed_query_list, dim=0)

        # 提取特征（只需要v，不需要s）
        support_features = self.model(support_signals)  # [n_support, feature_dim]
        query_features = self.model(query_signals)      # [n_query, feature_dim]

        # 标签重映射
        unique_labels = torch.unique(support_labels)
        n_ways = len(unique_labels)
        label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

        remapped_support_labels = torch.tensor(
            [label_mapping[label.item()] for label in support_labels],
            device=self.device
        )
        remapped_query_labels = torch.tensor(
            [label_mapping[label.item()] for label in query_labels],
            device=self.device
        )

        # 计算query到所有support样本的欧氏距离
        distances = self.distance_metric(query_features, support_features)  # [n_query, n_support]

        # 对每个query，计算到每个类的距离（取该类中最近样本的距离作为logit）
        logits = torch.zeros(query_features.size(0), n_ways).to(self.device)

        for i in range(n_ways):
            # 找到属于第i类的support样本
            class_mask = (remapped_support_labels == i)
            class_distances = distances[:, class_mask]  # [n_query, k_shot]

            # 使用最小距离（最近邻）
            logits[:, i] = -class_distances.min(dim=1)[0]  # 负距离作为logit

        # 计算概率和损失
        probabilities = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, remapped_query_labels)

        # 计算准确率
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = (predictions == remapped_query_labels).float().mean()

        # 构造输出字典
        output = {
            'loss': loss,
            'accuracy': accuracy,
            'probabilities': probabilities,
            'distances': distances,
        }

        # 返回三元组（与源文件保持一致）
        return loss, accuracy.item(), output

    def train_step_batch(self, meta_batch, batch_idx):
        """
        单个batch的训练步骤（完全按照源文件的逻辑）

        Args:
            meta_batch: 元组 (support_signals, support_labels, query_signals, query_labels)
            batch_idx: 当前batch索引

        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        # 解包meta_batch（关键：按照源文件的格式）
        support_signals, support_labels, query_signals, query_labels = meta_batch
        batch_size = support_signals.size(0)

        total_loss = 0
        total_acc = 0

        self.optimizer.zero_grad()

        # 对batch中的每个task单独处理
        for i in range(batch_size):
            loss, acc, output = self._single_task_forward(
                support_signals[i],
                support_labels[i],
                query_signals[i],
                query_labels[i]
            )

            # 梯度累积（除以batch_size）
            (loss / batch_size).backward()

            total_loss += loss.item()
            total_acc += acc

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('CNN_baseline.py.CNNBaselineTrainer.train_step_batch.max_norm'))
        self.optimizer.step()

        return total_loss / batch_size, total_acc / batch_size

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        processed_batches = 0

        # 使用tqdm显示进度
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]") as pbar:
            for batch_idx, meta_batch in enumerate(pbar):
                # 批处理训练步骤
                loss, acc = self.train_step_batch(meta_batch, batch_idx)

                total_loss += loss
                total_accuracy += acc
                processed_batches += 1

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{acc:.4f}'
                })

                # 清理内存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        avg_loss = total_loss / processed_batches
        avg_accuracy = total_accuracy / processed_batches

        return avg_loss, avg_accuracy

    def validate(self, val_loader):
        """验证（使用与训练相同的数据格式）"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        processed_batches = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]") as pbar:
                for batch_idx, meta_batch in enumerate(pbar):
                    # 解包meta_batch
                    support_signals, support_labels, query_signals, query_labels = meta_batch
                    batch_size = support_signals.size(0)

                    batch_loss = 0
                    batch_accuracy = 0

                    # 对batch中的每个task单独处理
                    for i in range(batch_size):
                        loss, acc, output = self._single_task_forward(
                            support_signals[i],
                            support_labels[i],
                            query_signals[i],
                            query_labels[i]
                        )

                        batch_loss += loss.item()
                        batch_accuracy += acc

                    # 平均
                    batch_loss = batch_loss / batch_size
                    batch_accuracy = batch_accuracy / batch_size

                    total_loss += batch_loss
                    total_accuracy += batch_accuracy
                    processed_batches += 1

                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'acc': f'{batch_accuracy:.4f}'
                    })

                    # 清理内存
                    del output
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

        avg_loss = total_loss / processed_batches
        avg_accuracy = total_accuracy / processed_batches

        return avg_loss, avg_accuracy

    def train(self, train_loader, val_loader, num_epochs, save_path=None):
        """
        完整训练流程（按照源文件的训练逻辑）

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_path: 模型保存路径
        """
        # 学习率调度器
        save_path = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.train.save_path', save_path)
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=_cfg_require('CNN_baseline.py.CNNBaselineTrainer.train.step_size'),
            gamma=_cfg_require('CNN_baseline.py.CNNBaselineTrainer.train.gamma')
        )

        print("\n" + "="*50)
        print("Starting CNN Baseline Training")
        print("="*50)
        print(f"总训练轮数: {num_epochs} epochs")
        print(f"每个epoch有 {len(train_loader)} 个批次")
        print(f"实际batch_size: {train_loader.batch_size}")
        print(f"每个epoch训练 {len(train_loader) * train_loader.batch_size} 个任务")
        print("="*50)

        best_val_acc = 0
        early_stop_counter = 0
        early_stop_patience = _cfg_require('CNN_baseline.py.CNNBaselineTrainer.train.early_stop_patience')  # <configured>个epoch没提升就停止

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            if (epoch+1)%5==0:
                # 验证
                val_loss, val_acc = self.validate(val_loader)

                # 打印结果
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")

                # 保存最优模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if self.save_middle:
                        self.save_model(save_path)
                        print(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f"No improvement ({early_stop_counter}/{early_stop_patience})")

                # Early stopping
                if early_stop_counter >= early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best Val Acc: {best_val_acc:.4f}")
                    break

                print("-" * 50)

            # 学习率调整
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            # 清理
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best Val Acc: {best_val_acc:.4f}")
        print(f"Model saved to: {save_path}")
        print(f"{'='*50}\n")

        return best_val_acc

    def evaluate(self, test_loader, num_episodes=None):
        """
        评估模型 - 统计准确率的均值和标准差

        关键：数据格式与训练不同！
        - 训练：每次读取一个batch的tasks
        - 评估：每次读取单个task，且需要permute和flatten

        Args:
            test_loader: 测试数据加载器
            num_episodes: 测试episode数量

        Returns:
            mean_accuracy: 平均准确率
            std_accuracy: 准确率标准差
        """
        num_episodes = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.evaluate.num_episodes', num_episodes)
        self.model.eval()
        accuracies = []

        print("\n" + "="*50)
        print(f"Evaluating CNN Baseline on {num_episodes} episodes")
        print("="*50)

        iterator = tqdm(test_loader, total=num_episodes, desc="Testing")

        with torch.no_grad():
            for task_id, meta_task in enumerate(iterator):
                if task_id >= num_episodes:
                    break

                # 解包数据（每次是单个task）
                support_signals, support_labels, query_signals, query_labels = meta_task

                # ========== 关键：按照源文件的数据处理方式 ==========
                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                # 数据增强（与源文件保持一致）
                if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
                    processed_support_list = []
                    for signal in support_signals.unbind(0):
                        processed_support_list.append(
                            crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                        )
                    support_signals = torch.stack(processed_support_list, dim=0)

                    processed_query_list = []
                    for signal in query_signals.unbind(0):
                        processed_query_list.append(
                            crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                        )
                    query_signals = torch.stack(processed_query_list, dim=0)

                # 提取特征
                support_features = self.model(support_signals)
                query_features = self.model(query_signals)

                # 标签重映射
                unique_labels = torch.unique(support_labels)
                n_ways = len(unique_labels)
                label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

                remapped_support_labels = torch.tensor(
                    [label_mapping[label.item()] for label in support_labels],
                    device=self.device
                )
                remapped_query_labels = torch.tensor(
                    [label_mapping[label.item()] for label in query_labels],
                    device=self.device
                )

                # 计算距离和logits
                distances = self.distance_metric(query_features, support_features)
                logits = torch.zeros(query_features.size(0), n_ways).to(self.device)

                for i in range(n_ways):
                    class_mask = (remapped_support_labels == i)
                    class_distances = distances[:, class_mask]
                    logits[:, i] = -class_distances.min(dim=1)[0]

                # 计算准确率
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == remapped_query_labels).float().mean()
                accuracies.append(accuracy.item())

                # 更新进度条
                if len(accuracies) > 0:
                    current_mean = np.mean(accuracies)
                    current_std = np.std(accuracies)
                    iterator.set_postfix({
                        'mean_acc': f'{current_mean:.4f}',
                        'std': f'{current_std:.4f}'
                    })

                # 清理内存
                if task_id % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        # 计算统计量
        accuracies = np.array(accuracies[:num_episodes])
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # <configured> 置信区间
        confidence_interval = 1.96 * std_accuracy / np.sqrt(len(accuracies))

        print("\n" + "="*50)
        print("Evaluation Results:")
        print("="*50)
        print(f"Number of episodes: {len(accuracies)}")
        print(f"Mean Accuracy: {mean_accuracy*100:.2f}%")
        print(f"Std Deviation: {std_accuracy*100:.2f}%")
        print(f"95% Confidence Interval: ±{confidence_interval*100:.2f}%")
        print(f"Final Result: {mean_accuracy*100:.2f}% ± {confidence_interval*100:.2f}%")
        print("="*50)

        return mean_accuracy, std_accuracy


    def evaluate_single_task(self, test_loader):
        '\n        评估单个任务（用于full_evaluation）\n        \n        Args:\n            test_loader: 只包含<configured>个task的DataLoader\n            \n        Returns:\n            accuracy: 单个任务的准确率（float）\n        '
        self.model.eval()

        with torch.no_grad():
            # 获取单个task
            meta_task = next(iter(test_loader))
            support_signals, support_labels, query_signals, query_labels = meta_task

            # 数据处理（与evaluate方法保持一致）
            support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
            support_labels = support_labels.to(self.device).flatten()
            query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
            query_labels = query_labels.to(self.device).flatten()

            # 数据增强
            if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
                processed_support_list = []
                for signal in support_signals.unbind(0):
                    processed_support_list.append(
                        crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                    )
                support_signals = torch.stack(processed_support_list, dim=0)

                processed_query_list = []
                for signal in query_signals.unbind(0):
                    processed_query_list.append(
                        crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                    )
                query_signals = torch.stack(processed_query_list, dim=0)

            # 提取特征
            support_features = self.model(support_signals)
            query_features = self.model(query_signals)

            # 标签重映射
            unique_labels = torch.unique(support_labels)
            n_ways = len(unique_labels)
            label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

            remapped_support_labels = torch.tensor(
                [label_mapping[label.item()] for label in support_labels],
                device=self.device
            )
            remapped_query_labels = torch.tensor(
                [label_mapping[label.item()] for label in query_labels],
                device=self.device
            )

            # 计算距离和logits
            distances = self.distance_metric(query_features, support_features)
            logits = torch.zeros(query_features.size(0), n_ways).to(self.device)

            for i in range(n_ways):
                class_mask = (remapped_support_labels == i)
                class_distances = distances[:, class_mask]
                logits[:, i] = -class_distances.min(dim=1)[0]

            # 计算准确率
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == remapped_query_labels).float().mean()

            return accuracy.item()

    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """
        完整的评估协议（符合论文标准）

        对不同的n_way和k_shot组合进行评估，每个组合评估n_trials次

        Args:
            test_dataset: 测试数据集（UAVDataset）
            n_trials: 每个配置的试验次数
            q_query: 每类的查询样本数

        Returns:
            results: 字典，key为'{n_way}w{k_shot}s'，value为(mean_acc, std_acc)
        """
        n_trials = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('CNN_baseline.py.CNNBaselineTrainer.full_evaluation.q_query', q_query)
        from amgpn.data.episodes import MetaDataset

        self.model.eval()
        results = {}

        print("\n" + "="*70)
        print("Starting Full Evaluation Protocol")
        print("="*70)

        for n_way in [8, 7, 6, 5, 4, 3]:
            for k_shot in [10, 5, 1]:
                print(f"\n{'='*50}")
                print(f"Evaluating {n_way}-way {k_shot}-shot")
                print(f"{'='*50}")

                accuracies = []

                # 使用tqdm显示trials进度
                pbar = tqdm(range(n_trials),
                           desc=f"{n_way}w{k_shot}s",
                           position=0,
                           leave=True)

                for trial in pbar:
                    # 为每个trial创建一个新的任务
                    meta_test = MetaDataset(
                        test_dataset,
                        n_way=n_way,
                        k_shot=k_shot,
                        q_query=q_query,
                        num_tasks=_cfg_require('CNN_baseline.py.CNNBaselineTrainer.full_evaluation.num_tasks'),
                        seed=random.randint(0, _cfg_require('CNN_baseline.py.CNNBaselineTrainer.full_evaluation.size_or_budget'))
                    )

                    loader = DataLoader(meta_test, batch_size=_cfg_require('CNN_baseline.py.CNNBaselineTrainer.full_evaluation.batch_size'), shuffle=False)

                    # 评估单个任务
                    acc = self.evaluate_single_task(loader)
                    accuracies.append(acc)

                    # 实时更新进度条显示当前平均准确率
                    if len(accuracies) > 0:
                        pbar.set_postfix({
                            'mean_acc': f"{np.mean(accuracies):.4f}",
                            'current': f"{acc:.4f}"
                        })

                    # 清理内存
                    del meta_test, loader

                    # 每<configured>个trial进行一次显存清理
                    if (trial + 1) % 10 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # 计算统计量
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                ci_95 = 1.96 * std_acc / np.sqrt(len(accuracies))

                results[f'{n_way}w{k_shot}s'] = (mean_acc, std_acc)

                print(f"\nResult: {mean_acc*100:.2f}% ± {ci_95*100:.2f}% (std: {std_acc*100:.2f}%)")

                # 清理
                del accuracies, pbar
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        # 最终清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 打印汇总结果
        print("\n" + "="*70)
        print("Full Evaluation Results Summary")
        print("="*70)
        print(f"{'Configuration':<15} {'Accuracy (%)':<15} {'Std (%)':<10} {'95% CI':<10}")
        print("-"*70)
        for config, (mean_acc, std_acc) in results.items():
            ci_95 = 1.96 * std_acc / np.sqrt(n_trials)
            print(f"{config:<15} {mean_acc*100:>6.2f} ± {ci_95*100:>5.2f}   {std_acc*100:>6.2f}%")
        print("="*70)

        return results

# ==================== 简化的CNN模型 ====================

class SEModule(nn.Module):
    """SE注意力模块"""
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNeXtBlock(nn.Module):
    """ResNeXt Block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=None,
                 cardinality=None, reduction=None):
        stride = _cfg_resolve('CNN_baseline.py.ResNeXtBlock.__init__.stride', stride)
        cardinality = _cfg_resolve('CNN_baseline.py.ResNeXtBlock.__init__.cardinality', cardinality)
        reduction = _cfg_resolve('CNN_baseline.py.ResNeXtBlock.__init__.reduction', reduction)
        super().__init__()

        mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=_cfg_require('CNN_baseline.py.ResNeXtBlock.__init__.kernel_size'), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=_cfg_require('CNN_baseline.py.ResNeXtBlock.__init__.kernel_size__2'), stride=stride,
                              padding=_cfg_require('CNN_baseline.py.ResNeXtBlock.__init__.padding'), groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                              kernel_size=_cfg_require('CNN_baseline.py.ResNeXtBlock.__init__.kernel_size__3'), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.SiLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=_cfg_require('CNN_baseline.py.ResNeXtBlock.__init__.kernel_size__4'), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.attention = SEModule(out_channels, reduction=reduction)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.attention is not None:
            out = self.attention(out)

        out += identity
        out = self.relu(out)

        return out

class CNN_Baseline_Simple(nn.Module):
    '\n    简化的<configured>层CNN Baseline\n    \n    使用标准卷积层\n    保持与GPN相近的参数量和相同的输入输出维度\n    \n    Args:\n        feature_dim: 输出特征维度（由外部配置提供）\n    '
    def __init__(self, feature_dim=None):
        feature_dim = _cfg_resolve('CNN_baseline.py.CNN_Baseline_Simple.__init__.feature_dim', feature_dim)
        super().__init__()

        self.feature_dim = feature_dim

        # Layer <configured>: <configured> → <configured>
        self.conv1 = nn.Conv2d(_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.Conv2d_arg0'), _cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.Conv2d_arg1'), kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.kernel_size'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.stride'), padding=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.padding'), bias=False)
        self.bn1 = nn.BatchNorm2d(_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.BatchNorm2d_arg0'))
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.kernel_size__2'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.stride__2'), padding=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.padding__2'))
        # Output: [B, <configured>, <configured>, <configured>]

        # Layer <configured>: <configured> → <configured>
        self.conv2 = nn.Conv2d(_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.Conv2d_arg0__2'), _cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.Conv2d_arg1__2'), kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.kernel_size__3'), padding=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.padding__3'), bias=False)
        self.bn2 = nn.BatchNorm2d(_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.BatchNorm2d_arg0__2'))
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.kernel_size__4'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.stride__3'))
        # Output: [B, <configured>, <configured>, <configured>]

        # Layer <configured>: <configured> → <configured>
        self.conv3 = nn.Conv2d(_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.Conv2d_arg0__3'), _cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.Conv2d_arg1__3'), kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.kernel_size__5'), padding=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.padding__4'), bias=False)
        self.bn3 = nn.BatchNorm2d(_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.BatchNorm2d_arg0__3'))
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.kernel_size__6'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.stride__4'))
        # Output: [B, <configured>, <configured>, <configured>]

        # Layer <configured>: <configured> → <configured>
        self.conv4 = nn.Conv2d(_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.Conv2d_arg0__4'), _cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.Conv2d_arg1__4'), kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.kernel_size__7'), padding=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.padding__5'), bias=False)
        self.bn4 = nn.BatchNorm2d(_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.BatchNorm2d_arg0__4'))
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.kernel_size__8'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline_Simple.__init__.stride__5'))
        # Output: [B, <configured>, <configured>, <configured>]

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def forward(self, x):
        '\n        前向传播\n        \n        Args:\n            x: [B, <configured>, <configured>, <configured>]\n            \n        Returns:\n            features: [B, <configured>] 特征向量\n        '
        # Layer <configured>
        x = self.conv1(x)       # [B, <configured>, <configured>, <configured>]
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)       # [B, <configured>, <configured>, <configured>]

        # Layer <configured>
        x = self.conv2(x)       # [B, <configured>, <configured>, <configured>]
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)       # [B, <configured>, <configured>, <configured>]

        # Layer <configured>
        x = self.conv3(x)       # [B, <configured>, <configured>, <configured>]
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)       # [B, <configured>, <configured>, <configured>]

        # Layer <configured>
        x = self.conv4(x)       # [B, <configured>, <configured>, <configured>]
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)       # [B, <configured>, <configured>, <configured>]

        # Global Average Pooling
        x = self.avgpool(x)     # [B, <configured>, <configured>, <configured>]
        features = x.flatten(1) # [B, <configured>]

        return features

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CNN_Baseline(nn.Module):
    """
    CNN Baseline模型 - 只输出特征向量

    与GPN_Optimized使用相同的骨干网络，但只输出embedding特征（不需要precision）
    """
    def __init__(self, reduction=None, feature_dim=None):
        reduction = _cfg_resolve('CNN_baseline.py.CNN_Baseline.__init__.reduction', reduction)
        feature_dim = _cfg_resolve('CNN_baseline.py.CNN_Baseline.__init__.feature_dim', feature_dim)
        super().__init__()

        self.feature_dim = feature_dim

        # 输入层
        self.conv1 = nn.Conv2d(_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.Conv2d_arg0'), _cfg_require('CNN_baseline.py.CNN_Baseline.__init__.Conv2d_arg1'), kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.kernel_size'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride'),
                              padding=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.padding'), bias=False)
        self.bn1 = nn.BatchNorm2d(_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.BatchNorm2d_arg0'))
        self.relu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.kernel_size__2'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__2'), padding=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.padding__2'))

        # Stage <configured>: <configured>→<configured>
        self.block1 = ResNeXtBlock(_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.ResNeXtBlock_arg0'), _cfg_require('CNN_baseline.py.CNN_Baseline.__init__.ResNeXtBlock_arg1'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__3'), cardinality=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.cardinality'), reduction=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.reduction__2'))
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.kernel_size__3'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__4'))

        # Stage <configured>: <configured>→<configured>
        self.block2 = ResNeXtBlock(_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.ResNeXtBlock_arg0__2'), _cfg_require('CNN_baseline.py.CNN_Baseline.__init__.ResNeXtBlock_arg1__2'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__5'), cardinality=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.cardinality__2'), reduction=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.reduction__3'))
        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.kernel_size__4'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__6'))

        # Stage <configured>: <configured>→<configured>
        self.block3 = ResNeXtBlock(_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.ResNeXtBlock_arg0__3'), _cfg_require('CNN_baseline.py.CNN_Baseline.__init__.ResNeXtBlock_arg1__3'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__7'), cardinality=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.cardinality__3'), reduction=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.reduction__4'))
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.kernel_size__5'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__8'))

        self.block4 = ResNeXtBlock(_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.ResNeXtBlock_arg0__4'), _cfg_require('CNN_baseline.py.CNN_Baseline.__init__.ResNeXtBlock_arg1__4'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__9'), cardinality=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.cardinality__4'), reduction=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.reduction__5'))
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.kernel_size__6'), stride=_cfg_require('CNN_baseline.py.CNN_Baseline.__init__.stride__10'))

        # Global Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 只保留前<configured>维作为特征（与GPN保持一致）
        # 或者可以添加一个额外的全连接层来映射到指定维度

        self._initialize_weights()

    def forward(self, x):
        '\n        前向传播 - 只输出embedding特征\n        \n        Args:\n            x: [B, <configured>, <configured>, <configured>]\n            \n        Returns:\n            features: [B, feature_dim] 特征向量\n        '
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)
        x = self.pool4(x)  # [B, <configured>, <configured>, <configured>]


        # Global Average Pooling
        features = self.avgpool(x).flatten(1)  # [B, feature_dim]

        return features

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
