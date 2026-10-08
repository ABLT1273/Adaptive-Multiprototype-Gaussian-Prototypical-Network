from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
#GPN模型二次复现：更严谨的参数配置
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_loader import *
from loss_Copy1 import GPNLoss
import gc

def count_parameters(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 分别统计SE参数
    se_params = 0
    for name, module in model.named_modules():
        if isinstance(module, SEModule):
            se_params += sum(p.numel() for p in module.parameters())

    print(f"Total parameters: {total:,} ({total/1e3:.1f}K)")
    print(f"Trainable parameters: {trainable:,}")
    print(f"SE module parameters: {se_params:,} ({se_params/1e3:.1f}K)")
    print(f"Non-SE parameters: {total-se_params:,} ({(total-se_params)/1e3:.1f}K)")

    return total

class GPNTrainer:
    """
    GPN模型训练器
    """
    def __init__(self, model,device,feature_dim=None, lr=None):
        feature_dim = _cfg_resolve('old/GPN_ori.py.GPNTrainer.__init__.feature_dim', feature_dim)
        lr = _cfg_resolve('old/GPN_ori.py.GPNTrainer.__init__.lr', lr)
        self.device = device
        self.feature_dim = feature_dim

        # 初始化模型
        self.model = model
        self.model.to(device)
        self.loss_fn = GPNLoss()
        self.lr=lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"GPN模型初始化完成，设备: {self.device}")

    def save_model(self, save_path=None):
        """
        仅保存模型权重（用于部署）
        """
        save_path = _cfg_resolve('old/GPN_ori.py.GPNTrainer.save_model.save_path', save_path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_dim': self.feature_dim,
        }, save_path)
        print(f"Model weights saved to {save_path}")

    def load_model(self, model_path):
        """
        仅加载模型权重
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from {model_path}")

    def compute_gaussian_prototypes(self, support_v, support_s, support_labels, n_ways):
        '\n        计算高斯原型和精度矩阵（论文公式<configured>-<configured>）\n        \n        Args:\n            support_v: 支持集embedding特征 [n_ways * k_shot, feature_dim]\n            support_s: 支持集precision特征 [n_ways * k_shot, feature_dim]\n            support_labels: 支持集标签 [n_ways * k_shot]\n            n_ways: 类别数\n            k_shot: 每类样本数\n            \n        Returns:\n            prototypes: 各类原型 [n_ways, feature_dim]\n            precision_matrices: 各类精度矩阵 [n_ways, feature_dim, feature_dim]\n        '
        feature_dim = support_v.shape[1]

        prototypes = torch.zeros(n_ways, feature_dim, device=self.device)
        precision_matrices = torch.zeros(n_ways, feature_dim, feature_dim, device=self.device)

        sigma = support_s

        # 为每个类别计算原型
        unique_labels = torch.unique(support_labels)

        for i, class_label in enumerate(unique_labels):
            class_mask = (support_labels == class_label)

            class_v = support_v[class_mask]  # [k_shot, feature_dim]
            class_sigma = sigma[class_mask]  # [k_shot, feature_dim]

            weighted_sum = torch.sum(class_sigma * class_v, dim=0)  # [feature_dim]
            sigma_sum = torch.sum(class_sigma, dim=0)  # [feature_dim]
            prototypes[i] = weighted_sum / (sigma_sum + 1e-8)

            # 计算精度矩阵（对角矩阵，公式<configured>）
            precision_diag = torch.mean(class_sigma, dim=0)  # [feature_dim]
            precision_matrices[i] = torch.diag(precision_diag)

        return prototypes, precision_matrices


    def train_step_batch(self, meta_batch):
        """
        批处理训练步骤 - 累积梯度后统一更新
        """
        self.model.train()

        batch_size = len(meta_batch[0])  # meta_batch中任务的数量
        total_loss = 0
        total_acc = 0

        # 清零梯度
        self.optimizer.zero_grad()

        # 处理批内所有任务，累积梯度
        for i in range(batch_size):
            support_signals, support_labels = meta_batch[0][i], meta_batch[1][i]
            query_signals, query_labels = meta_batch[2][i], meta_batch[3][i]

            # 单个任务的前向传播和损失计算
            loss, acc = self._single_task_forward(
                support_signals, support_labels,
                query_signals, query_labels
            )

            # 累积损失（自动累积梯度）
            (loss / batch_size).backward()  # 除以batch_size来平均化梯度

            total_loss += loss.item()
            total_acc += acc

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('old/GPN_ori.py.GPNTrainer.train_step_batch.max_norm'))

        # 统一更新参数
        self.optimizer.step()

        return total_loss / batch_size, total_acc / batch_size

    def _single_task_forward(self, support_signals, support_labels, query_signals, query_labels):
        """
        单个任务的前向传播（不更新参数）
        """
        # 移动到设备
        support_signals = support_signals.to(self.device).float().unsqueeze(1)
        support_labels = support_labels.to(self.device)
        query_signals = query_signals.to(self.device).float().unsqueeze(1)
        query_labels = query_labels.to(self.device)

        # 前向传播
        support_v, support_s = self.model(support_signals)
        query_v, query_s = self.model(query_signals)

        # 标签重映射
        unique_labels = torch.unique(support_labels)
        n_ways = len(unique_labels)
        label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

        remapped_support_labels = torch.tensor([label_mapping[label.item()] for label in support_labels], device=self.device)
        remapped_query_labels = torch.tensor([label_mapping[label.item()] for label in query_labels], device=self.device)

        # 计算高斯原型
        prototypes, precision_matrices = self.compute_gaussian_prototypes(
            support_v, support_s, remapped_support_labels, n_ways
        )

        # 计算损失
        loss, probabilities, distances = self.loss_fn(
            query_v, prototypes, precision_matrices, remapped_query_labels
        )

        # 计算准确率
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = (predictions == remapped_query_labels).float().mean()

        return loss, accuracy.item()

    def train(self,train_loader,test_loader, num_tasks=None, n_ways=None, epochs=None,save_path=None):
        """
        改进的元学习训练循环
        """
        num_tasks = _cfg_resolve('old/GPN_ori.py.GPNTrainer.train.num_tasks', num_tasks)
        n_ways = _cfg_resolve('old/GPN_ori.py.GPNTrainer.train.n_ways', n_ways)
        epochs = _cfg_resolve('old/GPN_ori.py.GPNTrainer.train.epochs', epochs)
        save_path = _cfg_resolve('old/GPN_ori.py.GPNTrainer.train.save_path', save_path)
        self.model.train()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=_cfg_require('old/GPN_ori.py.GPNTrainer.train.step_size'), gamma=_cfg_require('old/GPN_ori.py.GPNTrainer.train.gamma'))

        print(f"开始元学习训练: {epochs} epochs, {num_tasks} tasks/epoch")

        best_acc=0
        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            processed_tasks = 0
            # 使用DataLoader的自然批处理
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, meta_batch in enumerate(pbar):

                    if processed_tasks >= num_tasks:
                        break


                    # 批处理训练步骤
                    loss, acc = self.train_step_batch(meta_batch)
                    total_loss += loss
                    total_acc += acc
                    processed_tasks += len(meta_batch[0])  # 实际处理的任务数

                    pbar.set_postfix(
                        loss=f'{loss:.4f}',
                        accuracy=f'{acc:.4f}',
                        tasks=f'{processed_tasks}/{num_tasks}'
                    )

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)

            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

            if (epoch+1)%10==0:
                avg_acc=self.evaluate(test_loader=test_loader,n_ways=n_ways)
                if avg_acc>best_acc:
                    best_acc=avg_acc
                    self.save_model(save_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    def evaluate(self, test_loader, n_ways, show_progress=True):
        """
        评估模型性能

        Args:
            test_loader: 数据加载器
            n_ways: 类别数
            show_progress: 是否显示进度条
        """
        self.model.eval()
        total_acc = 0

        # 根据参数决定是否使用tqdm
        iterator = tqdm(test_loader, desc="Evaluating") if show_progress else test_loader

        with torch.no_grad():
            for meta_task in iterator:
                support_signals, support_labels, query_signals, query_labels = meta_task

                # 移动到设备
                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                # 前向传播
                support_v, support_s = self.model(support_signals)

                # 重新映射标签
                unique_labels = torch.unique(support_labels)
                label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

                remapped_support_labels = torch.tensor(
                    [label_mapping[label.item()] for label in support_labels],
                    device=self.device
                )
                remapped_query_labels = torch.tensor(
                    [label_mapping[label.item()] for label in query_labels],
                    device=self.device
                )

                # 计算高斯原型
                prototypes, precision_matrices = self.compute_gaussian_prototypes(
                    support_v, support_s, remapped_support_labels, n_ways
                )

                # 计算距离和准确率
                query_v, query_s = self.model(query_signals)
                distances = self.loss_fn.compute_mahalanobis_distance(
                    query_v, prototypes, precision_matrices
                )
                logits = -distances
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == remapped_query_labels).float().mean()
                total_acc += accuracy.item()

        avg_acc = total_acc / len(test_loader)
        print(f"当前测试acc:{avg_acc}")
        return avg_acc
    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """完整的评估协议（符合论文）"""
        n_trials = _cfg_resolve('old/GPN_ori.py.GPNTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('old/GPN_ori.py.GPNTrainer.full_evaluation.q_query', q_query)
        results = {}

        for n_way in [3, 4, 5, 6, 7, 8]:
            for k_shot in [1, 5, 10]:
                print(f"\n{'='*50}")
                print(f"Evaluating {n_way}-way {k_shot}-shot")
                print(f"{'='*50}")

                accuracies = []

                # 使用tqdm显示trials进度，position参数避免嵌套冲突
                pbar = tqdm(range(n_trials),
                           desc=f"{n_way}w{k_shot}s",
                           position=0,
                           leave=True)

                for trial in pbar:
                    meta_test = MetaDataset(
                        test_dataset, n_way, k_shot, q_query=q_query,
                        batch_size=_cfg_require('old/GPN_ori.py.GPNTrainer.full_evaluation.batch_size'), num_tasks=_cfg_require('old/GPN_ori.py.GPNTrainer.full_evaluation.num_tasks'), epochs=_cfg_require('old/GPN_ori.py.GPNTrainer.full_evaluation.epochs')
                    )
                    loader = DataLoader(meta_test, batch_size=_cfg_require('old/GPN_ori.py.GPNTrainer.full_evaluation.batch_size__2'))

                    # 关闭内层进度条
                    acc = self.evaluate(loader, n_way, show_progress=False)
                    accuracies.append(acc)

                    # 实时更新进度条显示当前平均准确率
                    if len(accuracies) > 0:
                        pbar.set_postfix({
                            'mean_acc': f"{np.mean(accuracies):.4f}",
                            'current': f"{acc:.4f}"
                        })

                    # ===== 关键：每个trial后立即清理 =====
                    del meta_test, loader, acc

                    # 每<configured>个trial进行一次显存清理
                    if (trial + 1) % 10 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                results[f'{n_way}w{k_shot}s'] = (mean_acc, std_acc)
                print(f"\nResult: {mean_acc:.4f} ± {std_acc:.4f}")

                del accuracies, pbar
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 确保CUDA操作完成


        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return results


def main():
    # 参数设置
    FEATURE_DIM = _cfg_require('old/GPN_ori.py.main.FEATURE_DIM')
    N_WAY = _cfg_require('old/GPN_ori.py.main.N_WAY')
    K_SHOT = _cfg_require('old/GPN_ori.py.main.K_SHOT')
    Q_QUERY = _cfg_require('old/GPN_ori.py.main.Q_QUERY')
    NUM_TASKS = _cfg_require('old/GPN_ori.py.main.NUM_TASKS')
    EPOCHS = _cfg_require('old/GPN_ori.py.main.EPOCHS')
    LR = _cfg_require('old/GPN_ori.py.main.LR')
    BATCH_SIZE = _cfg_require('old/GPN_ori.py.main.BATCH_SIZE')
    SEED = _cfg_require('old/GPN_ori.py.main.SEED')
    SAVE_MODEL_PATH=_cfg_require('old/GPN_ori.py.main.SAVE_MODEL_PATH')

    # 数据路径
    dataset_path = _cfg_require('old/GPN_ori.py.main.dataset_path')

    # 创建原始数据集
    train_dataset=UAVDataset(data_dir_path=dataset_path,seed=SEED,is_train=True,train_ratio=_cfg_require('old/GPN_ori.py.main.train_ratio'),max_sample_count=_cfg_require('old/GPN_ori.py.main.max_sample_count'))
    test_dataset=UAVDataset(data_dir_path=dataset_path,seed=SEED,is_train=False,train_ratio=_cfg_require('old/GPN_ori.py.main.train_ratio__2'),max_sample_count=_cfg_require('old/GPN_ori.py.main.max_sample_count__2'))

    # 创建元任务划分集
    meta_train_dataset = MetaDataset(train_dataset, N_WAY, K_SHOT, Q_QUERY,batch_size=BATCH_SIZE,num_tasks=NUM_TASKS,epochs=EPOCHS)
    meta_test_dataset = MetaDataset(test_dataset, N_WAY, K_SHOT, Q_QUERY,batch_size=BATCH_SIZE,num_tasks=_cfg_require('old/GPN_ori.py.main.num_tasks'),epochs=_cfg_require('old/GPN_ori.py.main.epochs'))

    # DataLoader
    train_loader = DataLoader(meta_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=_cfg_require('old/GPN_ori.py.main.num_workers'))
    test_loader = DataLoader(meta_test_dataset, batch_size=_cfg_require('old/GPN_ori.py.main.batch_size'), shuffle=False)

    #创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_without_se = GPN(use_se=_cfg_require('old/GPN_ori.py.main.use_se'))
    params_without_se = count_parameters(model_without_se)
    # 创建训练器
    meta_learner = GPNTrainer(model_without_se,device=device,feature_dim=FEATURE_DIM,lr=LR)
    print("开始训练...")
    meta_learner.train(train_loader=train_loader,test_loader=test_loader,num_tasks=NUM_TASKS,n_ways=N_WAY,epochs=EPOCHS,save_path=SAVE_MODEL_PATH)

    print("训练完成！")

    meta_learner.full_evaluation(test_dataset=test_dataset,n_trials=_cfg_require('old/GPN_ori.py.main.n_trials'))

if __name__ == '__main__':
    main()


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 确保reduced_channels至少为<configured>
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNeXtBlock(nn.Module):
    """
    标准ResNeXt Bottleneck Block
    结构: conv1→BN→ReLU → conv2→BN→ReLU → conv3→BN → Add → ReLU → MaxPool
    """
    def __init__(self, channels=None, cardinality=None,use_se=None,se_reduction=None):
        channels = _cfg_resolve('old/GPN_ori.py.ResNeXtBlock.__init__.channels', channels)
        cardinality = _cfg_resolve('old/GPN_ori.py.ResNeXtBlock.__init__.cardinality', cardinality)
        use_se = _cfg_resolve('old/GPN_ori.py.ResNeXtBlock.__init__.use_se', use_se)
        se_reduction = _cfg_resolve('old/GPN_ori.py.ResNeXtBlock.__init__.se_reduction', se_reduction)
        super().__init__()

        # 第一个<configured>×<configured>卷积
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=_cfg_require('old/GPN_ori.py.ResNeXtBlock.__init__.kernel_size'), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # <configured>×<configured>分组卷积（ResNeXt核心）
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=_cfg_require('old/GPN_ori.py.ResNeXtBlock.__init__.kernel_size__2'),
                              padding=_cfg_require('old/GPN_ori.py.ResNeXtBlock.__init__.padding'), groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # 第二个<configured>×<configured>卷积
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=_cfg_require('old/GPN_ori.py.ResNeXtBlock.__init__.kernel_size__3'), bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

        # MaxPool放在残差连接之后
        self.maxpool = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_ori.py.ResNeXtBlock.__init__.kernel_size__4'), stride=_cfg_require('old/GPN_ori.py.ResNeXtBlock.__init__.stride'))

        # SE模块（在残差连接之后）
        self.use_se = use_se
        if use_se:
            self.se = SEModule(channels=channels, reduction=se_reduction)

    def forward(self, x):
        identity = x

        # conv1 → BN → ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # conv2 → BN → ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Add + ReLU（论文标准）
        out += identity
        out = self.relu(out)

        # SE模块（在Add+ReLU之后）
        if self.use_se:
            out = self.se(out)

        return out


class GPN(nn.Module):
    '\n    Gaussian Prototype Network (<configured>通道版本)\n    \n    架构:\n    - Stem: <configured>×<configured> conv(<configured>→<configured>) + MaxPool + <configured>×<configured> conv(<configured>→<configured>)\n    - <configured>个相同的ResNeXt Block (<configured>→<configured>)\n    - Channel Half: <configured> → <configured>+<configured>\n    - FC: <configured>→<configured> (v和s各自)\n    \n    总参数: ~<configured> (without SE)\n    '
    def __init__(self, use_se=None):
        use_se = _cfg_resolve('old/GPN_ori.py.GPN.__init__.use_se', use_se)
        super().__init__()

        self.use_se = use_se
        self.channels = _cfg_require('old/GPN_ori.py.GPN.__init__.channels')
        self.cardinality = _cfg_require('old/GPN_ori.py.GPN.__init__.cardinality')

        # ===== STEM =====
        # <configured>×<configured> conv: <configured>→<configured>
        self.stem_conv1 = nn.Conv2d(_cfg_require('old/GPN_ori.py.GPN.__init__.Conv2d_arg0'), _cfg_require('old/GPN_ori.py.GPN.__init__.Conv2d_arg1'), kernel_size=_cfg_require('old/GPN_ori.py.GPN.__init__.kernel_size'), stride=_cfg_require('old/GPN_ori.py.GPN.__init__.stride'),
                                    padding=_cfg_require('old/GPN_ori.py.GPN.__init__.padding'), bias=False)
        self.stem_bn1 = nn.BatchNorm2d(_cfg_require('old/GPN_ori.py.GPN.__init__.BatchNorm2d_arg0'))
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_maxpool = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_ori.py.GPN.__init__.kernel_size__2'), stride=_cfg_require('old/GPN_ori.py.GPN.__init__.stride__2'), padding=_cfg_require('old/GPN_ori.py.GPN.__init__.padding__2'))

        # <configured>×<configured> conv: <configured>→<configured>
        self.stem_conv2 = nn.Conv2d(_cfg_require('old/GPN_ori.py.GPN.__init__.Conv2d_arg0__2'), self.channels, kernel_size=_cfg_require('old/GPN_ori.py.GPN.__init__.kernel_size__3'), bias=False)
        self.stem_bn2 = nn.BatchNorm2d(self.channels)
        # 输出: <configured>×<configured>×<configured>

        self.block1 = ResNeXtBlock(self.channels, self.cardinality,use_se=use_se,se_reduction=_cfg_require('old/GPN_ori.py.GPN.__init__.se_reduction'))  # <configured>×<configured>×<configured>
        self.block2 = ResNeXtBlock(self.channels, self.cardinality,use_se=use_se,se_reduction=_cfg_require('old/GPN_ori.py.GPN.__init__.se_reduction__2'))  # <configured>×<configured>×<configured>
        self.block3 = ResNeXtBlock(self.channels, self.cardinality,use_se=use_se,se_reduction=_cfg_require('old/GPN_ori.py.GPN.__init__.se_reduction__3'))  # <configured>×<configured>×<configured>
        self.block4 = ResNeXtBlock(self.channels, self.cardinality,use_se=use_se,se_reduction=_cfg_require('old/GPN_ori.py.GPN.__init__.se_reduction__4'))  # <configured>×<configured>×<configured>

        # ===== Global Average Pooling =====
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ===== Fully Connected Layers =====
        # Channel Half后每个分支<configured>维
        fc_input_dim = self.channels // 2  # <configured>
        fc_output_dim = _cfg_require('old/GPN_ori.py.GPN.__init__.size_or_budget')  # 调整为<configured>以达到<configured>参数

        self.fc_v = nn.Linear(fc_input_dim, fc_output_dim, bias=False)
        self.fc_s = nn.Linear(fc_input_dim, fc_output_dim, bias=False)

        self._initialize_weights()


    def forward(self, x, return_intermediate=False):
        '\n        前向传播\n        \n        Args:\n            x: [B, <configured>, <configured>, <configured>]\n            return_intermediate: 是否返回中间特征\n            \n        Returns:\n            v: [B, <configured>] embedding feature\n            s: [B, <configured>] precision feature (σ ≥ <configured>)\n            features: (optional) 中间特征字典\n        '
        features = {} if return_intermediate else None

        # ===== Stem =====
        x = self.stem_conv1(x)      # [B, <configured>, <configured>, <configured>]
        x = self.stem_bn1(x)
        x = self.stem_relu(x)
        x = self.stem_maxpool(x)    # [B, <configured>, <configured>, <configured>]

        x = self.stem_conv2(x)      # [B, <configured>, <configured>, <configured>]
        x = self.stem_bn2(x)
        x = self.stem_relu(x)

        if return_intermediate:
            features['stem'] = x

        x = self.block1(x)          # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block1'] = x

        x = self.block2(x)          # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block2'] = x

        x = self.block3(x)          # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block3'] = x

        x = self.block4(x)          # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block4'] = x

        # ===== Channel Half =====
        # 沿通道维度分割: <configured> → <configured>+<configured>
        v_features = x[:, :64, :, :]   # [B, <configured>, <configured>, <configured>]
        s_features = x[:, 64:, :, :]   # [B, <configured>, <configured>, <configured>]

        if return_intermediate:
            features['v_features'] = v_features
            features['s_features'] = s_features

        # ===== Global Average Pool + Flatten =====
        v_pooled = self.avgpool(v_features).flatten(1)  # [B, <configured>]
        s_pooled = self.avgpool(s_features).flatten(1)  # [B, <configured>]

        # ===== Fully Connected =====
        v = self.fc_v(v_pooled)  # [B, <configured>]
        s = self.fc_s(s_pooled)  # [B, <configured>]

        s = 1 + F.softplus(s)

        if return_intermediate:
            return v, s, features
        return v, s

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
