from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
#GPN_change模型迭代

#使用轴注意力机制和模块化设计
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from pathlib import Path
from tqdm import tqdm
from data_loader import *
from loss_Copy1 import GPNLoss,GPNLoss_Advanced
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
    def __init__(self, model,device,loss_feature_dim=None, lr=None,metric_type=None):
        loss_feature_dim = _cfg_resolve('GPN_V2.py.GPNTrainer.__init__.loss_feature_dim', loss_feature_dim)
        lr = _cfg_resolve('GPN_V2.py.GPNTrainer.__init__.lr', lr)
        metric_type = _cfg_resolve('GPN_V2.py.GPNTrainer.__init__.metric_type', metric_type)
        self.device = device
        self.loss_feature_dim = loss_feature_dim

        # 初始化模型
        self.model = model
        self.model.to(device)
        self.loss_fn = GPNLoss_Advanced(loss_feature_dim=loss_feature_dim, metric_type=metric_type)
        self.loss_fn.to(device)

        self.lr=lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"GPN模型初始化完成，设备: {self.device}")


    def save_model(self, save_path=None):
        """
        保存模型和损失函数的所有可学习参数
        """
        save_path = _cfg_resolve('GPN_V2.py.GPNTrainer.save_model.save_path', save_path)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }

        if hasattr(self, 'loss_fn') and hasattr(self.loss_fn, 'state_dict'):
            checkpoint['loss_fn_state_dict'] = self.loss_fn.state_dict()
            # 记录使用的度量类型
            if hasattr(self.loss_fn, 'distance_metric'):
                checkpoint['metric_type'] = self.loss_fn.distance_metric.__class__.__name__

        torch.save(checkpoint, save_path)
        print(f"Model and loss function saved to {save_path}")

    def load_model(self, model_path):
        """
        加载模型和损失函数参数
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'loss_fn_state_dict' in checkpoint:
            if hasattr(self, 'loss_fn') and hasattr(self.loss_fn, 'load_state_dict'):
                self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
                print(f"Loss function parameters loaded")
            else:
                print("Warning: Checkpoint contains loss_fn but current trainer doesn't have compatible loss_fn")

        print(f"Model loaded from {model_path}")

    def compute_gaussian_prototypes(self, support_v, support_s, support_labels, n_ways):
        '\n        计算原型和精度矩阵（论文公式<configured>-<configured>）\n        \n        Args:\n            support_v: 支持集embedding特征 [n_ways * k_shot, feature_dim]\n            support_s: 支持集precision特征 [n_ways * k_shot, feature_dim]\n            support_labels: 支持集标签 [n_ways * k_shot]\n            n_ways: 类别数\n            k_shot: 每类样本数\n            \n        Returns:\n            prototypes: 各类原型 [n_ways, feature_dim]\n            precision_matrices: 各类精度矩阵 [n_ways, feature_dim, feature_dim]\n        '
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('GPN_V2.py.GPNTrainer.train_step_batch.max_norm'))

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
        num_tasks = _cfg_resolve('GPN_V2.py.GPNTrainer.train.num_tasks', num_tasks)
        n_ways = _cfg_resolve('GPN_V2.py.GPNTrainer.train.n_ways', n_ways)
        epochs = _cfg_resolve('GPN_V2.py.GPNTrainer.train.epochs', epochs)
        save_path = _cfg_resolve('GPN_V2.py.GPNTrainer.train.save_path', save_path)
        self.model.train()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=_cfg_require('GPN_V2.py.GPNTrainer.train.step_size'), gamma=_cfg_require('GPN_V2.py.GPNTrainer.train.gamma'))

        print(f"开始元学习训练: {epochs} epochs, {num_tasks} tasks/epoch")

        best_acc=0
        early_stop=0
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
                print(f"当前测试acc:{avg_acc}")

                if avg_acc>best_acc:
                    best_acc=avg_acc
                    self.save_model(save_path)
                    early_stop=0
                else:
                    early_stop+=1
                    print(f"第{early_stop}次未增长")
                    if early_stop>=4:
                        break

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
                # 简洁的调用方式
                distances = self.loss_fn.compute_distance(query_v, prototypes,
                                                          precision_matrices)

                logits = -distances
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == remapped_query_labels).float().mean()
                total_acc += accuracy.item()
        avg_acc = total_acc / len(test_loader)
        return avg_acc
    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """完整的评估协议（符合论文）"""
        n_trials = _cfg_resolve('GPN_V2.py.GPNTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('GPN_V2.py.GPNTrainer.full_evaluation.q_query', q_query)
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
                        batch_size=_cfg_require('GPN_V2.py.GPNTrainer.full_evaluation.batch_size'), num_tasks=_cfg_require('GPN_V2.py.GPNTrainer.full_evaluation.num_tasks'), epochs=_cfg_require('GPN_V2.py.GPNTrainer.full_evaluation.epochs')
                    )
                    loader = DataLoader(meta_test, batch_size=_cfg_require('GPN_V2.py.GPNTrainer.full_evaluation.batch_size__2'))

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
    FEATURE_DIM = _cfg_require('GPN_V2.py.main.FEATURE_DIM')
    N_WAY = _cfg_require('GPN_V2.py.main.N_WAY')
    K_SHOT = _cfg_require('GPN_V2.py.main.K_SHOT')
    Q_QUERY = _cfg_require('GPN_V2.py.main.Q_QUERY')
    NUM_TASKS = _cfg_require('GPN_V2.py.main.NUM_TASKS')
    EPOCHS = _cfg_require('GPN_V2.py.main.EPOCHS')
    LR = _cfg_require('GPN_V2.py.main.LR')
    BATCH_SIZE = _cfg_require('GPN_V2.py.main.BATCH_SIZE')
    SEED = _cfg_require('GPN_V2.py.main.SEED')
    SAVE_MODEL_PATH=_cfg_require('GPN_V2.py.main.SAVE_MODEL_PATH')

    # 数据路径
    dataset_path = _cfg_require('GPN_V2.py.main.dataset_path')

    # 创建原始数据集
    train_dataset=UAVDataset(data_dir_path=dataset_path,seed=SEED,is_train=True,train_ratio=_cfg_require('GPN_V2.py.main.train_ratio'),max_sample_count=_cfg_require('GPN_V2.py.main.max_sample_count'))
    test_dataset=UAVDataset(data_dir_path=dataset_path,seed=SEED,is_train=False,train_ratio=_cfg_require('GPN_V2.py.main.train_ratio__2'),max_sample_count=_cfg_require('GPN_V2.py.main.max_sample_count__2'))

    # 创建元任务划分集
    meta_train_dataset = MetaDataset(train_dataset, N_WAY, K_SHOT, Q_QUERY,batch_size=BATCH_SIZE,num_tasks=NUM_TASKS,epochs=EPOCHS)
    meta_test_dataset = MetaDataset(test_dataset, N_WAY, K_SHOT, Q_QUERY,batch_size=BATCH_SIZE,num_tasks=_cfg_require('GPN_V2.py.main.num_tasks'),epochs=_cfg_require('GPN_V2.py.main.epochs'))

    # DataLoader
    train_loader = DataLoader(meta_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=_cfg_require('GPN_V2.py.main.num_workers'))
    test_loader = DataLoader(meta_test_dataset, batch_size=_cfg_require('GPN_V2.py.main.batch_size'), shuffle=False)

    #创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_without_se = GPN(use_se=_cfg_require('GPN_V2.py.main.use_se'))
    params_without_se = count_parameters(model_without_se)
    # 创建训练器
    meta_learner = GPNTrainer(model_without_se,device=device,feature_dim=FEATURE_DIM,lr=LR)
    print("开始训练...")
    meta_learner.train(train_loader=train_loader,test_loader=test_loader,num_tasks=NUM_TASKS,n_ways=N_WAY,epochs=EPOCHS,save_path=SAVE_MODEL_PATH)

    print("训练完成！")

    meta_learner.full_evaluation(test_dataset=test_dataset,n_trials=_cfg_require('GPN_V2.py.main.n_trials'))

if __name__ == '__main__':
    main()


class DecoupledAxisAttention(nn.Module):
    """
    解耦的轴注意力 - 针对RF信号时频特性

    核心思想：
    - 频率轴：强注意力（物理相关）
    - 时间轴：弱注意力（随机性高）
    """
    def __init__(self, channels, freq_time_ratio=None):
        freq_time_ratio = _cfg_resolve('GPN_V2.py.DecoupledAxisAttention.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()
        self.freq_weight = freq_time_ratio/10.0
        self.time_weight = 1/10.0

        # 频率轴注意力（沿时间维度池化，保留频率信息）
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # [B, C, H, <configured>]
            nn.Conv2d(channels, channels, kernel_size=_cfg_require('GPN_V2.py.DecoupledAxisAttention.__init__.kernel_size'), padding=_cfg_require('GPN_V2.py.DecoupledAxisAttention.__init__.padding')),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # 时间轴注意力（沿频率维度池化，保留时间信息）
        self.time_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # [B, C, <configured>, W]
            nn.Conv2d(channels, channels, kernel_size=_cfg_require('GPN_V2.py.DecoupledAxisAttention.__init__.kernel_size__2'), padding=_cfg_require('GPN_V2.py.DecoupledAxisAttention.__init__.padding__2')),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 频率维度注意力（强）
        freq_att = self.freq_attention(x)  # [B, C, H, <configured>]
        x_freq = x * (1 + self.freq_weight * (freq_att - 0.5))

        # 时间维度注意力（弱）
        time_att = self.time_attention(x_freq)  # [B, C, <configured>, W]
        x_out = x_freq * (1 + self.time_weight * (time_att - 0.5))

        return x_out


class FrequencyPriorityCA(nn.Module):
    """
    频率优先的坐标注意力

    改进点：
    - 频率轴：完整分辨率，大编码器
    - 时间轴：降采样，小编码器
    - 频率权重 >> 时间权重
    """
    def __init__(self, channels, reduction=None, freq_time_ratio=None):
        reduction = _cfg_resolve('GPN_V2.py.FrequencyPriorityCA.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V2.py.FrequencyPriorityCA.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()
        self.freq_time_ratio = freq_time_ratio

        # 频率池化（保留完整信息）
        self.pool_freq = nn.AdaptiveAvgPool2d((None, 1))

        # 时间池化（降采样，减少随机性）
        self.pool_time = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.AvgPool2d(_cfg_require('GPN_V2.py.FrequencyPriorityCA.__init__.AvgPool2d_arg0'), stride=_cfg_require('GPN_V2.py.FrequencyPriorityCA.__init__.stride'))  # 额外<configured>倍降采样
        )

        # 频率编码器（大容量）
        freq_reduced = max(1, channels // reduction)
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(channels, freq_reduced, _cfg_require('GPN_V2.py.FrequencyPriorityCA.__init__.Conv2d_arg2__3')),
            nn.BatchNorm2d(freq_reduced),
            nn.SiLU()
        )

        # 时间编码器（小容量）
        time_reduced = max(1, channels // (reduction * 4))
        self.time_encoder = nn.Sequential(
            nn.Conv2d(channels, time_reduced, _cfg_require('GPN_V2.py.FrequencyPriorityCA.__init__.Conv2d_arg2__4')),
            nn.BatchNorm2d(time_reduced),
            nn.SiLU()
        )

        # 解码器
        self.freq_decoder = nn.Conv2d(freq_reduced, channels, _cfg_require('GPN_V2.py.FrequencyPriorityCA.__init__.Conv2d_arg2'))
        self.time_decoder = nn.Conv2d(time_reduced, channels, _cfg_require('GPN_V2.py.FrequencyPriorityCA.__init__.Conv2d_arg2__2'))

    def forward(self, x):
        B, C, H, W = x.shape

        # 频率维度处理（完整分辨率）
        x_freq = self.pool_freq(x)  # [B, C, H, <configured>]
        x_freq = self.freq_encoder(x_freq)
        freq_att = self.freq_decoder(x_freq).sigmoid()  # [B, C, H, <configured>]

        # 时间维度处理（降采样）
        x_time = self.pool_time(x)  # [B, C, <configured>, W//<configured>]
        x_time = self.time_encoder(x_time)
        time_att = self.time_decoder(x_time)  # [B, C, <configured>, W//<configured>]

        # 上采样回原尺寸（4D张量用bilinear）
        time_att = F.interpolate(time_att, size=(1, W),
                                mode='bilinear', align_corners=False)
        time_att = time_att.sigmoid()

        # 加权融合（频率权重更大）
        out = x * (freq_att * self.freq_time_ratio + time_att)

        return out


class RepVGGBlock(nn.Module):
    """
    RepVGG块 - 训练推理解耦

    训练时：3x3 + 1x1 + Identity（多分支）
    推理时：重参数化为单个3x3卷积

    注意力机制：SE / decoupled / freq_priority 三选一
    """
    def __init__(self, in_channels, out_channels, stride=None,
                 attention_type=None, reduction=None,freq_time_ratio=None):
        """
        Args:
            attention_type: 'se' / 'decoupled' / 'freq_priority' / None
            reduction: SE模块的reduction ratio
        """
        stride = _cfg_resolve('GPN_V2.py.RepVGGBlock.__init__.stride', stride)
        attention_type = _cfg_resolve('GPN_V2.py.RepVGGBlock.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V2.py.RepVGGBlock.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V2.py.RepVGGBlock.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 3x3卷积分支
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('GPN_V2.py.RepVGGBlock.__init__.kernel_size'),
                     stride=stride, padding=_cfg_require('GPN_V2.py.RepVGGBlock.__init__.padding'), bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 1x1卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('GPN_V2.py.RepVGGBlock.__init__.kernel_size__2'),
                     stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Identity分支
        self.identity = nn.BatchNorm2d(in_channels) \
            if in_channels == out_channels and stride == 1 else None

        self.act = nn.SiLU(inplace=True)

        # 注意力机制（三选一）
        if attention_type == 'se':
            self.attention = SEModule(out_channels, reduction=reduction)
        elif attention_type == 'decoupled':
            self.attention = DecoupledAxisAttention(out_channels,freq_time_ratio)
        elif attention_type == 'freq_priority':
            self.attention = FrequencyPriorityCA(out_channels,reduction, freq_time_ratio)
        else:
            self.attention = None

    def forward(self, x):
        if self.training:
            # 训练模式：多分支
            out = self.conv3x3(x) + self.conv1x1(x)
            if self.identity is not None:
                out += self.identity(x)
        else:
            # 推理模式：单分支（需要先调用switch_to_deploy）
            out = self.conv3x3(x)

        out = self.act(out)

        if self.attention is not None:
            out = self.attention(out)

        return out


class DropPath(nn.Module):
    """DropPath正则化"""
    def __init__(self, drop_prob=None):
        drop_prob = _cfg_resolve('GPN_V2.py.DropPath.__init__.drop_prob', drop_prob)
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt风格的Block

    架构：DWConv 7x7 → LayerNorm → 1x1 Conv(4x) → GELU → 1x1 Conv → DropPath

    注意力机制：SE / decoupled / freq_priority 三选一
    """
    def __init__(self, in_channels, out_channels, stride=None,
                 expansion=None, drop_path=None,
                 attention_type=None, reduction=None,freq_time_ratio=None):
        """
        Args:
            attention_type: 'se' / 'decoupled' / 'freq_priority' / None
            reduction: SE模块的reduction ratio
        """
        stride = _cfg_resolve('GPN_V2.py.ConvNeXtBlock.__init__.stride', stride)
        expansion = _cfg_resolve('GPN_V2.py.ConvNeXtBlock.__init__.expansion', expansion)
        drop_path = _cfg_resolve('GPN_V2.py.ConvNeXtBlock.__init__.drop_path', drop_path)
        attention_type = _cfg_resolve('GPN_V2.py.ConvNeXtBlock.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V2.py.ConvNeXtBlock.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V2.py.ConvNeXtBlock.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()

        # 如果stride><configured>，使用下采样
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=_cfg_require('GPN_V2.py.ConvNeXtBlock.__init__.kernel_size__3'),
                         stride=_cfg_require('GPN_V2.py.ConvNeXtBlock.__init__.stride__2'), groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('GPN_V2.py.ConvNeXtBlock.__init__.kernel_size__4'), bias=False),
            )
            in_channels = out_channels

        # 大kernel深度卷积
        self.dwconv = nn.Conv2d(in_channels, in_channels,
                               kernel_size=_cfg_require('GPN_V2.py.ConvNeXtBlock.__init__.kernel_size'), padding=_cfg_require('GPN_V2.py.ConvNeXtBlock.__init__.padding'),
                               groups=in_channels, bias=False)

        # LayerNorm (通道维度)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

        # 反向瓶颈
        hidden_channels = int(in_channels * expansion)
        self.pwconv1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_channels, out_channels)

        # 注意力机制（三选一）
        if attention_type == 'se':
            self.attention = SEModule(out_channels, reduction=reduction)
        elif attention_type == 'decoupled':
            self.attention = DecoupledAxisAttention(out_channels,freq_time_ratio)
        elif attention_type == 'freq_priority':
            self.attention = FrequencyPriorityCA(out_channels,reduction, freq_time_ratio)
        else:
            self.attention = None

        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels and stride == 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                     kernel_size=_cfg_require('GPN_V2.py.ConvNeXtBlock.__init__.kernel_size__2'), bias=False)

    def forward(self, x):
        # 下采样
        if self.downsample is not None:
            x = self.downsample(x)

        identity = self.shortcut(x)

        # 深度卷积
        out = self.dwconv(x)

        # LayerNorm: [B,C,H,W] -> [B,H,W,C]
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)

        # 反向瓶颈
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        # 转回 [B,C,H,W]
        out = out.permute(0, 3, 1, 2)

        # 注意力
        if self.attention is not None:
            out = self.attention(out)

        # DropPath + Residual
        out = identity + self.drop_path(out)

        return out


# ==================== 兼容的SE模块（保留原有接口） ====================

class SEModule(nn.Module):
    """原始SE模块 - 保持兼容性"""
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
    '\n    ResNeXt Bottleneck Block\n    \n    标准ResNeXt结构: <configured>×<configured>扩展 → <configured>×<configured>分组卷积 → <configured>×<configured>压缩\n    \n    注意力机制：SE / decoupled / freq_priority 三选一\n    '
    def __init__(self, in_channels, out_channels, stride=None,
                 cardinality=None, attention_type=None, reduction=None,freq_time_ratio=None):
        """
        Args:
            cardinality: 分组数
            attention_type: 'se' / 'decoupled' / 'freq_priority' / None
            reduction: SE模块的reduction ratio
        """
        stride = _cfg_resolve('GPN_V2.py.ResNeXtBlock.__init__.stride', stride)
        cardinality = _cfg_resolve('GPN_V2.py.ResNeXtBlock.__init__.cardinality', cardinality)
        attention_type = _cfg_resolve('GPN_V2.py.ResNeXtBlock.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V2.py.ResNeXtBlock.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V2.py.ResNeXtBlock.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()

        mid_channels = out_channels

        # 1x1 扩展卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=_cfg_require('GPN_V2.py.ResNeXtBlock.__init__.kernel_size'), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=_cfg_require('GPN_V2.py.ResNeXtBlock.__init__.kernel_size__2'), stride=stride,
                              padding=_cfg_require('GPN_V2.py.ResNeXtBlock.__init__.padding'), groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 压缩卷积
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                              kernel_size=_cfg_require('GPN_V2.py.ResNeXtBlock.__init__.kernel_size__3'), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.SiLU(inplace=True)

        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=_cfg_require('GPN_V2.py.ResNeXtBlock.__init__.kernel_size__4'), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 注意力机制（三选一）
        if attention_type == 'se':
            self.attention = SEModule(out_channels, reduction=reduction)
        elif attention_type == 'decoupled':
            self.attention = DecoupledAxisAttention(out_channels,freq_time_ratio)
        elif attention_type == 'freq_priority':
            self.attention = FrequencyPriorityCA(out_channels,reduction,freq_time_ratio)
        else:
            self.attention = None

    def forward(self, x):
        identity = self.shortcut(x)

        # Bottleneck路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 残差连接
        out += identity
        out = self.relu(out)

        # 注意力模块（在Add+ReLU之后）
        if self.attention is not None:
            out = self.attention(out)

        return out

class GPN_Optimized(nn.Module):
    "\n    优化的GPN模型 - 完全兼容现有训练器\n    \n    架构特点：\n    - Stage <configured>-<configured>: 可选RepVGG（浅层，训练稳定）\n    - Stage <configured>-<configured>: 可选ConvNeXt（深层，大感受野）\n    - 全局：统一的注意力机制（SE / decoupled / freq_priority）\n    - 输出：v (embedding), s (precision) - 与原模型一致\n    \n    参数说明：\n        use_repvgg: Stage <configured>-<configured>是否使用RepVGG（默认False，使用ResNeXt）\n        use_convnext: Stage <configured>-<configured>是否使用ConvNeXt（默认False，使用ResNeXt）\n        attention_type: 注意力类型\n            - 'se': 标准SE模块（默认）\n            - 'decoupled': 解耦时频注意力\n            - 'freq_priority': 频率优先CA\n            - None: 不使用注意力\n        reduction: SE模块的reduction ratio（由外部配置提供）\n    "
    def __init__(self,
                 use_repvgg=None,
                 use_convnext=None,
                 attention_type=None,
                 reduction=None,
                 freq_time_ratio=None):
        use_repvgg = _cfg_resolve('GPN_V2.py.GPN_Optimized.__init__.use_repvgg', use_repvgg)
        use_convnext = _cfg_resolve('GPN_V2.py.GPN_Optimized.__init__.use_convnext', use_convnext)
        attention_type = _cfg_resolve('GPN_V2.py.GPN_Optimized.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V2.py.GPN_Optimized.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V2.py.GPN_Optimized.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()

        self.use_repvgg = use_repvgg
        self.use_convnext = use_convnext
        self.attention_type = attention_type

        # === Stem（与原模型保持一致） ===
        self.conv1 = nn.Conv2d(_cfg_require('GPN_V2.py.GPN_Optimized.__init__.Conv2d_arg0'), _cfg_require('GPN_V2.py.GPN_Optimized.__init__.Conv2d_arg1'), kernel_size=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.kernel_size'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride'),
                              padding=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.padding'), bias=False)
        self.bn1 = nn.BatchNorm2d(_cfg_require('GPN_V2.py.GPN_Optimized.__init__.BatchNorm2d_arg0'))
        self.relu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.kernel_size__2'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__2'), padding=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.padding__2'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_repvgg:
            self.block1 = RepVGGBlock(
                24, 48, stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__7'),
                attention_type=attention_type,
                reduction=reduction,
                freq_time_ratio=freq_time_ratio
            )
        else:
            self.block1 = ResNeXtBlock(
                _cfg_require('GPN_V2.py.GPN_Optimized.__init__.ResNeXtBlock_arg0'), _cfg_require('GPN_V2.py.GPN_Optimized.__init__.ResNeXtBlock_arg1'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__8'), cardinality=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.cardinality'),
                attention_type=attention_type,
                reduction=48 if attention_type == 'se' else reduction,
                freq_time_ratio=freq_time_ratio
            )
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.kernel_size__3'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__3'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_repvgg:
            self.block2 = RepVGGBlock(
                48, 96, stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__9'),
                attention_type=attention_type,
                reduction=reduction,
                freq_time_ratio=freq_time_ratio
            )
        else:
            self.block2 = ResNeXtBlock(
                _cfg_require('GPN_V2.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__2'), _cfg_require('GPN_V2.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__2'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__10'), cardinality=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.cardinality__2'),
                attention_type=attention_type,
                reduction=32 if attention_type == 'se' else reduction,
                freq_time_ratio=freq_time_ratio
            )
        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.kernel_size__4'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__4'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_convnext:
            self.block3 = ConvNeXtBlock(
                96, _cfg_require('GPN_V2.py.GPN_Optimized.__init__.size_or_budget'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__11'),
                expansion=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.expansion'), drop_path=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.drop_path'),
                attention_type=attention_type,
                reduction=reduction,
                freq_time_ratio=freq_time_ratio
            )
        else:
            self.block3 = ResNeXtBlock(
                _cfg_require('GPN_V2.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__3'), _cfg_require('GPN_V2.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__3'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__12'), cardinality=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.cardinality__3'),
                attention_type=attention_type,
                reduction=24 if attention_type == 'se' else reduction,
                freq_time_ratio=freq_time_ratio
            )
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.kernel_size__5'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__5'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_convnext:
            self.block4 = ConvNeXtBlock(
                _cfg_require('GPN_V2.py.GPN_Optimized.__init__.size_or_budget__2'), _cfg_require('GPN_V2.py.GPN_Optimized.__init__.size_or_budget__3'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__13'),
                expansion=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.expansion__2'), drop_path=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.drop_path__2'),
                attention_type=attention_type,
                reduction=reduction,
                freq_time_ratio=freq_time_ratio
            )
        else:
            self.block4 = ResNeXtBlock(
                _cfg_require('GPN_V2.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__4'), _cfg_require('GPN_V2.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__4'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__14'), cardinality=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.cardinality__4'),
                attention_type=attention_type,
                reduction=80 if attention_type == 'se' else reduction,
                freq_time_ratio=freq_time_ratio
            )
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.kernel_size__6'), stride=_cfg_require('GPN_V2.py.GPN_Optimized.__init__.stride__6'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # === Global Pooling ===
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def forward(self, x, return_intermediate=False):
        '\n        前向传播 - 对齐原模型接口\n        \n        Args:\n            x: [B, <configured>, <configured>, <configured>]\n            return_intermediate: 是否返回中间特征\n            \n        Returns:\n            v: [B, <configured>] embedding特征\n            s: [B, <configured>] precision特征\n            features: (可选) 中间特征字典\n        '
        features = {} if return_intermediate else None

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, <configured>, <configured>, <configured>]

        # Stage <configured>-<configured>
        x = self.block1(x)
        x = self.pool1(x)  # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block1'] = x

        x = self.block2(x)
        x = self.pool2(x)  # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block2'] = x

        x = self.block3(x)
        x = self.pool3(x)  # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block3'] = x

        x = self.block4(x)
        x = self.pool4(x)  # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block4'] = x

        # Channel Split: v和s
        v_features = x[:, :_cfg_require('GPN_V2.py.GPN_Optimized.forward.size_or_budget'), :, :]  # [B, <configured>, <configured>, <configured>]
        s_features = x[:, _cfg_require('GPN_V2.py.GPN_Optimized.forward.size_or_budget__2'):, :, :]  # [B, <configured>, <configured>, <configured>]

        if return_intermediate:
            features['v_features'] = v_features
            features['s_features'] = s_features

        # Global Average Pooling
        v = self.avgpool(v_features).flatten(1)  # [B, <configured>]
        s = self.avgpool(s_features).flatten(1)  # [B, <configured>]

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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
