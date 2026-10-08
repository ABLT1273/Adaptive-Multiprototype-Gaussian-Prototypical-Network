from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
#GPN模型迭代：在原模型基础上改进

#使用SiLU替代ReLU
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
        loss_feature_dim = _cfg_resolve('old/GPN_change.py.GPNTrainer.__init__.loss_feature_dim', loss_feature_dim)
        lr = _cfg_resolve('old/GPN_change.py.GPNTrainer.__init__.lr', lr)
        metric_type = _cfg_resolve('old/GPN_change.py.GPNTrainer.__init__.metric_type', metric_type)
        self.device = device
        self.feature_dim = loss_feature_dim

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
        save_path = _cfg_resolve('old/GPN_change.py.GPNTrainer.save_model.save_path', save_path)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'feature_dim': self.feature_dim,
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('old/GPN_change.py.GPNTrainer.train_step_batch.max_norm'))

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
        num_tasks = _cfg_resolve('old/GPN_change.py.GPNTrainer.train.num_tasks', num_tasks)
        n_ways = _cfg_resolve('old/GPN_change.py.GPNTrainer.train.n_ways', n_ways)
        epochs = _cfg_resolve('old/GPN_change.py.GPNTrainer.train.epochs', epochs)
        save_path = _cfg_resolve('old/GPN_change.py.GPNTrainer.train.save_path', save_path)
        self.model.train()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=_cfg_require('old/GPN_change.py.GPNTrainer.train.step_size'), gamma=_cfg_require('old/GPN_change.py.GPNTrainer.train.gamma'))

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

            if (epoch+1)%3==0:
                avg_acc=self.evaluate(test_loader=test_loader,n_ways=n_ways)
                print(f"当前测试acc:{avg_acc}")

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
                distances = self.loss_fn.distance_metric(
                    query_v, prototypes
                )

                logits = -distances
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == remapped_query_labels).float().mean()
                total_acc += accuracy.item()

        avg_acc = total_acc / len(test_loader)
        return avg_acc
    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """完整的评估协议（符合论文）"""
        n_trials = _cfg_resolve('old/GPN_change.py.GPNTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('old/GPN_change.py.GPNTrainer.full_evaluation.q_query', q_query)
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
                        batch_size=_cfg_require('old/GPN_change.py.GPNTrainer.full_evaluation.batch_size'), num_tasks=_cfg_require('old/GPN_change.py.GPNTrainer.full_evaluation.num_tasks'), epochs=_cfg_require('old/GPN_change.py.GPNTrainer.full_evaluation.epochs')
                    )
                    loader = DataLoader(meta_test, batch_size=_cfg_require('old/GPN_change.py.GPNTrainer.full_evaluation.batch_size__2'))

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
    FEATURE_DIM = _cfg_require('old/GPN_change.py.main.FEATURE_DIM')
    N_WAY = _cfg_require('old/GPN_change.py.main.N_WAY')
    K_SHOT = _cfg_require('old/GPN_change.py.main.K_SHOT')
    Q_QUERY = _cfg_require('old/GPN_change.py.main.Q_QUERY')
    NUM_TASKS = _cfg_require('old/GPN_change.py.main.NUM_TASKS')
    EPOCHS = _cfg_require('old/GPN_change.py.main.EPOCHS')
    LR = _cfg_require('old/GPN_change.py.main.LR')
    BATCH_SIZE = _cfg_require('old/GPN_change.py.main.BATCH_SIZE')
    SEED = _cfg_require('old/GPN_change.py.main.SEED')
    SAVE_MODEL_PATH=_cfg_require('old/GPN_change.py.main.SAVE_MODEL_PATH')

    # 数据路径
    dataset_path = _cfg_require('old/GPN_change.py.main.dataset_path')

    # 创建原始数据集
    train_dataset=UAVDataset(data_dir_path=dataset_path,seed=SEED,is_train=True,train_ratio=_cfg_require('old/GPN_change.py.main.train_ratio'),max_sample_count=_cfg_require('old/GPN_change.py.main.max_sample_count'))
    test_dataset=UAVDataset(data_dir_path=dataset_path,seed=SEED,is_train=False,train_ratio=_cfg_require('old/GPN_change.py.main.train_ratio__2'),max_sample_count=_cfg_require('old/GPN_change.py.main.max_sample_count__2'))

    # 创建元任务划分集
    meta_train_dataset = MetaDataset(train_dataset, N_WAY, K_SHOT, Q_QUERY,batch_size=BATCH_SIZE,num_tasks=NUM_TASKS,epochs=EPOCHS)
    meta_test_dataset = MetaDataset(test_dataset, N_WAY, K_SHOT, Q_QUERY,batch_size=BATCH_SIZE,num_tasks=_cfg_require('old/GPN_change.py.main.num_tasks'),epochs=_cfg_require('old/GPN_change.py.main.epochs'))

    # DataLoader
    train_loader = DataLoader(meta_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=_cfg_require('old/GPN_change.py.main.num_workers'))
    test_loader = DataLoader(meta_test_dataset, batch_size=_cfg_require('old/GPN_change.py.main.batch_size'), shuffle=False)

    #创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_without_se = GPN(use_se=_cfg_require('old/GPN_change.py.main.use_se'))
    params_without_se = count_parameters(model_without_se)
    # 创建训练器
    meta_learner = GPNTrainer(model_without_se,device=device,feature_dim=FEATURE_DIM,lr=LR)
    print("开始训练...")
    meta_learner.train(train_loader=train_loader,test_loader=test_loader,num_tasks=NUM_TASKS,n_ways=N_WAY,epochs=EPOCHS,save_path=SAVE_MODEL_PATH)

    print("训练完成！")

    meta_learner.full_evaluation(test_dataset=test_dataset,n_trials=_cfg_require('old/GPN_change.py.main.n_trials'))

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
    '\n    ResNeXt Bottleneck Block with SE\n    标准ResNeXt结构: <configured>×<configured>扩展 → <configured>×<configured>分组卷积 → <configured>×<configured>压缩\n    '
    def __init__(self, in_channels, out_channels, stride=None,
                 cardinality=None, use_se=None, se_reduction=None):
        stride = _cfg_resolve('old/GPN_change.py.ResNeXtBlock.__init__.stride', stride)
        cardinality = _cfg_resolve('old/GPN_change.py.ResNeXtBlock.__init__.cardinality', cardinality)
        use_se = _cfg_resolve('old/GPN_change.py.ResNeXtBlock.__init__.use_se', use_se)
        se_reduction = _cfg_resolve('old/GPN_change.py.ResNeXtBlock.__init__.se_reduction', se_reduction)
        super().__init__()

        # 中间通道数（bottleneck width）
        # 通常设置为out_channels的<configured>/<configured>或其他比例
        mid_channels = out_channels

        # 1x1 扩展卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=_cfg_require('old/GPN_change.py.ResNeXtBlock.__init__.kernel_size'), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=_cfg_require('old/GPN_change.py.ResNeXtBlock.__init__.kernel_size__2'), stride=stride,
                              padding=_cfg_require('old/GPN_change.py.ResNeXtBlock.__init__.padding'), groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 压缩卷积
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                              kernel_size=_cfg_require('old/GPN_change.py.ResNeXtBlock.__init__.kernel_size__3'), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.SiLU(inplace=True)

        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=_cfg_require('old/GPN_change.py.ResNeXtBlock.__init__.kernel_size__4'), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # SE模块（在残差连接之后）
        self.use_se = use_se
        if use_se:
            self.se = SEModule(out_channels, reduction=se_reduction)

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

        # SE模块（在Add+ReLU之后）
        if self.use_se:
            out = self.se(out)

        return out

class GPN(nn.Module):
    '\n    高斯原型网络 (Gaussian Prototype Network)\n    \n    架构参数（符合论文<configured>要求）:\n    - Initial: <configured>→<configured> (<configured>×<configured> conv)\n    - BLOCK-<configured>: <configured>→<configured> (SE r=<configured>)\n    - BLOCK-<configured>: <configured>→<configured> (SE r=<configured>) \n    - BLOCK-<configured>: <configured>→<configured> (SE r=<configured>)\n    - BLOCK-<configured>: <configured>→<configured> (SE r=<configured>)\n    - Channel Half: <configured> → <configured>(v) + <configured>(s)\n    \n    总参数: ~<configured> (with SE), ~<configured> (without SE)\n    '
    def __init__(self, use_se=None):
        use_se = _cfg_resolve('old/GPN_change.py.GPN.__init__.use_se', use_se)
        super().__init__()

        self.use_se = use_se

        # Initial <configured>×<configured> 卷积层
        self.conv1 = nn.Conv2d(_cfg_require('old/GPN_change.py.GPN.__init__.Conv2d_arg0'), _cfg_require('old/GPN_change.py.GPN.__init__.Conv2d_arg1'), kernel_size=_cfg_require('old/GPN_change.py.GPN.__init__.kernel_size'), stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride'),
                              padding=_cfg_require('old/GPN_change.py.GPN.__init__.padding'), bias=False)
        self.bn1 = nn.BatchNorm2d(_cfg_require('old/GPN_change.py.GPN.__init__.BatchNorm2d_arg0'))
        self.relu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_change.py.GPN.__init__.kernel_size__2'), stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__2'), padding=_cfg_require('old/GPN_change.py.GPN.__init__.padding__2'))
        # 输出: <configured>×<configured>×<configured>

        self.block1 = ResNeXtBlock(
            in_channels=_cfg_require('old/GPN_change.py.GPN.__init__.in_channels'),
            out_channels=_cfg_require('old/GPN_change.py.GPN.__init__.out_channels'),
            stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__3'),
            cardinality=_cfg_require('old/GPN_change.py.GPN.__init__.cardinality'),
            use_se=use_se,
            se_reduction=_cfg_require('old/GPN_change.py.GPN.__init__.se_reduction')
        )
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_change.py.GPN.__init__.kernel_size__3'), stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__4'))
        # 输出: <configured>×<configured>×<configured>

        self.block2 = ResNeXtBlock(
            in_channels=_cfg_require('old/GPN_change.py.GPN.__init__.in_channels__2'),
            out_channels=_cfg_require('old/GPN_change.py.GPN.__init__.out_channels__2'),
            stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__5'),
            cardinality=_cfg_require('old/GPN_change.py.GPN.__init__.cardinality__2'),
            use_se=use_se,
            se_reduction=_cfg_require('old/GPN_change.py.GPN.__init__.se_reduction__2')
        )
        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_change.py.GPN.__init__.kernel_size__4'), stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__6'))
        # 输出: <configured>×<configured>×<configured>

        self.block3 = ResNeXtBlock(
            in_channels=_cfg_require('old/GPN_change.py.GPN.__init__.in_channels__3'),
            out_channels=_cfg_require('old/GPN_change.py.GPN.__init__.out_channels__3'),
            stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__7'),
            cardinality=_cfg_require('old/GPN_change.py.GPN.__init__.cardinality__3'),
            use_se=use_se,
            se_reduction=_cfg_require('old/GPN_change.py.GPN.__init__.se_reduction__3')  # r≈<configured>
        )
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_change.py.GPN.__init__.kernel_size__5'), stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__8'))
        # 输出: <configured>×<configured>×<configured>

        self.block4 = ResNeXtBlock(
            in_channels=_cfg_require('old/GPN_change.py.GPN.__init__.in_channels__4'),
            out_channels=_cfg_require('old/GPN_change.py.GPN.__init__.out_channels__4'),
            stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__9'),
            cardinality=_cfg_require('old/GPN_change.py.GPN.__init__.cardinality__4'),
            use_se=use_se,
            se_reduction=_cfg_require('old/GPN_change.py.GPN.__init__.se_reduction__4')  # r≈<configured>
        )
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_change.py.GPN.__init__.kernel_size__6'), stride=_cfg_require('old/GPN_change.py.GPN.__init__.stride__10'))
        # 输出: <configured>×<configured>×<configured>

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def forward(self, x, return_intermediate=False):
        '\n        前向传播\n        \n        Args:\n            x: 输入 [B, <configured>, <configured>, <configured>]\n            return_intermediate: 是否返回中间特征\n            \n        Returns:\n            v: embedding feature [B, <configured>]\n            s: precision feature [B, <configured>]\n            features: (可选) 中间特征字典\n        '
        features = {} if return_intermediate else None

        # Stem
        x = self.conv1(x)       # [B, <configured>, <configured>, <configured>]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, <configured>, <configured>, <configured>]

        # <configured>个ResNeXt Blocks
        x = self.block1(x)      # [B, <configured>, <configured>, <configured>]
        x = self.pool1(x)       # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block1'] = x

        x = self.block2(x)      # [B, <configured>, <configured>, <configured>]
        x = self.pool2(x)       # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block2'] = x

        x = self.block3(x)      # [B, <configured>, <configured>, <configured>]
        x = self.pool3(x)       # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block3'] = x

        x = self.block4(x)      # [B, <configured>, <configured>, <configured>]
        x = self.pool4(x)       # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block4'] = x

        # Channel Half: 沿通道维度分割
        # 方案<configured>: 直接split（无额外参数）
        v_features = x[:, :_cfg_require('old/GPN_change.py.GPN.forward.size_or_budget'), :, :]   # [B, <configured>, <configured>, <configured>]
        s_features = x[:, _cfg_require('old/GPN_change.py.GPN.forward.size_or_budget__2'):, :, :]   # [B, <configured>, <configured>, <configured>]

        if return_intermediate:
            features['v_features'] = v_features
            features['s_features'] = s_features

        # GlobalAvgPool + Flatten
        v = self.avgpool(v_features).flatten(1)  # [B, <configured>]
        s = self.avgpool(s_features).flatten(1)  # [B, <configured>]

        # 确保s为正值（论文Eq.<configured>要求）
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MetaSGDTrainer(GPNTrainer):
    """
    简化的Meta-SGD实现，直接优化学习率参数
    """
    def __init__(self, model, device, loss_feature_dim=None, base_lr=None, meta_lr=None, l1_lambda=None,use_parent_train=True,
                loss_alpha=None, sinkhorn_eps=None, sinkhorn_iter=None
                ):
        loss_feature_dim = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.__init__.loss_feature_dim', loss_feature_dim)
        base_lr = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.__init__.base_lr', base_lr)
        meta_lr = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.__init__.meta_lr', meta_lr)
        l1_lambda = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.__init__.l1_lambda', l1_lambda)
        loss_alpha = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.__init__.loss_alpha', loss_alpha)
        sinkhorn_eps = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.__init__.sinkhorn_eps', sinkhorn_eps)
        sinkhorn_iter = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.__init__.sinkhorn_iter', sinkhorn_iter)
        super().__init__(model=model, device=device, loss_feature_dim=loss_feature_dim, lr=base_lr)
        self.use_parent_train=use_parent_train
        self.meta_lr = meta_lr
        self.base_lr = base_lr

        # 为每个模型参数创建对应的学习率参数
        self.alphas = nn.ParameterDict()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 直接学习正值学习率，使用较小的初始值
                self.alphas[name.replace('.', '_')] = nn.Parameter(
                    torch.full_like(param, self.base_lr)
                )

        # 只优化学习率参数
        self.meta_optimizer = torch.optim.Adam(self.alphas.values(), lr=self.meta_lr)

        print(f"初始化了 {len(self.alphas)} 个学习率参数")
        print(f"初始平均学习率: {torch.mean(torch.stack([torch.mean(p) for p in self.alphas.values()])):.6f}")

    def meta_train_step(self, meta_batch):
        """
        使用 higher 库实现 Meta-SGD，修复计算图断开的问题。
        """
        self.model.train()
        batch_size = len(meta_batch[0])

        total_meta_loss = 0
        total_acc = 0

        self.meta_optimizer.zero_grad() # 清零 alpha/beta 的梯度

        for i in range(batch_size):
            support_signals, support_labels = meta_batch[0][i], meta_batch[1][i]
            query_signals, query_labels = meta_batch[2][i], meta_batch[3][i]

            # 移动到设备和标签映射（与原代码相同）
            support_signals = support_signals.to(self.device).float().unsqueeze(1)
            support_labels = support_labels.to(self.device)
            query_signals = query_signals.to(self.device).float().unsqueeze(1)
            query_labels = query_labels.to(self.device)

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

            # <configured>. 核心修复：使用 higher 创建可微分的模型 fmodel
            with higher.innerloop_ctx(
                self.model,
                torch.optim.SGD(self.model.parameters(), lr=_cfg_require('old/GPN_change.py.MetaSGDTrainer.meta_train_step.lr')), # 优化器占位符
                copy_initial_weights=False,
                track_higher_grads=True # 必须为 True，确保外循环梯度能回传
            ) as (fmodel, diffopt):

                # 内循环：快速适应
                for step in range(5):
                    # 前向传播 and 计算 L_support
                    support_v, support_s = fmodel(support_signals)
                    prototypes, precision_matrices = self.compute_gaussian_prototypes(
                        support_v, support_s, remapped_support_labels, n_ways
                    )
                    support_loss, _, _ = self.loss_fn(
                        support_v, prototypes, precision_matrices, remapped_support_labels
                    )

                    grads = torch.autograd.grad(
                        support_loss,
                        fmodel.parameters(),
                        create_graph=True, # 必须为 True，确保梯度本身可导
                        allow_unused=True
                    )

                    # <configured>. 使用学习到的 alpha 进行可微分参数更新
                    adapted_params_list = []

                    for param_idx, (name, param) in enumerate(fmodel.named_parameters()):
                        grad = grads[param_idx]
                        alpha_name = name.replace('.', '_')

                        # 获取 alpha (并使用 clamp 限制范围)
                        alpha = self.alphas.get(alpha_name)
                        if alpha is not None:
                            alpha = torch.clamp(alpha, min=1e-6)

                        if grad is not None and alpha is not None:
                            # 核心：param - alpha * grad。由于 param 和 alpha 都是张量
                            # 且在 higher 环境中，此操作会保留计算图
                            adapted_param = param - alpha * grad
                        else:
                            adapted_param = param

                        adapted_params_list.append(adapted_param)

                    # <configured>. 覆盖 fmodel 的参数，继续内循环
                    fmodel.update_params(adapted_params_list)

                # 外循环：查询集评估
                query_v, query_s = fmodel(query_signals)
                # 必须再次前向传播支持集特征以计算原型
                support_v_eval, support_s_eval = fmodel(support_signals)

                prototypes, precision_matrices = self.compute_gaussian_prototypes(
                    support_v_eval, support_s_eval, remapped_support_labels, n_ways
                )

                meta_loss, probabilities, _ = self.loss_fn(
                    query_v, prototypes, precision_matrices, remapped_query_labels
                )

                (meta_loss / batch_size).backward()

                total_meta_loss += meta_loss.item()

                # 计算准确率
                predictions = torch.argmin(probabilities, dim=1)
                accuracy = (predictions == remapped_query_labels).float().mean()
                total_acc += accuracy.item()

        # 梯度裁剪和参数更新 (只针对 self.alphas)
        torch.nn.utils.clip_grad_norm_(self.alphas.values(), max_norm=_cfg_require('old/GPN_change.py.MetaSGDTrainer.meta_train_step.max_norm'))
        self.meta_optimizer.step()

        # ... (打印梯度检查信息，可以保留作为调试辅助) ...
        grad_count = 0
        total_grad_norm = 0
        for name, alpha in self.alphas.items():
            if alpha.grad is not None:
                grad_count += 1
                total_grad_norm += alpha.grad.norm().item()

        if grad_count == 0:
            print("警告：学习率参数没有接收到梯度！(请检查 higher 和 CUDA 设置)")
        else:
            avg_grad_norm = total_grad_norm / grad_count


        return total_meta_loss / batch_size, total_acc / batch_size
    def train(self, train_loader, test_loader,num_tasks=None, n_ways=None, epochs=None,save_path=None):

        num_tasks = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.train.num_tasks', num_tasks)
        n_ways = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.train.n_ways', n_ways)
        epochs = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.train.epochs', epochs)
        save_path = _cfg_resolve('old/GPN_change.py.MetaSGDTrainer.train.save_path', save_path)
        if self.use_parent_train:
            # 调用父类方法
            return super().train(train_loader,test_loader, num_tasks, n_ways, epochs)
        else:
            """
            Meta-SGD 训练循环
            """
            self.model.train()
            print(f"开始Meta-SGD元学习训练: {epochs} epochs, {num_tasks} tasks/epoch")

            # 使用 Meta Optimizer 的 scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimizer, step_size=_cfg_require('old/GPN_change.py.MetaSGDTrainer.train.step_size'), gamma=_cfg_require('old/GPN_change.py.MetaSGDTrainer.train.gamma'))

            for epoch in range(epochs):
                total_loss = 0
                total_acc = 0
                processed_tasks = 0

                # 记录 epoch 开始时的平均学习率
                epoch_start_lr = torch.mean(torch.stack([torch.mean(torch.clamp(p, min=1e-6)) for p in self.alphas.values()]))

                with tqdm(train_loader, desc=f"Meta-SGD Epoch {epoch+1}/{epochs}") as pbar:
                    for batch_idx, meta_batch in enumerate(pbar):
                        if processed_tasks >= num_tasks:
                            break

                        # 关键步骤：调用修复后的 meta_train_step
                        loss, acc = self.meta_train_step(meta_batch)

                        total_loss += loss
                        total_acc += acc
                        processed_tasks += len(meta_batch[0])

                        # 计算当前学习率（用于 tqdm 实时显示）
                        current_lr = torch.mean(torch.stack([torch.mean(torch.clamp(p, min=1e-6)) for p in self.alphas.values()]))

                        pbar.set_postfix(
                            meta_loss=f'{loss:.4f}',
                            accuracy=f'{acc:.4f}',
                            avg_lr=f'{current_lr:.6f}'
                        )

                scheduler.step()
                avg_loss = total_loss / len(train_loader)
                avg_acc = total_acc / len(train_loader)

                # 记录 epoch 结束时的平均学习率
                epoch_end_lr = torch.mean(torch.stack([torch.mean(torch.clamp(p, min=1e-6)) for p in self.alphas.values()]))

                # --- 详细日志打印 (替换原有的 conditional 打印块) ---
                print(f"Meta-SGD Epoch {epoch+1} - Meta Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
                print(f"学习率变化: {epoch_start_lr:.6f} -> {epoch_end_lr:.6f} (变化: {epoch_end_lr - epoch_start_lr:.6f})")
