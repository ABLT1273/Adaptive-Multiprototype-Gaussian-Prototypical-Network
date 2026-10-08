from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
#主要改进：全Block替换: RepVGG(浅层) + ConvNeXt(深层)
#         注意力: Coordinate Attention (保留空间信息)
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import GPNLoss_Advanced
from pathlib import Path
from tqdm import tqdm
from data_loader import *
import gc

def count_parameters(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 分别统计CA参数
    ca_params = 0
    for name, module in model.named_modules():
        if isinstance(module, CoordinateAttention):
            ca_params += sum(p.numel() for p in module.parameters())

    print(f"Total parameters: {total:,} ({total/1e3:.1f}K)")
    print(f"Trainable parameters: {trainable:,}")
    print(f"CA module parameters: {ca_params:,} ({ca_params/1e3:.1f}K)")
    print(f"Non-CA parameters: {total-ca_params:,} ({(total-ca_params)/1e3:.1f}K)")

    return total

class GPNTrainer:
    """
    GPN模型训练器
    """
    def __init__(self, model,device,loss_feature_dim=None, lr=None,metric_type=None):
        loss_feature_dim = _cfg_resolve('old/GPN_new.py.GPNTrainer.__init__.loss_feature_dim', loss_feature_dim)
        lr = _cfg_resolve('old/GPN_new.py.GPNTrainer.__init__.lr', lr)
        metric_type = _cfg_resolve('old/GPN_new.py.GPNTrainer.__init__.metric_type', metric_type)
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
        save_path = _cfg_resolve('old/GPN_new.py.GPNTrainer.save_model.save_path', save_path)
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
            class_sigma = sigma[class_mask]

            weighted_sum = torch.sum(class_sigma * class_v, dim=0)
            sigma_sum = torch.sum(class_sigma, dim=0)
            prototypes[i] = weighted_sum / (sigma_sum + 1e-8)

            # 计算精度矩阵（对角矩阵，公式<configured>）
            precision_diag = torch.mean(class_sigma, dim=0)
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('old/GPN_new.py.GPNTrainer.train_step_batch.max_norm'))

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
        num_tasks = _cfg_resolve('old/GPN_new.py.GPNTrainer.train.num_tasks', num_tasks)
        n_ways = _cfg_resolve('old/GPN_new.py.GPNTrainer.train.n_ways', n_ways)
        epochs = _cfg_resolve('old/GPN_new.py.GPNTrainer.train.epochs', epochs)
        save_path = _cfg_resolve('old/GPN_new.py.GPNTrainer.train.save_path', save_path)
        self.model.train()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=_cfg_require('old/GPN_new.py.GPNTrainer.train.step_size'), gamma=_cfg_require('old/GPN_new.py.GPNTrainer.train.gamma'))

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
        self.model.switch_to_deploy()
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
        print(f"当前测试acc:{avg_acc}")
        return avg_acc
    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """完整的评估协议（符合论文）"""
        n_trials = _cfg_resolve('old/GPN_new.py.GPNTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('old/GPN_new.py.GPNTrainer.full_evaluation.q_query', q_query)
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
                        batch_size=_cfg_require('old/GPN_new.py.GPNTrainer.full_evaluation.batch_size'), num_tasks=_cfg_require('old/GPN_new.py.GPNTrainer.full_evaluation.num_tasks'), epochs=_cfg_require('old/GPN_new.py.GPNTrainer.full_evaluation.epochs')
                    )
                    loader = DataLoader(meta_test, batch_size=_cfg_require('old/GPN_new.py.GPNTrainer.full_evaluation.batch_size__2'))

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

class CoordinateAttention(nn.Module):
    '\n    Coordinate Attention模块 - 替代传统SE\n    保留空间位置信息的注意力机制\n    论文: Coordinate Attention for Efficient Mobile Network Design (CVPR <configured>)\n    '
    def __init__(self, channels, reduction=None):
        reduction = _cfg_resolve('old/GPN_new.py.CoordinateAttention.__init__.reduction', reduction)
        super().__init__()

        # 确保reduced_channels至少为<configured>
        reduced_channels = max(8, channels // reduction)

        # X方向和Y方向的平均池化通过Conv实现
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 共享的1x1卷积（降维）
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=_cfg_require('old/GPN_new.py.CoordinateAttention.__init__.kernel_size'), bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.SiLU(inplace=True)

        # 分别为H和W生成注意力权重（升维）
        self.conv_h = nn.Conv2d(reduced_channels, channels, kernel_size=_cfg_require('old/GPN_new.py.CoordinateAttention.__init__.kernel_size__2'), bias=False)
        self.conv_w = nn.Conv2d(reduced_channels, channels, kernel_size=_cfg_require('old/GPN_new.py.CoordinateAttention.__init__.kernel_size__3'), bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # 沿H和W方向池化
        x_h = self.pool_h(x)  # [B, C, H, <configured>]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, <configured>]

        # 拼接
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, <configured>]

        # 共享变换
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分离H和W
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 应用注意力
        out = x * a_h * a_w

        return out


class ConvNeXtBlock(nn.Module):
    '\n    ConvNeXt风格的Block - 现代化设计\n    \n    架构: DWConv 7x7 -> LayerNorm -> 1x1 Conv(4x expansion) -> GELU -> 1x1 Conv -> DropPath\n    \n    关键改进:\n    <configured>. 大kernel深度卷积(7x7) - 更大感受野\n    <configured>. LayerNorm替代BatchNorm - 更稳定\n    <configured>. 反向瓶颈设计(先扩张后压缩)\n    <configured>. GELU激活函数\n    <configured>. DropPath正则化\n    '
    def __init__(self, in_channels, out_channels, stride=None,
                 expansion=None, drop_path=None, use_ca=None):
        stride = _cfg_resolve('old/GPN_new.py.ConvNeXtBlock.__init__.stride', stride)
        expansion = _cfg_resolve('old/GPN_new.py.ConvNeXtBlock.__init__.expansion', expansion)
        drop_path = _cfg_resolve('old/GPN_new.py.ConvNeXtBlock.__init__.drop_path', drop_path)
        use_ca = _cfg_resolve('old/GPN_new.py.ConvNeXtBlock.__init__.use_ca', use_ca)
        super().__init__()

        # 如果stride><configured>，使用额外的下采样
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=_cfg_require('old/GPN_new.py.ConvNeXtBlock.__init__.kernel_size__3'),
                         stride=_cfg_require('old/GPN_new.py.ConvNeXtBlock.__init__.stride__2'), groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('old/GPN_new.py.ConvNeXtBlock.__init__.kernel_size__4'), bias=False),
            )

        # 调整in_channels（如果有下采样）
        if stride > 1:
            in_channels = out_channels

        # 大kernel深度卷积
        self.dwconv = nn.Conv2d(in_channels, in_channels,
                               kernel_size=_cfg_require('old/GPN_new.py.ConvNeXtBlock.__init__.kernel_size'), padding=_cfg_require('old/GPN_new.py.ConvNeXtBlock.__init__.padding'),
                               groups=in_channels, bias=False)

        # LayerNorm (通道维度)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

        # 反向瓶颈：1x1扩张卷积
        hidden_channels = int(in_channels * expansion)
        self.pwconv1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()

        # 1x1压缩卷积
        self.pwconv2 = nn.Linear(hidden_channels, out_channels)

        # Coordinate Attention (可选)
        self.use_ca = use_ca
        if use_ca and in_channels == out_channels:
            self.ca = CoordinateAttention(out_channels, reduction=_cfg_require('old/GPN_new.py.ConvNeXtBlock.__init__.reduction'))
        else:
            self.ca = None

        # DropPath正则化
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels and stride == 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                     kernel_size=_cfg_require('old/GPN_new.py.ConvNeXtBlock.__init__.kernel_size__2'), bias=False)

    def forward(self, x):
        # 下采样（如果需要）
        if self.downsample is not None:
            x = self.downsample(x)

        identity = self.shortcut(x)

        # 深度卷积
        out = self.dwconv(x)

        # LayerNorm需要转换维度: [B,C,H,W] -> [B,H,W,C]
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)

        # 反向瓶颈
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        # 转回 [B,C,H,W]
        out = out.permute(0, 3, 1, 2)

        # Coordinate Attention
        if self.ca is not None:
            out = self.ca(out)

        # DropPath + Residual
        out = identity + self.drop_path(out)

        return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        drop_prob = _cfg_resolve('old/GPN_new.py.DropPath.__init__.drop_prob', drop_prob)
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


class RepVGGBlock(nn.Module):
    '\n    RepVGG风格的Block - 训练推理解耦\n    \n    训练时: 3x3 Conv + 1x1 Conv + Identity (多分支)\n    推理时: 重参数化为单个3x3 Conv (高效)\n    \n    论文: RepVGG: Making VGG-style ConvNets Great Again (CVPR <configured>)\n    '
    def __init__(self, in_channels, out_channels, stride=None, use_ca=None):
        stride = _cfg_resolve('old/GPN_new.py.RepVGGBlock.__init__.stride', stride)
        use_ca = _cfg_resolve('old/GPN_new.py.RepVGGBlock.__init__.use_ca', use_ca)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 3x3卷积分支
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('old/GPN_new.py.RepVGGBlock.__init__.kernel_size'),
                     stride=stride, padding=_cfg_require('old/GPN_new.py.RepVGGBlock.__init__.padding'), bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 1x1卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('old/GPN_new.py.RepVGGBlock.__init__.kernel_size__2'),
                     stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.identity = nn.BatchNorm2d(in_channels) \
            if in_channels == out_channels and stride == 1 else None

        self.act = nn.SiLU(inplace=True)

        # Coordinate Attention
        self.use_ca = use_ca
        if use_ca:
            self.ca = CoordinateAttention(out_channels, reduction=_cfg_require('old/GPN_new.py.RepVGGBlock.__init__.reduction'))

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

        if self.use_ca:
            out = self.ca(out)

        return out

    def switch_to_deploy(self):
        """
        重参数化：将多分支融合为单个3x3卷积
        部署前调用此函数
        """
        if hasattr(self, 'conv_fused'):
            return

        # 获取3x3卷积的权重
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv3x3)

        # 获取1x1卷积的权重（需要pad到3x3）
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv1x1)
        kernel1x1 = F.pad(kernel1x1, [1, 1, 1, 1])  # pad到3x3

        # 获取identity的权重
        kernel_identity = 0
        bias_identity = 0
        if self.identity is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)
            # 转换为3x3卷积kernel
            kernel_identity = F.pad(
                torch.eye(self.in_channels).view(self.in_channels, self.in_channels, 1, 1),
                [1, 1, 1, 1]
            ).to(kernel3x3.device)

        # 融合所有分支
        kernel_fused = kernel3x3 + kernel1x1 + kernel_identity
        bias_fused = bias3x3 + bias1x1 + bias_identity

        # 创建融合后的卷积层
        self.conv_fused = nn.Conv2d(
            self.in_channels, self.out_channels,
            kernel_size=_cfg_require('old/GPN_new.py.RepVGGBlock.switch_to_deploy.kernel_size'), stride=self.stride, padding=_cfg_require('old/GPN_new.py.RepVGGBlock.switch_to_deploy.padding'), bias=True
        )
        self.conv_fused.weight.data = kernel_fused
        self.conv_fused.bias.data = bias_fused

        # 删除原始分支（节省内存）
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')
        if self.identity is not None:
            self.__delattr__('identity')

    def _fuse_bn_tensor(self, branch):
        """融合Conv+BN为单个Conv"""
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            # Identity分支
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels
                kernel_value = torch.zeros((self.in_channels, self.in_channels, 3, 3))
                for i in range(self.in_channels):
                    kernel_value[i, i, 1, 1] = 1
                self.id_tensor = kernel_value.to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class GPN_Optimized(nn.Module):
    '\n    优化的GPN模型 - 方案A: 平衡性能\n    \n    架构改进:\n    <configured>. Stage1-<configured>: RepVGG Blocks (训练稳定 + 推理高效)\n    <configured>. Stage3-<configured>: ConvNeXt Blocks (现代架构 + 更大感受野)\n    <configured>. 全局使用Coordinate Attention (空间位置信息)\n    <configured>. LayerNorm + GELU (更稳定的训练)\n    <configured>. DropPath正则化 (防止过拟合)\n    \n    预期效果:\n    - 精度提升: <configured>-<configured>\n    - 推理速度: +<configured>-<configured>\n    - 参数量: 略增(~<configured>)\n    '
    def __init__(self, use_repvgg=None, use_convnext=None):
        use_repvgg = _cfg_resolve('old/GPN_new.py.GPN_Optimized.__init__.use_repvgg', use_repvgg)
        use_convnext = _cfg_resolve('old/GPN_new.py.GPN_Optimized.__init__.use_convnext', use_convnext)
        super().__init__()

        self.use_repvgg = use_repvgg
        self.use_convnext = use_convnext

        # Initial Stem
        self.stem = nn.Sequential(
            nn.Conv2d(_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.Conv2d_arg0'), _cfg_require('old/GPN_new.py.GPN_Optimized.__init__.Conv2d_arg1'), kernel_size=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.kernel_size__5'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__5'), padding=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.padding'), bias=False),
            nn.BatchNorm2d(_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.BatchNorm2d_arg0')),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.kernel_size__6'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__6'), padding=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.padding__2'))
        )
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_repvgg:
            self.block1 = RepVGGBlock(24, 48, stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__7'), use_ca=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.use_ca'))
        else:
            self.block1 = ConvNeXtBlock(24, 48, stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__8'), use_ca=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.use_ca__2'))
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.kernel_size'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # Stage <configured>: RepVGG Block
        if use_repvgg:
            self.block2 = RepVGGBlock(48, 96, stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__9'), use_ca=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.use_ca__3'))
        else:
            self.block2 = ConvNeXtBlock(48, 96, stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__10'), use_ca=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.use_ca__4'))

        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.kernel_size__2'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__2'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_convnext:
            self.block3 = ConvNeXtBlock(96, _cfg_require('old/GPN_new.py.GPN_Optimized.__init__.size_or_budget'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__11'),
                                       expansion=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.expansion'), drop_path=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.drop_path'), use_ca=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.use_ca__5'))
        else:
            self.block3 = RepVGGBlock(96, _cfg_require('old/GPN_new.py.GPN_Optimized.__init__.size_or_budget__2'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__12'), use_ca=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.use_ca__6'))
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.kernel_size__3'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__3'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # Stage <configured>: ConvNeXt Block
        if use_convnext:
            self.block4 = ConvNeXtBlock(_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.size_or_budget__3'), _cfg_require('old/GPN_new.py.GPN_Optimized.__init__.size_or_budget__4'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__13'),
                                       expansion=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.expansion__2'), drop_path=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.drop_path__2'), use_ca=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.use_ca__7'))
        else:
            self.block4 = RepVGGBlock(_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.size_or_budget__5'), _cfg_require('old/GPN_new.py.GPN_Optimized.__init__.size_or_budget__6'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__14'), use_ca=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.use_ca__8'))
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.kernel_size__4'), stride=_cfg_require('old/GPN_new.py.GPN_Optimized.__init__.stride__4'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def forward(self, x, return_intermediate=False):
        '\n        前向传播\n        \n        Args:\n            x: 输入 [B, <configured>, <configured>, <configured>]\n            return_intermediate: 是否返回中间特征\n            \n        Returns:\n            v: embedding feature [B, <configured>]\n            s: precision feature [B, <configured>]\n            features: (可选) 中间特征字典\n        '
        features = {} if return_intermediate else None

        # Stem
        x = self.stem(x)  # [B, <configured>, <configured>, <configured>]

        # Stage <configured>-<configured>
        x = self.block1(x)  # [B, <configured>, <configured>, <configured>]
        x = self.pool1(x)   # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block1'] = x

        x = self.block2(x)  # [B, <configured>, <configured>, <configured>]
        x = self.pool2(x)   # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block2'] = x

        x = self.block3(x)  # [B, <configured>, <configured>, <configured>]
        x = self.pool3(x)   # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block3'] = x

        x = self.block4(x)  # [B, <configured>, <configured>, <configured>]
        x = self.pool4(x)   # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block4'] = x

        # Channel Split: v和s
        v_features = x[:, :_cfg_require('old/GPN_new.py.GPN_Optimized.forward.size_or_budget'), :, :]  # [B, <configured>, <configured>, <configured>]
        s_features = x[:, _cfg_require('old/GPN_new.py.GPN_Optimized.forward.size_or_budget__2'):, :, :]  # [B, <configured>, <configured>, <configured>]

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

    def switch_to_deploy(self):
        '\n        切换到部署模式 - 重参数化RepVGG块\n        在推理前调用此函数可提速<configured>-<configured>\n        '
        if self.use_repvgg:
            print("正在重参数化RepVGG块...")
            if hasattr(self.block1, 'switch_to_deploy'):
                self.block1.switch_to_deploy()
            if hasattr(self.block2, 'switch_to_deploy'):
                self.block2.switch_to_deploy()
            print("重参数化完成！模型已优化为推理模式")

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


class MetaSGDTrainer(GPNTrainer):
    """
    简化的Meta-SGD实现，直接优化学习率参数
    """
    def __init__(self, model, device, feature_dim=None, base_lr=None, meta_lr=None, l1_lambda=None,use_parent_train=True):
        feature_dim = _cfg_resolve('old/GPN_new.py.MetaSGDTrainer.__init__.feature_dim', feature_dim)
        base_lr = _cfg_resolve('old/GPN_new.py.MetaSGDTrainer.__init__.base_lr', base_lr)
        meta_lr = _cfg_resolve('old/GPN_new.py.MetaSGDTrainer.__init__.meta_lr', meta_lr)
        l1_lambda = _cfg_resolve('old/GPN_new.py.MetaSGDTrainer.__init__.l1_lambda', l1_lambda)
        super().__init__(model=model, device=device, feature_dim=feature_dim, lr=base_lr,
                         loss_alpha=loss_alpha, sinkhorn_eps=sinkhorn_eps, sinkhorn_iter=sinkhorn_iter)
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
                torch.optim.SGD(self.model.parameters(), lr=_cfg_require('old/GPN_new.py.MetaSGDTrainer.meta_train_step.lr')), # 优化器占位符
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
        torch.nn.utils.clip_grad_norm_(self.alphas.values(), max_norm=_cfg_require('old/GPN_new.py.MetaSGDTrainer.meta_train_step.max_norm'))
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

        num_tasks = _cfg_resolve('old/GPN_new.py.MetaSGDTrainer.train.num_tasks', num_tasks)
        n_ways = _cfg_resolve('old/GPN_new.py.MetaSGDTrainer.train.n_ways', n_ways)
        epochs = _cfg_resolve('old/GPN_new.py.MetaSGDTrainer.train.epochs', epochs)
        save_path = _cfg_resolve('old/GPN_new.py.MetaSGDTrainer.train.save_path', save_path)
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
            scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optimizer, step_size=_cfg_require('old/GPN_new.py.MetaSGDTrainer.train.step_size'), gamma=_cfg_require('old/GPN_new.py.MetaSGDTrainer.train.gamma'))

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


if __name__ == "__main__":
    print("=" * 60)
    print("GPN优化模型 - 方案A: 平衡性能")
    print("=" * 60)

    # 创建模型
    model = GPN_Optimized(use_repvgg=_cfg_require('old/GPN_new.py.module.use_repvgg'), use_convnext=_cfg_require('old/GPN_new.py.module.use_convnext'))

    # 统计参数
    print("\n模型参数统计:")
    count_parameters(model)

    # 测试前向传播
    print("\n测试前向传播...")
    x = torch.randn(2, 1, _cfg_require('old/GPN_new.py.module.size_or_budget'), _cfg_require('old/GPN_new.py.module.size_or_budget__2'))

    # 训练模式
    model.train()
    v, s = model(x)
    print(f"✓ 训练模式 - v: {v.shape}, s: {s.shape}")
    print(f"  s值范围: [{s.min().item():.4f}, {s.max().item():.4f}] (应>1)")

    # 推理模式（重参数化）
    model.eval()
    model.switch_to_deploy()
    v, s = model(x)
    print(f"✓ 推理模式 - v: {v.shape}, s: {s.shape}")

    # 中间特征
    v, s, features = model(x, return_intermediate=True)
    print(f"\n✓ 中间特征:")
    for name, feat in features.items():
        if isinstance(feat, torch.Tensor):
            print(f"  {name}: {feat.shape}")
