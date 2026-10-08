from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
#在V3基础上改进多原型初始化,已回退
#新增：提取边缘和角点特征，混入trainer类
#新增：截取中间频率特征，比例截取高低频（低能）
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from pathlib import Path
from tqdm import tqdm
from amgpn.data.wavelet_episodes import *
from amgpn.legacy.v3.fixed_prototypes_loss import MultiPrototypeGPNLoss
from amgpn.data.feature_maps import *
import gc
from amgpn.data.preprocessing import crop_and_rescale_symmetric

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
    def __init__(self, model, device, use_multi=None, mlr=None, prototypes_per_class=None,
                 use_edge_corner=None,lambda_L=None,lambda_H=None,cut_ratio=None,resize=None):  # 新增开关参数
        """
        Args:
            use_edge_corner: 是否使用边缘和角点特征增强
        """
        use_multi = _cfg_resolve('GPN_V4.py.GPNTrainer.__init__.use_multi', use_multi)
        mlr = _cfg_resolve('GPN_V4.py.GPNTrainer.__init__.mlr', mlr)
        prototypes_per_class = _cfg_resolve('GPN_V4.py.GPNTrainer.__init__.prototypes_per_class', prototypes_per_class)
        use_edge_corner = _cfg_resolve('GPN_V4.py.GPNTrainer.__init__.use_edge_corner', use_edge_corner)
        lambda_L = _cfg_resolve('GPN_V4.py.GPNTrainer.__init__.lambda_L', lambda_L)
        lambda_H = _cfg_resolve('GPN_V4.py.GPNTrainer.__init__.lambda_H', lambda_H)
        cut_ratio = _cfg_resolve('GPN_V4.py.GPNTrainer.__init__.cut_ratio', cut_ratio)
        resize = _cfg_resolve('GPN_V4.py.GPNTrainer.__init__.resize', resize)
        self.prototypes_per_class = prototypes_per_class
        self.device = device
        self.use_edge_corner = use_edge_corner
        self.cut_ratio=cut_ratio
        self.resize=resize
        # 初始化模型
        self.model = model
        self.model.to(device)
        self.use_multi = use_multi
        self.loss_fn = MultiPrototypeGPNLoss(self.prototypes_per_class, use_multi=self.use_multi)
        self.loss_fn.to(device)

        self.lambda_L=lambda_L
        self.lambda_H=lambda_H
        # 初始化特征提取器
        if self.use_edge_corner:
            self.feature_extractor = NoiseInsensitiveFeatureExtractor(
                lambda_L=self.lambda_L,
                lambda_H=self.lambda_H,
                delta=_cfg_require('GPN_V4.py.GPNTrainer.__init__.delta'),
                gamma=_cfg_require('GPN_V4.py.GPNTrainer.__init__.gamma'),
                eta_percentile=_cfg_require('GPN_V4.py.GPNTrainer.__init__.eta_percentile'),
                gaussian_window_size=_cfg_require('GPN_V4.py.GPNTrainer.__init__.gaussian_window_size')
            )

            # 不需要.to(device)，因为它只是计算操作，会自动在输入的设备上运行

            # 修改模型第一层
            self._adapt_model_input()

        self.model_params = list(self.model.parameters())
        self.mlr = mlr
        self.optimizer_model = torch.optim.Adam(self.model_params, lr=self.mlr)

        if self.use_edge_corner:
            print(f"噪声不敏感特征提取器已启用")
        print(f"GPN模型初始化完成，设备: {self.device}")


    def _adapt_model_input(self):
        '\n        修改模型第一层以接受<configured>通道输入\n        '
        old_conv = self.model.conv1

        # 创建新的<configured>通道卷积层
        new_conv = nn.Conv2d(
            in_channels=_cfg_require('GPN_V4.py.GPNTrainer._adapt_model_input.in_channels'),  # 改为<configured>通道
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # 初始化新卷积层的权重
        with torch.no_grad():
            # 第<configured>通道：原始时频图，使用原权重
            new_conv.weight[:, 0:1, :, :] = old_conv.weight

            # 第<configured>-<configured>通道：边缘和角点，使用原权重的平均值初始化
            avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 1:3, :, :] = avg_weight.repeat(1, 2, 1, 1)

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        # 替换
        self.model.conv1 = new_conv.to(self.device)

    def save_model(self, save_path=None):
        """
        保存模型和损失函数的所有可学习参数
        """

        save_path = _cfg_resolve('GPN_V4.py.GPNTrainer.save_model.save_path', save_path)
        filtered_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if not ("total_ops" in k or "total_params" in k)
        }
        checkpoint = {
            'model_state_dict': filtered_state_dict,
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
    def compute_multi_gaussian_prototypes(self, support_v, support_s, support_labels, n_ways, k_shot):
        """
        计算多高斯原型

        Args:
            support_v: [n_ways * k_shot, feature_dim]
            support_s: [n_ways * k_shot, feature_dim]
            support_labels: [n_ways * k_shot]
            n_ways: 类别数
            k_shot: 每类样本数

        Returns:
            prototypes: [n_ways * prototypes_per_class, feature_dim]
            precision_matrices: [n_ways * prototypes_per_class, feature_dim, feature_dim]
            prototype_assignments: [n_ways, prototypes_per_class] - 原型有效性掩码
        """
        feature_dim = support_v.shape[1]

        # 初始化多原型
        total_prototypes = n_ways * self.prototypes_per_class
        prototypes = torch.zeros(total_prototypes, feature_dim, device=self.device)
        precision_matrices = torch.zeros(total_prototypes, feature_dim, feature_dim, device=self.device)
        prototype_assignments = torch.ones(n_ways, self.prototypes_per_class, device=self.device)

        sigma = support_s
        unique_labels = torch.unique(support_labels)

        for i, class_label in enumerate(unique_labels):
            class_mask = (support_labels == class_label)
            class_v = support_v[class_mask]  # [k_shot, feature_dim]
            class_sigma = sigma[class_mask]  # [k_shot, feature_dim]

            # 原：单轮使用K-Means找到多个子簇中心
            if k_shot >= self.prototypes_per_class:
                # 简单K-Means初始化
                centroids = self._kmeans_plus_plus(class_v, self.prototypes_per_class)


                for j in range(self.prototypes_per_class):
                    proto_idx = i * self.prototypes_per_class + j

                    if j < len(centroids):
                        # 分配样本到最近的原型
                        distances = torch.cdist(class_v, centroids[j:j+1]).squeeze()
                        cluster_mask = (distances == torch.min(distances))


                        if cluster_mask.sum() > 0:
                            cluster_v = class_v[cluster_mask]
                            cluster_sigma = class_sigma[cluster_mask]

                            # 计算加权原型
                            weighted_sum = torch.sum(cluster_sigma * cluster_v, dim=0)
                            sigma_sum = torch.sum(cluster_sigma, dim=0)
                            prototypes[proto_idx] = weighted_sum / (sigma_sum + 1e-8)

                            # 计算精度矩阵
                            precision_diag = torch.mean(cluster_sigma, dim=0)
                            precision_matrices[proto_idx] = torch.diag(precision_diag)
                        else:
                            # 无效原型
                            prototype_assignments[i, j] = 0
                            # 选择距离所有已有原型最远的样本作为新原型
                            if j > 0:
                                # 计算到已有原型的距离
                                existing_protos = prototypes[i * self.prototypes_per_class : i * self.prototypes_per_class + j]
                                distances_to_existing = torch.cdist(class_v, existing_protos)  # [k_shot, j]
                                min_distances = torch.min(distances_to_existing, dim=1)[0]  # [k_shot]

                                # 选择最远的样本
                                farthest_idx = torch.argmax(min_distances)
                                prototypes[proto_idx] = class_v[farthest_idx]

                                # 使用该样本的sigma
                                precision_diag = class_sigma[farthest_idx]
                                precision_matrices[proto_idx] = torch.diag(precision_diag)
                            else:
                                # 第一个原型就是空的（极端情况）
                                prototypes[proto_idx] = centroids[j]
                                precision_matrices[proto_idx] = torch.eye(feature_dim, device=self.device)
                    else:
                        # 原型数量不足
                        prototype_assignments[i, j] = 0
                        prototypes[proto_idx] = torch.randn(feature_dim, device=self.device) * 0.01
                        precision_matrices[proto_idx] = torch.eye(feature_dim, device=self.device)
            else:
                # 样本太少，使用单个原型
                for j in range(self.prototypes_per_class):
                    proto_idx = i * self.prototypes_per_class + j
                    if j == 0:
                        # 第一个原型使用所有样本
                        weighted_sum = torch.sum(class_sigma * class_v, dim=0)
                        sigma_sum = torch.sum(class_sigma, dim=0)
                        prototypes[proto_idx] = weighted_sum / (sigma_sum + 1e-8)
                        precision_diag = torch.mean(class_sigma, dim=0)
                        precision_matrices[proto_idx] = torch.diag(precision_diag)
                    else:
                        # 其他原型标记为无效
                        prototype_assignments[i, j] = 0
                        prototypes[proto_idx] = prototypes[i * self.prototypes_per_class]  # 复制第一个原型
                        precision_matrices[proto_idx] = precision_matrices[i * self.prototypes_per_class]

        return prototypes, precision_matrices, prototype_assignments
    def _kmeans_plus_plus(self, data, k, max_iters=None):
        """
        简化的K-Means++初始化
        """
        max_iters = _cfg_resolve('GPN_V4.py.GPNTrainer._kmeans_plus_plus.max_iters', max_iters)
        n_samples = data.shape[0]
        centroids = []

        # 第一个中心随机选择
        first_idx = torch.randint(0, n_samples, (1,))
        centroids.append(data[first_idx])

        for _ in range(1, k):
            # 计算每个样本到最近中心的距离
            distances = []
            for sample in data:
                min_dist = float('inf')
                for center in centroids:
                    dist = torch.norm(sample - center)
                    if dist < min_dist:
                        min_dist = dist
                distances.append(min_dist)

            # 根据距离概率选择下一个中心
            distances = torch.tensor(distances)
            probabilities = distances / distances.sum()
            next_idx = torch.multinomial(probabilities, 1)
            centroids.append(data[next_idx])

        return torch.cat(centroids)
    def _soft_kmeans_with_uncertainty(self, class_v, class_sigma, k, max_iters=None,verbose=False):
        """
        考虑不确定性的软K-Means

        核心思想: 用sigma加权距离，迭代优化
        """
        max_iters = _cfg_resolve('GPN_V4.py.GPNTrainer._soft_kmeans_with_uncertainty.max_iters', max_iters)
        n_samples, feature_dim = class_v.shape

        # 初始化: 仍用K-Means++（快速）
        centroids = self._kmeans_plus_plus(class_v, k)

        for iteration in range(max_iters):
            # === E步: 计算后验概率（软分配） ===
            # 对每个样本，计算到各中心的"加权距离"
            responsibilities = torch.zeros(n_samples, k, device=class_v.device)

            for j in range(k):
                diff = class_v - centroids[j:j+1]  # [n_samples, feature_dim]

                # 用sigma作为权重（不确定性高的维度权重低）
                weighted_diff = diff / (class_sigma + 1e-8)
                distances = torch.sum(weighted_diff ** 2, dim=1)  # [n_samples]

                # 转换为概率（距离越小概率越大）
                responsibilities[:, j] = torch.exp(-distances / 2)

            # 归一化为概率
            responsibilities = responsibilities / (responsibilities.sum(dim=1, keepdim=True) + 1e-8)

            # === M步: 更新中心（加权平均） ===
            for j in range(k):
                weights = responsibilities[:, j:j+1]  # [n_samples, <configured>]

                # 同时考虑后验概率和sigma
                combined_weights = weights * class_sigma
                weighted_sum = torch.sum(combined_weights * class_v, dim=0)
                weight_sum = torch.sum(combined_weights, dim=0)

                centroids[j] = weighted_sum / (weight_sum + 1e-8)

            if verbose:
                # 计算簇内紧密度
                final_assignments_temp = torch.argmax(responsibilities, dim=1)
                for j in range(k):
                    cluster_mask = (final_assignments_temp == j)
                    if cluster_mask.sum() > 0:
                        cluster_samples = class_v[cluster_mask]
                        intra_dist = torch.cdist(cluster_samples, centroids[j:j+1]).mean()
                        print(f"  迭代{iteration+1}, 簇{j}: {cluster_mask.sum()}个样本, 平均距离={intra_dist:.4f}")
        # === 硬分配: 最终确定簇归属 ===
        final_assignments = torch.argmax(responsibilities, dim=1)

        return centroids, final_assignments
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


    def _single_task_forward(self, support_signals, support_labels, query_signals, query_labels):
        """
        单个任务的前向传播（不更新参数）
        """

        # 移动到设备
        support_signals = support_signals.to(self.device).float().unsqueeze(1)
        support_labels = support_labels.to(self.device)
        query_signals = query_signals.to(self.device).float().unsqueeze(1)
        query_labels = query_labels.to(self.device)

        if self.cut_ratio!=0:
            processed_support_list = []
            for signal in support_signals.unbind(0):
                # 对每个信号应用遮盖函数，并添加到列表中
                processed_support_list.append(crop_and_rescale_symmetric(signal, self.cut_ratio,self.resize))

            # 将处理后的信号重新堆叠成 [B, <configured>, <configured>, <configured>] 的 Tensor
            support_signals = torch.stack(processed_support_list, dim=0)

            # <configured>. 处理 query_signals
            processed_query_list = [] # 将列表名改为 processed_query_list 更清晰
            for signal in query_signals.unbind(0):
                # 对每个信号应用遮盖函数
                processed_query_list.append(crop_and_rescale_symmetric(signal,self.cut_ratio,self.resize))

            # 将处理后的信号重新堆叠成 [B, <configured>, <configured>, <configured>] 的 Tensor
            query_signals = torch.stack(processed_query_list, dim=0)


        # ========== 新增：提取边缘和角点特征 ==========
        if self.use_edge_corner:
            support_signals = self.feature_extractor.forward(support_signals)  # [N, <configured>, <configured>, <configured>]
            query_signals = self.feature_extractor.forward(query_signals)      # [M, <configured>, <configured>, <configured>]


        # 前向传播
        support_v, support_s = self.model(support_signals)
        query_v, query_s = self.model(query_signals)

        if self.use_multi:

            # 标签重映射
            unique_labels = torch.unique(support_labels)
            n_ways = len(unique_labels)
            k_shot = len(support_labels) // n_ways
            label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

            remapped_support_labels = torch.tensor([label_mapping[label.item()] for label in support_labels], device=self.device)
            remapped_query_labels = torch.tensor([label_mapping[label.item()] for label in query_labels], device=self.device)

            # 计算多高斯原型
            prototypes, precision_matrices, prototype_assignments = self.compute_multi_gaussian_prototypes(
                support_v, support_s, remapped_support_labels, n_ways, k_shot
            )

            # 计算损失（使用多原型版本）
            loss, probabilities, distances = self.loss_fn(
                query_v, prototypes, precision_matrices, remapped_query_labels, prototype_assignments
            )

        else:

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


    def train_step_batch(self, meta_batch, step_index):
        """
        批处理训练步骤

        Args:
            meta_batch: DataLoader 返回的批次
                - support_signals: [batch_size, n_way*k_shot, H, W]
                - support_labels: [batch_size, n_way*k_shot]
                - query_signals: [batch_size, n_way*q_query, H, W]
                - query_labels: [batch_size, n_way*q_query]
            step_index: 当前步骤索引

        Returns:
            avg_loss: 批次平均损失
            avg_acc: 批次平均准确率
        """
        self.model.train()

        support_signals, support_labels, query_signals, query_labels = meta_batch
        batch_size = support_signals.size(0)  # DataLoader 的 batch_size

        total_loss = 0
        total_acc = 0

        # 清零梯度
        self.optimizer_model.zero_grad()

        # 处理批内所有任务，累积梯度
        for i in range(batch_size):
            # 单个任务的前向传播和损失计算
            loss, acc = self._single_task_forward(
                support_signals[i], support_labels[i],
                query_signals[i], query_labels[i]
            )

            # 累积损失（自动累积梯度）
            (loss / batch_size).backward()  # 除以 batch_size 来平均化梯度

            total_loss += loss.item()
            total_acc += acc

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('GPN_V4.py.GPNTrainer.train_step_batch.max_norm'))

        # 统一更新参数
        self.optimizer_model.step()

        return total_loss / batch_size, total_acc / batch_size


    def train(self, train_loader, test_loader, epochs=None,
              n_ways=None, save_path=None):
        """
        改进的元学习训练循环

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            n_ways: 类别数（用于测试）
            save_path: 模型保存路径
        """
        epochs = _cfg_resolve('GPN_V4.py.GPNTrainer.train.epochs', epochs)
        n_ways = _cfg_resolve('GPN_V4.py.GPNTrainer.train.n_ways', n_ways)
        save_path = _cfg_resolve('GPN_V4.py.GPNTrainer.train.save_path', save_path)
        self.model.train()

        scheduler_model = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model,
            step_size=_cfg_require('GPN_V4.py.GPNTrainer.train.step_size'),
            gamma=_cfg_require('GPN_V4.py.GPNTrainer.train.gamma')
        )

        print(f"开始元学习训练: {epochs} epochs")
        print(f"每个 epoch 有 {len(train_loader)} 个批次")
        print(f"实际 batch_size: {train_loader.batch_size}")
        print(f"每个 epoch 训练 {len(train_loader) * train_loader.batch_size} 个任务")

        best_acc = 0
        early_stop = 0

        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            processed_batches = 0


            # 使用 tqdm 显示进度
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, meta_batch in enumerate(pbar):
                    # 批处理训练步骤
                    loss, acc = self.train_step_batch(meta_batch, batch_idx)

                    total_loss += loss
                    total_acc += acc
                    processed_batches += 1

                    # 更新进度条
                    pbar.set_postfix(
                        loss=f'{loss:.4f}',
                        accuracy=f'{acc:.4f}',
                        lr=f'{self.optimizer_model.param_groups[0]["lr"]:.6f}'
                    )

            # 学习率调度
            scheduler_model.step()

            # 计算平均指标
            avg_loss = total_loss / processed_batches
            avg_acc = total_acc / processed_batches

            print(f"Epoch {epoch+1} 训练完成:avg_loss: {avg_loss:.4f} avg_acc: {avg_acc:.4f}")

            # 定期评估
            if (epoch + 1) % 5 == 0:
                print(f"\n开始测试评估...")
                test_acc = self.evaluate(
                    test_loader=test_loader,
                    n_ways=n_ways,
                    show_progress=True,
                    show_error_stats=True
                )
                print(f"测试准确率: {test_acc:.4f}")

                if test_acc > best_acc:
                    best_acc = test_acc
                    early_stop = 0
                else:
                    early_stop += 1
                    print(f"未提升（连续 {early_stop} 次）")
                    if early_stop >= 4:
                        print(f"\n早停触发，训练结束")
                        break
        self.save_model(save_path)

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"\n训练完成！最佳测试准确率: {best_acc:.4f}")

    def evaluate(self, test_loader, n_ways, show_progress=True, show_error_stats=True):
        """
        评估模型性能，并统计错误识别的类别和多原型坍缩情况

        新增：
        - 多原型统计功能（use_multi=True时）
        - 原型坍缩检测

        Args:
            test_loader: 数据加载器
            n_ways: 类别数
            show_progress: 是否显示进度条
            show_error_stats: 是否显示错误统计信息

        Returns:
            avg_acc: 平均准确率
        """
        self.model.eval()
        total_acc = 0

        # 错误统计结构
        error_details = {}
        global_error_stats = {}
        class_stats = {}

        # 多原型统计
        prototype_stats = {}

        # ========== 新增：原型坍缩检测 ==========
        collapse_stats = {
            'class_distances': {},     # {class_id: [distances_list]}
            'class_similarities': {},  # {class_id: [similarities_list]}
            'class_variance': {},      # {class_id: [variance_list]}
        }

        iterator = tqdm(test_loader, desc="Evaluating") if show_progress else test_loader

        with torch.no_grad():
            for task_id, meta_task in enumerate(iterator):
                support_signals, support_labels, query_signals, query_labels = meta_task

                # 移动到设备
                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                if self.cut_ratio!=0:
                    processed_support_list = []
                    for signal in support_signals.unbind(0):
                        # 对每个信号应用遮盖函数，并添加到列表中
                        processed_support_list.append(crop_and_rescale_symmetric(signal, self.cut_ratio,self.resize))

                    # 将处理后的信号重新堆叠成 [B, <configured>, <configured>, <configured>] 的 Tensor
                    support_signals = torch.stack(processed_support_list, dim=0)

                    processed_query_list = [] # 将列表名改为 processed_query_list 更清晰
                    for signal in query_signals.unbind(0):
                    # 对每个信号应用遮盖函数
                        processed_query_list.append(crop_and_rescale_symmetric(signal,self.cut_ratio,self.resize))
                    # 将处理后的信号重新堆叠成 [B, <configured>, <configured>, <configured>] 的 Tensor
                    query_signals = torch.stack(processed_query_list, dim=0)

                if self.use_multi:
                    # 计算多高斯原型
                    k_shot = len(support_labels) // n_ways
                    prototypes, precision_matrices, prototype_assignments = self.compute_multi_gaussian_prototypes(
                        support_v, support_s, remapped_support_labels, n_ways, k_shot
                    )

                    # 收集原型统计信息
                    for i, class_label in enumerate(unique_labels):
                        original_class = class_label.item()

                        # 统计该类激活的原型数量
                        active_prototypes = torch.sum(prototype_assignments[i]).item()
                        total_prototypes = self.prototypes_per_class

                        if original_class not in prototype_stats:
                            prototype_stats[original_class] = {
                                'total_tasks': 0,
                                'active_sum': 0,
                                'max_prototypes': total_prototypes
                            }

                        prototype_stats[original_class]['total_tasks'] += 1
                        prototype_stats[original_class]['active_sum'] += active_prototypes

                        # ========== 新增：计算原型坍缩指标 ==========
                        # 获取该类的所有原型
                        start_idx = i * self.prototypes_per_class
                        end_idx = start_idx + self.prototypes_per_class
                        class_prototypes = prototypes[start_idx:end_idx]  # [prototypes_per_class, feature_dim]
                        class_assignments = prototype_assignments[i]  # [prototypes_per_class]

                        # 只考虑激活的原型
                        active_mask = class_assignments > 0
                        if active_mask.sum() > 1:  # 至少<configured>个原型才能计算坍缩
                            active_protos = class_prototypes[active_mask]  # [n_active, feature_dim]

                            # <configured>. 计算原型间的成对距离
                            proto_distances = self._compute_pairwise_distances(active_protos)

                            # <configured>. 计算原型间的余弦相似度
                            proto_similarities = self._compute_pairwise_similarities(active_protos)

                            # <configured>. 计算原型的方差（离散程度）
                            proto_variance = torch.var(active_protos, dim=0).mean().item()

                            # 保存统计
                            if original_class not in collapse_stats['class_distances']:
                                collapse_stats['class_distances'][original_class] = []
                                collapse_stats['class_similarities'][original_class] = []
                                collapse_stats['class_variance'][original_class] = []

                            collapse_stats['class_distances'][original_class].extend(proto_distances)
                            collapse_stats['class_similarities'][original_class].extend(proto_similarities)
                            collapse_stats['class_variance'][original_class].append(proto_variance)

                    # 计算距离和准确率
                    query_v, query_s = self.model(query_signals)
                    all_distances = self.loss_fn.distance_metric(query_v, prototypes, precision_matrices)
                    distances_reshaped = all_distances.view(len(query_v), n_ways, self.prototypes_per_class)
                    min_distances, _ = torch.min(distances_reshaped, dim=2)
                    logits = -min_distances

                else:
                    # 计算高斯原型
                    prototypes, precision_matrices = self.compute_gaussian_prototypes(
                        support_v, support_s, remapped_support_labels, n_ways
                    )
                    query_v, query_s = self.model(query_signals)
                    distances = self.loss_fn.distance_metric(query_v, prototypes, precision_matrices)
                    logits = -distances

                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == remapped_query_labels).float().mean()
                total_acc += accuracy.item()

                # 错误分析（保持原有代码）
                if show_error_stats:
                    predictions_cpu = predictions.cpu().numpy()
                    remapped_query_labels_cpu = remapped_query_labels.cpu().numpy()
                    task_errors = {}

                    for i in range(len(predictions_cpu)):
                        pred_remapped = predictions_cpu[i]
                        true_remapped = remapped_query_labels_cpu[i]
                        pred_original = remapped_to_original[pred_remapped]
                        true_original = original_query_labels[i]

                        if true_original not in class_stats:
                            class_stats[true_original] = {'total': 0, 'correct': 0, 'wrong': 0}

                        class_stats[true_original]['total'] += 1

                        if pred_remapped == true_remapped:
                            class_stats[true_original]['correct'] += 1
                        else:
                            class_stats[true_original]['wrong'] += 1

                            if true_original not in task_errors:
                                task_errors[true_original] = {}
                            if pred_original not in task_errors[true_original]:
                                task_errors[true_original][pred_original] = 0
                            task_errors[true_original][pred_original] += 1

                            if true_original not in global_error_stats:
                                global_error_stats[true_original] = {}
                            if pred_original not in global_error_stats[true_original]:
                                global_error_stats[true_original][pred_original] = 0
                            global_error_stats[true_original][pred_original] += 1

                    if task_errors:
                        error_details[task_id] = task_errors

            avg_acc = total_acc / len(test_loader)

        # ========== 打印统计信息 ==========
        if show_error_stats:
            print("\n" + "="*80)
            print("评估统计报告")
            print("="*80)

            # <configured>. 总体统计
            total_samples = sum(stats['total'] for stats in class_stats.values())
            total_errors = sum(stats['wrong'] for stats in class_stats.values())

            print(f"\n【总体统计】")
            print(f"  总样本数: {total_samples}")
            print(f"  错误样本数: {total_errors}")
            print(f"  准确率: {avg_acc*100:.2f}%")
            print(f"  错误率: {(total_errors/total_samples)*100:.2f}%")

            # <configured>. 按类别统计错误
            print(f"\n【各类别错误统计】")
            print(f"{'类别':<10} {'总数':>8} {'正确':>8} {'错误':>8} {'准确率':>10}")
            print("-" * 50)

            sorted_classes = sorted(class_stats.items(),
                                   key=lambda x: x[1]['wrong'],
                                   reverse=True)

            for class_id, stats in sorted_classes:
                if stats['wrong'] > 0:
                    acc = stats['correct'] / stats['total'] * 100
                    print(f"{class_id:<10} {stats['total']:>8} {stats['correct']:>8} {stats['wrong']:>8} {acc:>9.2f}%")

            # <configured>. 最容易混淆的类别对
            print(f"\n【最容易混淆的类别对】（Top 10）")
            print("-" * 80)

            confusion_pairs = []
            for true_class, error_dict in global_error_stats.items():
                for pred_class, count in error_dict.items():
                    confusion_pairs.append((true_class, pred_class, count))

            confusion_pairs.sort(key=lambda x: x[2], reverse=True)

            print(f"{'真实类':<10} {'预测类':<10} {'错误次数':>12}")
            print("-" * 35)
            for true_c, pred_c, cnt in confusion_pairs[:10]:
                print(f"{true_c:<10} {pred_c:<10} {cnt:>12}")

            # <configured>. 多原型统计
            if self.use_multi and prototype_stats:
                print(f"\n【多原型统计】")
                print("="*80)
                print(f"每个类最多可有 {self.prototypes_per_class} 个原型")
                print(f"\n{'类别':<10} {'任务数':>10} {'平均激活原型数':>18} {'利用率':>12}")
                print("-" * 55)

                total_tasks = 0
                total_active = 0

                for class_id in sorted(prototype_stats.keys()):
                    stats = prototype_stats[class_id]
                    avg_active = stats['active_sum'] / stats['total_tasks']
                    utilization = (avg_active / stats['max_prototypes']) * 100

                    print(f"{class_id:<10} {stats['total_tasks']:>10} {avg_active:>18.2f} {utilization:>11.1f}%")

                    total_tasks += stats['total_tasks']
                    total_active += stats['active_sum']

                print("-" * 55)
                overall_avg = total_active / total_tasks if total_tasks > 0 else 0
                overall_util = (overall_avg / self.prototypes_per_class) * 100
                print(f"{'平均':<10} {total_tasks:>10} {overall_avg:>18.2f} {overall_util:>11.1f}%")

                print(f"\n【原型坍缩分析】")
                print("="*80)

                collapse_summary = self._analyze_collapse(collapse_stats)

                print(f"\n{'类别':<10} {'平均距离':>12} {'平均相似度':>14} {'方差':>12} {'坍缩状态':>12}")
                print("-" * 65)

                for class_id in sorted(collapse_summary.keys()):
                    stats = collapse_summary[class_id]
                    status = self._get_collapse_status(stats)

                    print(f"{class_id:<10} {stats['avg_distance']:>12.4f} {stats['avg_similarity']:>14.4f} {stats['avg_variance']:>12.6f} {status:>12}")

                # 全局坍缩评估
                print("\n" + "-" * 65)
                global_collapse = self._evaluate_global_collapse(collapse_summary)
                print(f"\n【全局坍缩评估】")
                print(f"  严重坍缩类别: {global_collapse['severe']} 类")
                print(f"  中度坍缩类别: {global_collapse['moderate']} 类")
                print(f"  轻度坍缩类别: {global_collapse['mild']} 类")
                print(f"  健康类别:     {global_collapse['healthy']} 类")
                print(f"  总体健康度:   {global_collapse['health_score']:.1f}%")

                # 坍缩原因分析
                if global_collapse['severe'] > 0 or global_collapse['moderate'] > 0:
                    print(f"\n【坍缩原因分析】")
                    print("  可能原因:")
                    print("  1. K-shot样本数不足（当前k_shot < prototypes_per_class）")
                    print("  2. 学习率过大，导致原型快速收敛到同一点")
                    print("  3. 某些类的样本分布高度集中")
                    print("  4. 损失函数未充分鼓励原型多样性")
                    print(f"\n  建议措施:")
                    if global_collapse['severe'] > 2:
                        print(f"  ⚠️  减少原型数: {self.prototypes_per_class} → {max(2, int(overall_avg))}")
                    print(f"  ✓  添加原型多样性正则化损失")
                    print(f"  ✓  降低学习率或使用warmup")
                    print(f"  ✓  增加k_shot数量以提供更多样本")

            print("\n" + "="*80)

        return avg_acc


    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """完整的评估协议（符合论文）"""
        n_trials = _cfg_resolve('GPN_V4.py.GPNTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('GPN_V4.py.GPNTrainer.full_evaluation.q_query', q_query)
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
                        num_tasks=_cfg_require('GPN_V4.py.GPNTrainer.full_evaluation.num_tasks')
                    )
                    loader = DataLoader(meta_test, batch_size=_cfg_require('GPN_V4.py.GPNTrainer.full_evaluation.batch_size'))

                    # 关闭内层进度条
                    acc = self.evaluate(loader, n_way, show_progress=False, show_error_stats=False)
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
    def _compute_pairwise_distances(self, prototypes):
        """
        计算原型间的成对欧氏距离

        Args:
            prototypes: [n, feature_dim]

        Returns:
            distances: list of float - 所有成对距离
        """
        n = prototypes.shape[0]
        distances = []

        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(prototypes[i] - prototypes[j]).item()
                distances.append(dist)

        return distances


    def _compute_pairwise_similarities(self, prototypes):
        """
        计算原型间的成对余弦相似度

        Args:
            prototypes: [n, feature_dim]

        Returns:
            similarities: list of float - 所有成对相似度
        """
        n = prototypes.shape[0]
        similarities = []

        # 归一化
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        for i in range(n):
            for j in range(i + 1, n):
                sim = torch.dot(prototypes_norm[i], prototypes_norm[j]).item()
                similarities.append(sim)

        return similarities


    def _analyze_collapse(self, collapse_stats):
        """
        分析每个类的坍缩情况

        Args:
            collapse_stats: dict with 'class_distances', 'class_similarities', 'class_variance'

        Returns:
            summary: {class_id: {'avg_distance': float, 'avg_similarity': float, 'avg_variance': float}}
        """
        summary = {}

        for class_id in collapse_stats['class_distances'].keys():
            distances = collapse_stats['class_distances'][class_id]
            similarities = collapse_stats['class_similarities'][class_id]
            variances = collapse_stats['class_variance'][class_id]

            summary[class_id] = {
                'avg_distance': np.mean(distances) if distances else 0.0,
                'avg_similarity': np.mean(similarities) if similarities else 0.0,
                'avg_variance': np.mean(variances) if variances else 0.0,
            }

        return summary


    def _get_collapse_status(self, stats):
        "\n        根据统计数据判断坍缩状态\n\n        判断标准:\n        - 严重坍缩: 平均距离 < <configured> AND 平均相似度 > <configured> AND 方差 < <configured>\n        - 中度坍缩: 平均距离 < <configured> AND 平均相似度 > <configured> AND 方差 < <configured>\n        - 轻度坍缩: 平均距离 < <configured> AND 平均相似度 > <configured>\n        - 健康: 其他情况\n\n        Args:\n            stats: dict with 'avg_distance', 'avg_similarity', 'avg_variance'\n\n        Returns:\n            status: str - '严重', '中度', '轻度', '健康'\n        "
        dist = stats['avg_distance']
        sim = stats['avg_similarity']
        var = stats['avg_variance']

        if dist < 0.1 and sim > 0.95 and var < 0.001:
            return "严重 ⚠️"
        elif dist < 0.3 and sim > 0.85 and var < 0.01:
            return "中度 ⚠"
        elif dist < 0.5 and sim > 0.75:
            return "轻度 !"
        else:
            return "健康 ✓"


    def _evaluate_global_collapse(self, collapse_summary):
        """
        评估全局坍缩情况

        Args:
            collapse_summary: dict from _analyze_collapse

        Returns:
            global_stats: dict with counts and health score
        """
        severe = 0
        moderate = 0
        mild = 0
        healthy = 0

        for class_id, stats in collapse_summary.items():
            status = self._get_collapse_status(stats)

            if "严重" in status:
                severe += 1
            elif "中度" in status:
                moderate += 1
            elif "轻度" in status:
                mild += 1
            else:
                healthy += 1

        total = len(collapse_summary)
        health_score = (healthy * 100 + mild * 60 + moderate * 30) / total if total > 0 else 0

        return {
            'severe': severe,
            'moderate': moderate,
            'mild': mild,
            'healthy': healthy,
            'health_score': health_score
        }


class DecoupledAxisAttention(nn.Module):
    """
    解耦的轴注意力 - 针对RF信号时频特性

    核心思想：
    - 频率轴：强注意力（物理相关）
    - 时间轴：弱注意力（随机性高）
    """
    def __init__(self, channels, freq_time_ratio=None):
        freq_time_ratio = _cfg_resolve('GPN_V4.py.DecoupledAxisAttention.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()
        self.freq_weight = freq_time_ratio
        self.time_weight = 1

        # 频率轴注意力（沿时间维度池化，保留频率信息）
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # [B, C, H, <configured>]
            nn.Conv2d(channels, channels, kernel_size=_cfg_require('GPN_V4.py.DecoupledAxisAttention.__init__.kernel_size'), padding=_cfg_require('GPN_V4.py.DecoupledAxisAttention.__init__.padding')),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # 时间轴注意力（沿频率维度池化，保留时间信息）
        self.time_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # [B, C, <configured>, W]
            nn.Conv2d(channels, channels, kernel_size=_cfg_require('GPN_V4.py.DecoupledAxisAttention.__init__.kernel_size__2'), padding=_cfg_require('GPN_V4.py.DecoupledAxisAttention.__init__.padding__2')),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 频率维度注意力（强）
        freq_att = self.freq_attention(x)  # [B, C, H, <configured>]
        x_freq = x * (1 + self.freq_weight * (freq_att - 0.5))

        # 时间维度注意力（弱）
        time_att = self.time_attention(x)  # [B, C, <configured>, W]
        x_out = x_freq * (1 + self.time_weight * (time_att - 0.5))

        return x_out


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
        stride = _cfg_resolve('GPN_V4.py.RepVGGBlock.__init__.stride', stride)
        attention_type = _cfg_resolve('GPN_V4.py.RepVGGBlock.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V4.py.RepVGGBlock.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V4.py.RepVGGBlock.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # 3x3卷积分支
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('GPN_V4.py.RepVGGBlock.__init__.kernel_size'),
                     stride=stride, padding=_cfg_require('GPN_V4.py.RepVGGBlock.__init__.padding'), bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 1x1卷积分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('GPN_V4.py.RepVGGBlock.__init__.kernel_size__2'),
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
        drop_prob = _cfg_resolve('GPN_V4.py.DropPath.__init__.drop_prob', drop_prob)
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
        stride = _cfg_resolve('GPN_V4.py.ConvNeXtBlock.__init__.stride', stride)
        expansion = _cfg_resolve('GPN_V4.py.ConvNeXtBlock.__init__.expansion', expansion)
        drop_path = _cfg_resolve('GPN_V4.py.ConvNeXtBlock.__init__.drop_path', drop_path)
        attention_type = _cfg_resolve('GPN_V4.py.ConvNeXtBlock.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V4.py.ConvNeXtBlock.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V4.py.ConvNeXtBlock.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()

        # 如果stride><configured>，使用下采样
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=_cfg_require('GPN_V4.py.ConvNeXtBlock.__init__.kernel_size__3'),
                         stride=_cfg_require('GPN_V4.py.ConvNeXtBlock.__init__.stride__2'), groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=_cfg_require('GPN_V4.py.ConvNeXtBlock.__init__.kernel_size__4'), bias=False),
            )
            in_channels = out_channels

        # 大kernel深度卷积
        self.dwconv = nn.Conv2d(in_channels, in_channels,
                               kernel_size=_cfg_require('GPN_V4.py.ConvNeXtBlock.__init__.kernel_size'), padding=_cfg_require('GPN_V4.py.ConvNeXtBlock.__init__.padding'),
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
                                     kernel_size=_cfg_require('GPN_V4.py.ConvNeXtBlock.__init__.kernel_size__2'), bias=False)

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

class RelativePositionAttention2D(nn.Module):
    """
    相对位置注意力 - 2D版本(可插入ResNeXt)
    将序列展平后应用相对位置注意力,适用于时频图等2D数据
    """
    def __init__(self, channels, reduction=None, max_relative_position=None):
        '\n        Args:\n            channels: 输入通道数\n            reduction: 降维比例(用于减少计算量)\n            max_relative_position: 最大相对位置(由外部配置提供)\n        '
        reduction = _cfg_resolve('GPN_V4.py.RelativePositionAttention2D.__init__.reduction', reduction)
        max_relative_position = _cfg_resolve('GPN_V4.py.RelativePositionAttention2D.__init__.max_relative_position', max_relative_position)
        super().__init__()
        self.channels = channels
        self.max_relative_position = max_relative_position

        # 降维以减少计算量
        inner_dim = max(channels // reduction, 32)
        self.heads = _cfg_require('GPN_V4.py.RelativePositionAttention2D.__init__.heads')  # 多头注意力
        self.head_dim = inner_dim // self.heads

        # 相对位置嵌入表
        self.relative_position_bias = nn.Parameter(
            torch.randn(2 * max_relative_position - 1, self.heads)
        )

        # 通道降维
        self.reduce = nn.Conv2d(channels, inner_dim, _cfg_require('GPN_V4.py.RelativePositionAttention2D.__init__.Conv2d_arg2'))

        # QKV投影
        self.to_qkv = nn.Conv2d(inner_dim, inner_dim * 3, _cfg_require('GPN_V4.py.RelativePositionAttention2D.__init__.Conv2d_arg2__2'), bias=False)

        # 输出投影(恢复通道数)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, _cfg_require('GPN_V4.py.RelativePositionAttention2D.__init__.Conv2d_arg2__3')),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] (保持形状不变)
        """
        B, C, H, W = x.shape
        identity = x

        # 降维
        x_reduced = self.reduce(x)  # [B, inner_dim, H, W]

        # 展平空间维度
        x_flat = x_reduced.flatten(2).transpose(1, 2)  # [B, H*W, inner_dim]
        N = H * W

        # QKV投影
        qkv = self.to_qkv(x_reduced).flatten(2)  # [B, inner_dim*<configured>, H*W]
        qkv = qkv.reshape(B, 3, self.heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        # 注意力分数
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, heads, N, N]

        # 添加相对位置偏置
        relative_bias = self._get_relative_position_bias(N)
        attn = attn + relative_bias.unsqueeze(0)  # [B, heads, N, N]

        attn = attn.softmax(dim=-1)

        # 应用注意力
        out = (attn @ v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, -1).transpose(1, 2)  # [B, inner_dim, N]
        out = out.reshape(B, -1, H, W)  # [B, inner_dim, H, W]

        # 输出投影
        out = self.to_out(out)

        return out + identity  # 残差连接

    def _get_relative_position_bias(self, seq_len):
        """获取相对位置偏置"""
        device = self.relative_position_bias.device

        # 限制序列长度以避免内存溢出
        max_len = min(seq_len, self.max_relative_position)

        if seq_len <= self.max_relative_position:
            coords = torch.arange(seq_len, device=device)
            relative_coords = coords[:, None] - coords[None, :]
            relative_coords += self.max_relative_position - 1
            relative_coords = torch.clamp(relative_coords, 0, 2 * self.max_relative_position - 2)
            return self.relative_position_bias[relative_coords].permute(2, 0, 1)
        else:
            # 对于过大的特征图,使用分块策略
            return torch.zeros(self.heads, seq_len, seq_len, device=device)


class RFRelativePositionAttention2D(nn.Module):
    """
    RF信号专用的相对位置注意力 - 可插入ResNeXt版本
    针对时频图的频率和时间双轴设计
    """
    def __init__(self, channels, reduction=None, heads=None):
        """
        Args:
            channels: 输入通道数
            reduction: 降维比例
            heads: 注意力头数
        """
        reduction = _cfg_resolve('GPN_V4.py.RFRelativePositionAttention2D.__init__.reduction', reduction)
        heads = _cfg_resolve('GPN_V4.py.RFRelativePositionAttention2D.__init__.heads', heads)
        super().__init__()
        self.channels = channels
        self.heads = heads

        # 降维以减少计算量
        inner_dim = max(channels // reduction, 32)
        self.inner_dim = inner_dim

        # 动态相对位置编码(支持不同尺寸的输入)
        self.max_freq = _cfg_require('GPN_V4.py.RFRelativePositionAttention2D.__init__.size_or_budget')  # 最大频率bins
        self.max_time = _cfg_require('GPN_V4.py.RFRelativePositionAttention2D.__init__.size_or_budget__2')  # 最大时间steps

        self.freq_relative_bias = nn.Parameter(
            torch.randn(2 * self.max_freq - 1, heads)
        )
        self.time_relative_bias = nn.Parameter(
            torch.randn(2 * self.max_time - 1, heads)
        )

        # 通道降维
        self.reduce = nn.Conv2d(channels, inner_dim, _cfg_require('GPN_V4.py.RFRelativePositionAttention2D.__init__.Conv2d_arg2'))

        # QKV投影
        self.to_qkv = nn.Conv2d(inner_dim, inner_dim * 3, _cfg_require('GPN_V4.py.RFRelativePositionAttention2D.__init__.Conv2d_arg2__2'), bias=False)

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, _cfg_require('GPN_V4.py.RFRelativePositionAttention2D.__init__.Conv2d_arg2__3')),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - H:频率维度, W:时间维度
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x

        # 降维
        x_reduced = self.reduce(x)  # [B, inner_dim, H, W]

        # QKV投影
        qkv = self.to_qkv(x_reduced).reshape(B, 3, self.heads,
                                              self.inner_dim // self.heads, H, W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, heads, C//heads, H, W]

        scale = (self.inner_dim // self.heads) ** -0.5

        # === 频率轴注意力 ===
        q_f = q.mean(dim=-1)  # [B, heads, C//heads, H]
        k_f = k.mean(dim=-1)
        v_f = v.mean(dim=-1)

        attn_f = (q_f.transpose(-2, -1) @ k_f) * scale  # [B, heads, H, H]
        attn_f = attn_f + self._get_relative_bias(self.freq_relative_bias, H, self.max_freq)
        attn_f = F.softmax(attn_f, dim=-1)

        out_f = (attn_f @ v_f.transpose(-2, -1)).transpose(-2, -1)  # [B, heads, C//heads, H]
        out_f = out_f.unsqueeze(-1).expand(-1, -1, -1, -1, W)  # [B, heads, C//heads, H, W]

        # === 时间轴注意力 ===
        q_t = q.mean(dim=-2)  # [B, heads, C//heads, W]
        k_t = k.mean(dim=-2)
        v_t = v.mean(dim=-2)

        attn_t = (q_t.transpose(-2, -1) @ k_t) * scale  # [B, heads, W, W]
        attn_t = attn_t + self._get_relative_bias(self.time_relative_bias, W, self.max_time)
        attn_t = F.softmax(attn_t, dim=-1)

        out_t = (attn_t @ v_t.transpose(-2, -1)).transpose(-2, -1)  # [B, heads, C//heads, W]
        out_t = out_t.unsqueeze(-2).expand(-1, -1, -1, H, -1)  # [B, heads, C//heads, H, W]

        # === 融合双轴注意力 ===
        out = (out_f + out_t) / 2  # 平均融合
        out = out.reshape(B, self.inner_dim, H, W)

        # 输出投影
        out = self.to_out(out)

        return out + identity  # 残差连接

    def _get_relative_bias(self, bias_table, size, max_size):
        """获取相对位置偏置"""
        if size > max_size:
            # 如果输入尺寸超过预设最大值,返回零偏置
            return torch.zeros(self.heads, size, size,
                             device=bias_table.device, dtype=bias_table.dtype)

        coords = torch.arange(size, device=bias_table.device)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += max_size - 1
        relative_coords = torch.clamp(relative_coords, 0, 2 * max_size - 2)

        return bias_table[relative_coords].permute(2, 0, 1)  # [heads, size, size]
class OptimizedRFRelativeAttention(nn.Module):
    '\n    优化版RF相对位置注意力\n    \n    优化策略:\n    <configured>. 预计算相对位置索引(避免重复计算)\n    <configured>. 因式分解位置编码(减少参数量)\n    <configured>. 使用Embedding替代Parameter(更高效)\n    <configured>. 分离频率/时间的头(减少冗余)\n    '
    def __init__(self, channels, freq_bins=None, time_steps=None,
                 reduction=None, heads=None):
        """
        Args:
            channels: 输入通道数
            freq_bins: 固定频率bins(如果已知),用于预计算索引
            time_steps: 固定时间steps(如果已知)
            reduction: 降维比例
            heads: 注意力头数(必须是偶数)
        """
        reduction = _cfg_resolve('GPN_V4.py.OptimizedRFRelativeAttention.__init__.reduction', reduction)
        heads = _cfg_resolve('GPN_V4.py.OptimizedRFRelativeAttention.__init__.heads', heads)
        super().__init__()
        assert heads % 2 == 0, "heads必须是偶数以便分离频率/时间头"

        self.channels = channels
        self.heads = heads
        self.freq_heads = heads // 2  # 频率轴专用头
        self.time_heads = heads // 2  # 时间轴专用头

        # 降维
        inner_dim = max(channels // reduction, 32)
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // heads

        # Embedding内部有优化的查表实现,比直接索引Parameter快
        max_freq = freq_bins if freq_bins else _cfg_require('GPN_V4.py.OptimizedRFRelativeAttention.__init__.size_or_budget')
        max_time = time_steps if time_steps else _cfg_require('GPN_V4.py.OptimizedRFRelativeAttention.__init__.size_or_budget__2')

        self.freq_bias = nn.Embedding(2 * max_freq - 1, self.freq_heads)
        self.time_bias = nn.Embedding(2 * max_time - 1, self.time_heads)

        if freq_bins is not None:
            self.register_buffer('freq_indices',
                self._precompute_indices(freq_bins, max_freq))
        else:
            self.freq_indices = None

        if time_steps is not None:
            self.register_buffer('time_indices',
                self._precompute_indices(time_steps, max_time))
        else:
            self.time_indices = None

        self.max_freq = max_freq
        self.max_time = max_time

        # 通道降维
        self.reduce = nn.Conv2d(channels, inner_dim, _cfg_require('GPN_V4.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2'))

        self.to_q = nn.Conv2d(inner_dim, inner_dim, _cfg_require('GPN_V4.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2__2'), bias=False)
        self.to_k = nn.Conv2d(inner_dim, inner_dim, _cfg_require('GPN_V4.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2__3'), bias=False)
        self.to_v = nn.Conv2d(inner_dim, inner_dim, _cfg_require('GPN_V4.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2__4'), bias=False)

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, _cfg_require('GPN_V4.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2__5')),
            nn.BatchNorm2d(channels)
        )

    def _precompute_indices(self, size, max_size):
        """
        预计算相对位置索引
        这样在forward时只需查表,不需要重复计算
        """
        coords = torch.arange(size)
        relative_coords = coords[:, None] - coords[None, :]  # [size, size]
        relative_coords += max_size - 1  # 转换为正索引
        # Clamp确保索引在有效范围内
        relative_coords = torch.clamp(relative_coords, 0, 2 * max_size - 2)
        return relative_coords.long()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - H:频率, W:时间
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x

        # 降维
        x_reduced = self.reduce(x)  # [B, inner_dim, H, W]

        q = self.to_q(x_reduced)  # [B, inner_dim, H, W]
        k = self.to_k(x_reduced)
        v = self.to_v(x_reduced)

        # 重塑为多头格式
        q = q.reshape(B, self.heads, self.head_dim, H, W)
        k = k.reshape(B, self.heads, self.head_dim, H, W)
        v = v.reshape(B, self.heads, self.head_dim, H, W)

        # === 分离频率头和时间头 ===
        q_freq = q[:, :self.freq_heads]  # [B, freq_heads, head_dim, H, W]
        k_freq = k[:, :self.freq_heads]
        v_freq = v[:, :self.freq_heads]

        q_time = q[:, self.freq_heads:]  # [B, time_heads, head_dim, H, W]
        k_time = k[:, self.freq_heads:]
        v_time = v[:, self.freq_heads:]

        scale = self.head_dim ** -0.5

        # === 频率轴注意力(使用频率专用头) ===
        out_f = self._frequency_attention(
            q_freq, k_freq, v_freq, H, W, scale
        )  # [B, freq_heads, head_dim, H, W]

        # === 时间轴注意力(使用时间专用头) ===
        out_t = self._time_attention(
            q_time, k_time, v_time, H, W, scale
        )  # [B, time_heads, head_dim, H, W]

        # === 拼接频率头和时间头 ===
        out = torch.cat([out_f, out_t], dim=1)  # [B, heads, head_dim, H, W]
        out = out.reshape(B, self.inner_dim, H, W)

        # 输出投影
        out = self.to_out(out)

        return out + identity

    def _frequency_attention(self, q, k, v, H, W, scale):
        """
        频率轴注意力(沿高度维度)

        优化:使用预计算的索引和Embedding查表
        """
        # 沿时间维度平均(关注频率关系)
        q_f = q.mean(dim=-1)  # [B, freq_heads, head_dim, H]
        k_f = k.mean(dim=-1)
        v_f = v.mean(dim=-1)

        # 注意力分数
        attn = (q_f.transpose(-2, -1) @ k_f) * scale  # [B, freq_heads, H, H]

        # === 优化:使用预计算索引或动态计算 ===
        if self.freq_indices is not None and H == self.freq_indices.shape[0]:
            # 使用预计算的索引(最快)
            relative_bias = self.freq_bias(self.freq_indices)  # [H, H, freq_heads]
            relative_bias = relative_bias.permute(2, 0, 1)  # [freq_heads, H, H]
        else:
            # 动态计算(处理不同尺寸输入)
            relative_bias = self._get_dynamic_bias(
                self.freq_bias, H, self.max_freq
            )

        attn = attn + relative_bias.unsqueeze(0)  # [B, freq_heads, H, H]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = (attn @ v_f.transpose(-2, -1)).transpose(-2, -1)  # [B, freq_heads, head_dim, H]
        out = out.unsqueeze(-1).expand(-1, -1, -1, -1, W)  # [B, freq_heads, head_dim, H, W]

        return out

    def _time_attention(self, q, k, v, H, W, scale):
        """
        时间轴注意力(沿宽度维度)

        优化:使用预计算的索引和Embedding查表
        """
        # 沿频率维度平均(关注时间关系)
        q_t = q.mean(dim=-2)  # [B, time_heads, head_dim, W]
        k_t = k.mean(dim=-2)
        v_t = v.mean(dim=-2)

        # 注意力分数
        attn = (q_t.transpose(-2, -1) @ k_t) * scale  # [B, time_heads, W, W]

        # === 优化:使用预计算索引或动态计算 ===
        if self.time_indices is not None and W == self.time_indices.shape[0]:
            # 使用预计算的索引(最快)
            relative_bias = self.time_bias(self.time_indices)  # [W, W, time_heads]
            relative_bias = relative_bias.permute(2, 0, 1)  # [time_heads, W, W]
        else:
            # 动态计算(处理不同尺寸输入)
            relative_bias = self._get_dynamic_bias(
                self.time_bias, W, self.max_time
            )

        attn = attn + relative_bias.unsqueeze(0)  # [B, time_heads, W, W]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = (attn @ v_t.transpose(-2, -1)).transpose(-2, -1)  # [B, time_heads, head_dim, W]
        out = out.unsqueeze(-2).expand(-1, -1, -1, H, -1)  # [B, time_heads, head_dim, H, W]

        return out

    def _get_dynamic_bias(self, bias_embedding, size, max_size):
        """
        动态计算相对位置偏置(用于处理未预计算的尺寸)
        """
        if size > max_size:
            # 尺寸超出预设范围,返回零偏置
            num_heads = bias_embedding.embedding_dim
            return torch.zeros(
                num_heads, size, size,
                device=bias_embedding.weight.device,
                dtype=bias_embedding.weight.dtype
            )

        # 动态计算索引
        coords = torch.arange(size, device=bias_embedding.weight.device)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += max_size - 1
        relative_coords = torch.clamp(relative_coords, 0, 2 * max_size - 2)

        # Embedding查表
        relative_bias = bias_embedding(relative_coords.long())  # [size, size, num_heads]
        return relative_bias.permute(2, 0, 1)  # [num_heads, size, size]


# ============ 进一步优化:使用Flash Attention风格 ============

class FlashStyleRFAttention(nn.Module):
    '\n    Flash Attention风格的RF注意力\n    \n    核心优化:\n    <configured>. 分块计算注意力(减少峰值显存)\n    <configured>. 融合操作(减少内存访问)\n    <configured>. 在线softmax(避免存储完整注意力矩阵)\n    '
    def __init__(self, channels, reduction=None, heads=None,
                 freq_chunk_size=None, time_chunk_size=None):
        reduction = _cfg_resolve('GPN_V4.py.FlashStyleRFAttention.__init__.reduction', reduction)
        heads = _cfg_resolve('GPN_V4.py.FlashStyleRFAttention.__init__.heads', heads)
        freq_chunk_size = _cfg_resolve('GPN_V4.py.FlashStyleRFAttention.__init__.freq_chunk_size', freq_chunk_size)
        time_chunk_size = _cfg_resolve('GPN_V4.py.FlashStyleRFAttention.__init__.time_chunk_size', time_chunk_size)
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.freq_chunk_size = freq_chunk_size
        self.time_chunk_size = time_chunk_size

        inner_dim = max(channels // reduction, 32)
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // heads

        # 相对位置编码(分块尺寸)
        self.freq_bias = nn.Embedding(2 * freq_chunk_size - 1, heads // 2)
        self.time_bias = nn.Embedding(2 * time_chunk_size - 1, heads // 2)

        self.reduce = nn.Conv2d(channels, inner_dim, _cfg_require('GPN_V4.py.FlashStyleRFAttention.__init__.Conv2d_arg2'))
        self.to_qkv = nn.Conv2d(inner_dim, inner_dim * 3, _cfg_require('GPN_V4.py.FlashStyleRFAttention.__init__.Conv2d_arg2__2'), bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, _cfg_require('GPN_V4.py.FlashStyleRFAttention.__init__.Conv2d_arg2__3')),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        """
        分块计算注意力,显著减少显存峰值
        """
        B, C, H, W = x.shape
        identity = x

        x_reduced = self.reduce(x)
        qkv = self.to_qkv(x_reduced).reshape(
            B, 3, self.heads, self.head_dim, H, W
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # 分离频率/时间头
        q_f, q_t = q.chunk(2, dim=1)
        k_f, k_t = k.chunk(2, dim=1)
        v_f, v_t = v.chunk(2, dim=1)

        # === 分块频率注意力 ===
        out_f = self._chunked_frequency_attention(
            q_f, k_f, v_f, H, W
        )

        # === 分块时间注意力 ===
        out_t = self._chunked_time_attention(
            q_t, k_t, v_t, H, W
        )

        out = torch.cat([out_f, out_t], dim=1)
        out = out.reshape(B, self.inner_dim, H, W)

        out = self.to_out(out)
        return out + identity

    def _chunked_frequency_attention(self, q, k, v, H, W):
        """
        分块计算频率轴注意力
        关键:一次只计算一个chunk的注意力,避免存储完整的H×H矩阵
        """
        B, num_heads = q.shape[:2]
        chunk_size = self.freq_chunk_size

        # 平均时间维度
        q_f = q.mean(dim=-1)  # [B, heads, head_dim, H]
        k_f = k.mean(dim=-1)
        v_f = v.mean(dim=-1)

        # 初始化输出
        output = torch.zeros_like(q_f)
        scale = self.head_dim ** -0.5

        # 按chunk处理
        for i in range(0, H, chunk_size):
            end_i = min(i + chunk_size, H)
            q_chunk = q_f[:, :, :, i:end_i]  # [B, heads, head_dim, chunk]

            # 与所有k计算注意力
            attn_chunk = (q_chunk.transpose(-2, -1) @ k_f) * scale  # [B, heads, chunk, H]

            # 添加相对位置偏置(只计算当前chunk需要的部分)
            relative_bias = self._get_chunk_bias(
                self.freq_bias, i, end_i, H
            )
            attn_chunk = attn_chunk + relative_bias.unsqueeze(0)

            # Softmax和输出
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            output[:, :, :, i:end_i] = (
                attn_chunk @ v_f.transpose(-2, -1)
            ).transpose(-2, -1)

        return output.unsqueeze(-1).expand(-1, -1, -1, -1, W)

    def _chunked_time_attention(self, q, k, v, H, W):
        """分块计算时间轴注意力"""
        B, num_heads = q.shape[:2]
        chunk_size = self.time_chunk_size

        q_t = q.mean(dim=-2)  # [B, heads, head_dim, W]
        k_t = k.mean(dim=-2)
        v_t = v.mean(dim=-2)

        output = torch.zeros_like(q_t)
        scale = self.head_dim ** -0.5

        for i in range(0, W, chunk_size):
            end_i = min(i + chunk_size, W)
            q_chunk = q_t[:, :, :, i:end_i]

            attn_chunk = (q_chunk.transpose(-2, -1) @ k_t) * scale

            relative_bias = self._get_chunk_bias(
                self.time_bias, i, end_i, W
            )
            attn_chunk = attn_chunk + relative_bias.unsqueeze(0)

            attn_chunk = F.softmax(attn_chunk, dim=-1)
            output[:, :, :, i:end_i] = (
                attn_chunk @ v_t.transpose(-2, -1)
            ).transpose(-2, -1)

        return output.unsqueeze(-2).expand(-1, -1, -1, H, -1)

    def _get_chunk_bias(self, bias_embedding, start, end, total_size):
        """获取chunk对应的相对位置偏置"""
        chunk_len = end - start
        device = bias_embedding.weight.device

        # 当前chunk内的坐标
        chunk_coords = torch.arange(start, end, device=device)
        # 全部坐标
        all_coords = torch.arange(total_size, device=device)

        # 相对位置
        relative_coords = chunk_coords[:, None] - all_coords[None, :]
        relative_coords += self.freq_chunk_size - 1
        relative_coords = torch.clamp(
            relative_coords, 0, 2 * self.freq_chunk_size - 2
        )

        bias = bias_embedding(relative_coords.long())  # [chunk, total, heads]
        return bias.permute(2, 0, 1)  # [heads, chunk, total]


# ============ 更新ResNeXtBlock支持新的注意力机制 ============
class ResNeXtBlock(nn.Module):
    """
    ResNeXt Bottleneck Block

    支持的注意力机制:
    - 'se': 标准SE模块
    - 'decoupled': 解耦轴注意力
    - 'freq_priority': 频率优先注意力
    - 'relative_pos': 通用相对位置注意力(新增)
    - 'rf_relative_pos': RF信号专用相对位置注意力(新增)
    """
    def __init__(self, in_channels, out_channels, stride=None,
                 cardinality=None, attention_type=None, reduction=None,
                 freq_time_ratio=None,se_ref=False):
        stride = _cfg_resolve('GPN_V4.py.ResNeXtBlock.__init__.stride', stride)
        cardinality = _cfg_resolve('GPN_V4.py.ResNeXtBlock.__init__.cardinality', cardinality)
        attention_type = _cfg_resolve('GPN_V4.py.ResNeXtBlock.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V4.py.ResNeXtBlock.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V4.py.ResNeXtBlock.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()

        mid_channels = out_channels

        # 1x1 扩展卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=_cfg_require('GPN_V4.py.ResNeXtBlock.__init__.kernel_size'), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 分组卷积
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=_cfg_require('GPN_V4.py.ResNeXtBlock.__init__.kernel_size__2'), stride=stride,
                              padding=_cfg_require('GPN_V4.py.ResNeXtBlock.__init__.padding'), groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 压缩卷积
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                              kernel_size=_cfg_require('GPN_V4.py.ResNeXtBlock.__init__.kernel_size__3'), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.SiLU(inplace=True)

        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=_cfg_require('GPN_V4.py.ResNeXtBlock.__init__.kernel_size__4'), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        if se_ref==True:
            self.attention2 = SEModule(out_channels, reduction=reduction)
        else:
            self.attention2 = None
        # 注意力机制选择
        if attention_type == 'se':
            self.attention = SEModule(out_channels, reduction=reduction)
        elif attention_type == 'decoupled':
            self.attention = DecoupledAxisAttention(out_channels, freq_time_ratio)
        elif attention_type == 'freq_priority':
            self.attention = FrequencyPriorityCA(out_channels, reduction, freq_time_ratio)
        elif attention_type == 'relative_pos':
            # 新增:通用相对位置注意力
            self.attention = RelativePositionAttention2D(out_channels, reduction=reduction)
        elif attention_type == 'rf_relative_pos':
            # 新增:RF信号专用相对位置注意力
            self.attention = RFRelativePositionAttention2D(out_channels, reduction=reduction)
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

        if self.attention2 is not None:
            out = self.attention2(out)

        # 注意力模块(在Add之前应用)
        if self.attention is not None:
            out = self.attention(out)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

class GPN_Optimized(nn.Module):
    "\n    优化的GPN模型 - 完全兼容现有训练器\n    \n    架构特点：\n    - Stage <configured>-<configured>: 可选RepVGG（浅层，训练稳定）\n    - Stage <configured>-<configured>: 可选ConvNeXt（深层，大感受野）\n    - 全局：统一的注意力机制（SE / decoupled / freq_priority）\n    - 输出：v (embedding), s (precision) - 与原模型一致\n    \n    参数说明：\n        use_repvgg: Stage <configured>-<configured>是否使用RepVGG（默认False，使用ResNeXt）\n        use_convnext: Stage <configured>-<configured>是否使用ConvNeXt（默认False，使用ResNeXt）\n        attention_type: 注意力类型\n            - 'se': 标准SE模块（默认）\n            - 'decoupled': 解耦时频注意力\n            - 'freq_priority': 频率优先CA\n            - None: 不使用注意力\n        reduction: SE模块的reduction ratio（由外部配置提供）\n    "
    def __init__(self,
                 use_repvgg=None,
                 use_convnext=None,
                 attention_type=None,
                 reduction=None,
                 freq_time_ratio=None,
                 ca_input_channels=None):
        use_repvgg = _cfg_resolve('GPN_V4.py.GPN_Optimized.__init__.use_repvgg', use_repvgg)
        use_convnext = _cfg_resolve('GPN_V4.py.GPN_Optimized.__init__.use_convnext', use_convnext)
        attention_type = _cfg_resolve('GPN_V4.py.GPN_Optimized.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V4.py.GPN_Optimized.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V4.py.GPN_Optimized.__init__.freq_time_ratio', freq_time_ratio)
        ca_input_channels = _cfg_resolve('GPN_V4.py.GPN_Optimized.__init__.ca_input_channels', ca_input_channels)
        super().__init__()

        self.use_repvgg = use_repvgg
        self.use_convnext = use_convnext
        self.attention_type = attention_type
        self.use_ca = _cfg_require('GPN_V4.py.GPN_Optimized.__init__.use_ca')
        self.ca_position = 'input'

        if self.use_ca and self.ca_position == 'input':
            # 先用小卷积提升通道数（保持频率-时间结构）
            self.input_proj = nn.Sequential(
                nn.Conv2d(_cfg_require('GPN_V4.py.GPN_Optimized.__init__.Conv2d_arg0__2'), ca_input_channels, kernel_size=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.kernel_size__8'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__16'),
                         padding=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.padding__4'), bias=False),  # <configured>×<configured>小卷积，局部混合
                nn.BatchNorm2d(ca_input_channels),
                nn.SiLU(inplace=True)
            )

            # CA作用于低通道数的特征（更高效）
            self.ca_input = FrequencyPriorityCA(
                channels=ca_input_channels,
                reduction=max(1, ca_input_channels // 4),  # 自适应reduction
                freq_time_ratio=freq_time_ratio
            )
            # 调整Stem的输入通道
            self.conv1 = nn.Conv2d(ca_input_channels, _cfg_require('GPN_V4.py.GPN_Optimized.__init__.Conv2d_arg1'), kernel_size=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.kernel_size__6'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__6'),
                                  padding=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.padding__2'), bias=False)
        else:
            # 原始Stem
            self.conv1 = nn.Conv2d(_cfg_require('GPN_V4.py.GPN_Optimized.__init__.Conv2d_arg0'), _cfg_require('GPN_V4.py.GPN_Optimized.__init__.Conv2d_arg1__2'), kernel_size=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.kernel_size__7'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__7'),
                                  padding=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.padding__3'), bias=False)

        self.bn1 = nn.BatchNorm2d(_cfg_require('GPN_V4.py.GPN_Optimized.__init__.BatchNorm2d_arg0'))
        self.relu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.kernel_size'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride'), padding=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.padding'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_repvgg:
            self.block1 = RepVGGBlock(
                24, 48, stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__8'),
                attention_type=attention_type,
                reduction=reduction,
                freq_time_ratio=freq_time_ratio
            )
        else:
            self.block1 = ResNeXtBlock(
                _cfg_require('GPN_V4.py.GPN_Optimized.__init__.ResNeXtBlock_arg0'), _cfg_require('GPN_V4.py.GPN_Optimized.__init__.ResNeXtBlock_arg1'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__9'), cardinality=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.cardinality'),
                attention_type=attention_type,
                reduction=48 if attention_type == 'se' else reduction,
                freq_time_ratio=freq_time_ratio,
                se_ref=True
            )
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.kernel_size__2'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__2'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_repvgg:
            self.block2 = RepVGGBlock(
                48, 96, stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__10'),
                attention_type=attention_type,
                reduction=reduction,
                freq_time_ratio=freq_time_ratio
            )
        else:
            self.block2 = ResNeXtBlock(
                _cfg_require('GPN_V4.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__2'), _cfg_require('GPN_V4.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__2'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__11'), cardinality=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.cardinality__2'),
                attention_type=attention_type,
                reduction=32 if attention_type == 'se' else reduction,
                freq_time_ratio=freq_time_ratio
            )
        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.kernel_size__3'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__3'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_convnext:
            self.block3 = ConvNeXtBlock(
                96, _cfg_require('GPN_V4.py.GPN_Optimized.__init__.size_or_budget'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__12'),
                expansion=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.expansion'), drop_path=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.drop_path'),
                attention_type=attention_type,
                reduction=reduction,
                freq_time_ratio=freq_time_ratio
            )
        else:
            self.block3 = ResNeXtBlock(
                _cfg_require('GPN_V4.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__3'), _cfg_require('GPN_V4.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__3'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__13'), cardinality=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.cardinality__3'),
                attention_type=attention_type,
                reduction=24 if attention_type == 'se' else reduction,
                freq_time_ratio=freq_time_ratio
            )
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.kernel_size__4'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__4'))
        # 输出: [B, <configured>, <configured>, <configured>]

        if use_convnext:
            self.block4 = ConvNeXtBlock(
                _cfg_require('GPN_V4.py.GPN_Optimized.__init__.size_or_budget__2'), _cfg_require('GPN_V4.py.GPN_Optimized.__init__.size_or_budget__3'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__14'),
                expansion=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.expansion__2'), drop_path=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.drop_path__2'),
                attention_type=attention_type,
                reduction=reduction,
                freq_time_ratio=freq_time_ratio
            )
        else:
            self.block4 = ResNeXtBlock(
                _cfg_require('GPN_V4.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__4'), _cfg_require('GPN_V4.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__4'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__15'), cardinality=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.cardinality__4'),
                attention_type=attention_type,
                reduction=80 if attention_type == 'se' else reduction,
                freq_time_ratio=freq_time_ratio
            )
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.kernel_size__5'), stride=_cfg_require('GPN_V4.py.GPN_Optimized.__init__.stride__5'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # === Global Pooling ===
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def forward(self, x, return_intermediate=False):
        '\n        前向传播 - 对齐原模型接口\n        \n        Args:\n            x: [B, <configured>, <configured>, <configured>]\n            return_intermediate: 是否返回中间特征\n            \n        Returns:\n            v: [B, <configured>] embedding特征\n            s: [B, <configured>] precision特征\n            features: (可选) 中间特征字典\n        '
        features = {} if return_intermediate else None

        # Stem
        if self.use_ca and self.ca_position == 'input':
            # 先投影到低通道
            x = self.input_proj(x)  # [B, <configured>, <configured>, <configured>] → [B, <configured>, <configured>, <configured>]
            if return_intermediate:
                features['input_proj'] = x

            # CA作用于原始频谱结构
            x = self.ca_input(x)    # [B, <configured>, <configured>, <configured>] → [B, <configured>, <configured>, <configured>]
            if return_intermediate:
                features['ca_input'] = x
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
        v_features = x[:, :_cfg_require('GPN_V4.py.GPN_Optimized.forward.size_or_budget'), :, :]  # [B, <configured>, <configured>, <configured>]
        s_features = x[:, _cfg_require('GPN_V4.py.GPN_Optimized.forward.size_or_budget__2'):, :, :]  # [B, <configured>, <configured>, <configured>]

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
