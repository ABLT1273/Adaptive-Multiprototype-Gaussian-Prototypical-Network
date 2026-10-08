'\nInfinite Mixture Prototypes (IMP) - 全监督版本\n基于论文: Allen et al. "Infinite Mixture Prototypes for Few-Shot Learning" (ICML <configured>)\n\n适配场景：所有样本都有标签的监督学习\n核心机制：通过Dirichlet Process自适应地为每个类别创建多个原型\n'
from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from tqdm import tqdm
from typing import Tuple, List, Optional
import gc


class SupervisedIMPAlgorithm:
    '\n    全监督IMP算法 - 基于论文Algorithm <configured>改造\n    \n    核心改进：\n    <configured>. 摒弃无标签逻辑，专注监督学习\n    <configured>. 每个类别可以有多个原型（cluster）\n    <configured>. 自动推断每类的cluster数量\n    <configured>. 保持软分配机制\n    \n    理论依据：\n    - 论文<configured>节：多模态聚类\n    - 论文Algorithm <configured>：IMP推断流程\n    - 论文公式<configured>：λ阈值估计\n    - 论文公式<configured>：分类决策\n    '

    def __init__(self,
                 alpha: float = None,      # CRP浓度参数
                 sigma_init: float = None, # 初始cluster方差
                 device: torch.device = None):
        '\n        Args:\n            alpha: CRP浓度参数，控制创建新cluster的倾向\n                   - α越大 → λ越大 → 越容易创建新cluster\n                   - 建议范围: [<configured>, <configured>]\n            sigma_init: 初始化的cluster方差\n                        - 可学习参数，通过训练优化\n                        - 影响软分配和λ估计\n            device: 计算设备\n        '
        alpha = _cfg_resolve('loss_IMP.py.SupervisedIMPAlgorithm.__init__.alpha', alpha)
        sigma_init = _cfg_resolve('loss_IMP.py.SupervisedIMPAlgorithm.__init__.sigma_init', sigma_init)
        self.alpha = alpha
        self.sigma = sigma_init  # 统一使用一个可学习的方差
        self.device = device or torch.device('cpu')

    def estimate_lambda(self,
                       features: torch.Tensor,
                       labels: torch.Tensor) -> float:
        '\n        估计λ阈值 - 基于论文公式<configured>\n        \n        λ = <configured>σ log(α / (<configured> + ρ/σ)^(d/<configured>))\n        \n        理论解释：\n        - ρ: 类间原型方差（体现数据分布复杂度）\n        - σ: cluster内方差（可学习参数）\n        - α: 浓度参数（控制cluster创建倾向）\n        - d: 特征维度\n        \n        Args:\n            features: [n_samples, feature_dim] 样本特征\n            labels: [n_samples] 样本标签\n            \n        Returns:\n            lambda_threshold: 创建新cluster的距离阈值\n        '
        feature_dim = features.shape[1]

        # 估计ρ：计算各类原型间的方差
        unique_labels = torch.unique(labels)

        if len(unique_labels) > 1:
            # 计算每个类别的均值
            class_means = []
            for label in unique_labels:
                class_mask = (labels == label)
                if class_mask.sum() > 0:
                    class_mean = features[class_mask].mean(dim=0)
                    class_means.append(class_mean)

            if len(class_means) > 1:
                # 类间方差
                class_means = torch.stack(class_means)
                rho = torch.var(class_means, dim=0).mean().item()
            else:
                rho = 1.0
        else:
            rho = 1.0

        # 计算λ - 论文公式<configured>
        # 注意：需要防止log的参数为负
        ratio = rho / self.sigma
        denominator = (1 + ratio) ** (feature_dim / 2)

        if self.alpha / denominator > 0:
            lambda_val = 2 * self.sigma * math.log(self.alpha / denominator)
        else:
            lambda_val = 0.1  # 默认最小值

        # 确保λ为正（物理意义：距离阈值必须为正）
        return max(lambda_val, 0.01)

    def imp_clustering(self,
                      features: torch.Tensor,
                      labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '\n        IMP聚类算法 - 全监督版本\n        \n        流程（基于论文Algorithm <configured>）：\n        <configured>. 初始化：每类创建一个初始cluster（类均值）\n        <configured>. 估计λ阈值\n        <configured>. 遍历样本，按λ阈值决定是否创建新cluster\n        <configured>. 软分配：计算样本对cluster的归属概率\n        <configured>. 更新cluster均值\n        \n        关键约束（监督学习）：\n        - 每个样本只能分配给**同类别**的cluster\n        - 距离计算时，跨类别距离设为∞\n        \n        Args:\n            features: [n_samples, feature_dim] 嵌入特征 h_φ(x)\n            labels: [n_samples] 样本标签 y\n            \n        Returns:\n            cluster_means: [n_clusters, feature_dim] cluster均值 μ_c\n            cluster_labels: [n_clusters] cluster所属类别 l_c\n            cluster_sigmas: [n_clusters] cluster方差 σ_c\n            soft_assignments: [n_samples, n_clusters] 软分配概率 z_{i,c}\n        '
        n_samples, feature_dim = features.shape
        device = features.device

        # 为每个类别创建初始cluster（类均值）
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)

        cluster_means = []
        cluster_labels = []
        cluster_sigmas = []

        for label in unique_labels:
            class_mask = (labels == label)
            if class_mask.sum() > 0:
                # 类均值作为初始原型
                class_mean = features[class_mask].mean(dim=0)
                cluster_means.append(class_mean)
                cluster_labels.append(label.item())
                cluster_sigmas.append(self.sigma)

        lambda_threshold = self.estimate_lambda(features, labels)

        # 遍历所有样本，按照DP-means逻辑创建新cluster
        for i in range(n_samples):
            xi = features[i]      # 当前样本特征
            yi = labels[i].item() # 当前样本标签

            # 计算到所有**同类别**cluster的最小距离
            min_distance = float('inf')

            for c_idx in range(len(cluster_means)):
                lc = cluster_labels[c_idx]

                # 关键约束：只考虑同类别的cluster
                if lc == yi:
                    # 欧氏距离平方
                    distance = torch.norm(xi - cluster_means[c_idx]).item() ** 2

                    if distance < min_distance:
                        min_distance = distance

            # 如果最小距离 > λ，创建新cluster
            if min_distance > lambda_threshold:
                cluster_means.append(xi.clone())
                cluster_labels.append(yi)
                cluster_sigmas.append(self.sigma)

        # 转换为tensor
        cluster_means = torch.stack(cluster_means)
        cluster_labels = torch.tensor(cluster_labels, device=device)
        cluster_sigmas = torch.tensor(cluster_sigmas, device=device, dtype=torch.float32)

        # 计算 z_{i,c} = p(c|x_i)，只对同类别cluster有效
        soft_assignments = self._compute_soft_assignments(
            features, labels, cluster_means, cluster_labels, cluster_sigmas
        )

        cluster_means = self._update_cluster_means(
            features, soft_assignments, cluster_means
        )

        return cluster_means, cluster_labels, cluster_sigmas, soft_assignments

    def _compute_soft_assignments(self,
                                features: torch.Tensor,
                                labels: torch.Tensor,
                                cluster_means: torch.Tensor,
                                cluster_labels: torch.Tensor,
                                cluster_sigmas: torch.Tensor) -> torch.Tensor:
        "\n        计算软分配 - 基于高斯密度的归一化概率\n        \n        论文公式（Algorithm <configured>, Step <configured>）：\n        z_{i,c} = N(h_φ(x_i); μ_c, σ_c) / Σ_{c'} N(h_φ(x_i); μ_{c'}, σ_{c'})\n        \n        关键改进（监督学习）：\n        - 分母只对**同类别**的cluster求和\n        - 跨类别的归属概率设为<configured>\n        \n        Args:\n            features: [n_samples, feature_dim]\n            labels: [n_samples]\n            cluster_means: [n_clusters, feature_dim]  \n            cluster_labels: [n_clusters]\n            cluster_sigmas: [n_clusters]\n            \n        Returns:\n            soft_assignments: [n_samples, n_clusters]\n                             z_{i,c} = <configured> if label(x_i) ≠ label(cluster_c)\n        "
        n_samples = features.shape[0]
        n_clusters = cluster_means.shape[0]
        feature_dim = features.shape[1]

        # 初始化log概率矩阵
        log_probs = torch.full((n_samples, n_clusters), -float('inf'),
                              device=features.device)

        # 计算每个样本对每个cluster的高斯密度
        for c in range(n_clusters):
            mu_c = cluster_means[c]
            sigma_c = cluster_sigmas[c]
            lc = cluster_labels[c].item()

            # 找到属于该cluster类别的样本
            same_class_mask = (labels == lc)

            if same_class_mask.sum() > 0:
                # 计算距离
                diff = features[same_class_mask] - mu_c  # [n_same_class, feature_dim]
                squared_dist = torch.sum(diff ** 2, dim=1)  # [n_same_class]

                # 高斯密度的log形式
                log_gaussian = -0.5 * (
                    squared_dist / (sigma_c ** 2) +
                    feature_dim * math.log(2 * math.pi * sigma_c ** 2)
                )

                # 只对同类别样本有效
                log_probs[same_class_mask, c] = log_gaussian

        # Softmax归一化（按行）
        soft_assignments = F.softmax(log_probs, dim=1)

        return soft_assignments

    def _update_cluster_means(self,
                            features: torch.Tensor,
                            soft_assignments: torch.Tensor,
                            old_means: torch.Tensor) -> torch.Tensor:
        '\n        更新cluster均值 - 加权平均\n        \n        论文公式（Algorithm <configured>, Step <configured>）：\n        μ_c = Σ_i z_{i,c} h_φ(x_i) / Σ_i z_{i,c}\n        \n        Args:\n            features: [n_samples, feature_dim]\n            soft_assignments: [n_samples, n_clusters]\n            old_means: [n_clusters, feature_dim]\n            \n        Returns:\n            new_means: [n_clusters, feature_dim]\n        '
        n_clusters = old_means.shape[0]
        new_means = torch.zeros_like(old_means)

        for c in range(n_clusters):
            zi_c = soft_assignments[:, c]  # [n_samples]
            weight_sum = torch.sum(zi_c)

            if weight_sum > 1e-8:
                # 加权平均
                weighted_features = zi_c.unsqueeze(1) * features  # [n_samples, feature_dim]
                new_means[c] = torch.sum(weighted_features, dim=0) / weight_sum
            else:
                # 如果没有样本分配到该cluster，保持原均值
                new_means[c] = old_means[c]

        return new_means

    def classify_query(self,
                      query_features: torch.Tensor,
                      cluster_means: torch.Tensor,
                      cluster_labels: torch.Tensor,
                      cluster_sigmas: torch.Tensor,
                      use_mahalanobis: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        "\n        分类查询点 - 基于论文公式<configured>\n        \n        论文分类规则：\n        p(y'=n | x') = exp(-d(h_φ(x'), μ_{c*_n})) / Σ_{n'} exp(-d(h_φ(x'), μ_{c*_{n'}}))\n        \n        其中 c*_n = argmin_{c: l_c=n} d(h_φ(x'), μ_c)\n        \n        解释：对每个类别，找到最近的cluster，然后用softmax计算概率\n        \n        Args:\n            query_features: [n_query, feature_dim] 查询样本特征\n            cluster_means: [n_clusters, feature_dim]\n            cluster_labels: [n_clusters]\n            cluster_sigmas: [n_clusters]\n            use_mahalanobis: 是否使用马氏距离（默认欧氏）\n            \n        Returns:\n            class_probs: [n_query, n_classes] 类别概率分布\n            predictions: [n_query] 预测类别\n        "
        use_mahalanobis = _cfg_resolve('loss_IMP.py.SupervisedIMPAlgorithm.classify_query.use_mahalanobis', use_mahalanobis)
        n_query = query_features.shape[0]
        unique_classes = torch.unique(cluster_labels).cpu().numpy()
        n_classes = len(unique_classes)

        # 为每个查询点找到每个类别的最近cluster
        log_probs = torch.zeros(n_query, n_classes, device=query_features.device)

        for class_idx, class_label in enumerate(unique_classes):
            # 该类别的所有cluster
            class_cluster_mask = (cluster_labels == class_label)
            class_cluster_indices = torch.where(class_cluster_mask)[0]

            # 计算到该类所有cluster的距离
            min_distances = torch.full((n_query,), float('inf'),
                                      device=query_features.device)

            for c_idx in class_cluster_indices:
                mu_c = cluster_means[c_idx]
                sigma_c = cluster_sigmas[c_idx]

                if use_mahalanobis:
                    # 简化版：Σ = σ²I
                    diff = query_features - mu_c
                    distances = torch.sum(diff ** 2, dim=1) / (sigma_c ** 2)
                else:
                    # 欧氏距离平方
                    distances = torch.sum((query_features - mu_c) ** 2, dim=1)

                # 保留最小距离
                min_distances = torch.minimum(min_distances, distances)

            # 负距离用于softmax（距离越小，概率越大）
            log_probs[:, class_idx] = -min_distances

        # Softmax归一化
        class_probs = F.softmax(log_probs, dim=1)
        predictions = torch.argmax(class_probs, dim=1)

        # 将预测映射回原始标签
        predictions = torch.tensor([unique_classes[p] for p in predictions.cpu()],
                                   device=query_features.device)

        return class_probs, predictions


class SupervisedIMPPrototypeGenerator:
    '\n    全监督IMP原型生成器\n    \n    封装IMP算法，提供类似原有框架的接口\n    \n    使用方式：\n    <configured>. 初始化生成器\n    <configured>. 调用compute_imp_prototypes生成原型\n    <configured>. 使用SupervisedIMPLoss计算损失\n    '

    def __init__(self,
                 alpha: float = None,
                 sigma_init: float = None,
                 device: torch.device = None):
        """
        Args:
            alpha: CRP浓度参数
            sigma_init: 初始cluster方差
            device: 计算设备
        """
        alpha = _cfg_resolve('loss_IMP.py.SupervisedIMPPrototypeGenerator.__init__.alpha', alpha)
        sigma_init = _cfg_resolve('loss_IMP.py.SupervisedIMPPrototypeGenerator.__init__.sigma_init', sigma_init)
        self.imp_algo = SupervisedIMPAlgorithm(
            alpha=alpha,
            sigma_init=sigma_init,
            device=device
        )
        self.device = device or torch.device('cpu')

    def compute_imp_prototypes(self,
                              features: torch.Tensor,
                              labels: torch.Tensor,
                              n_ways: int,
                              verbose: bool = False) -> Tuple[List[torch.Tensor],
                                                              List[torch.Tensor],
                                                              List[int]]:
        """
        计算IMP原型

        Args:
            features: [n_samples, feature_dim] 特征向量
            labels: [n_samples] 标签
            n_ways: 类别数量
            verbose: 是否打印详细信息

        Returns:
            all_prototypes: List[Tensor], 每个元素是一个类的原型 [n_protos_c, feature_dim]
            all_precision_matrices: List[Tensor], 每个元素是一个类的精度矩阵（协方差逆）
            prototypes_per_class: List[int], 每个类的原型数量
        """
        # 运行IMP聚类
        cluster_means, cluster_labels, cluster_sigmas, soft_assignments = \
            self.imp_algo.imp_clustering(features, labels)

        # 按类别组织原型
        unique_labels = torch.unique(labels).cpu().numpy()
        all_prototypes = []
        all_precision_matrices = []
        prototypes_per_class = _cfg_require('loss_IMP.py.SupervisedIMPPrototypeGenerator.compute_imp_prototypes.prototypes_per_class')

        for class_label in unique_labels:
            # 该类别的所有cluster
            class_mask = (cluster_labels == class_label)
            class_prototypes = cluster_means[class_mask]  # [n_protos_c, feature_dim]
            class_sigmas = cluster_sigmas[class_mask]     # [n_protos_c]

            n_protos = class_prototypes.shape[0]
            feature_dim = class_prototypes.shape[1]

            precision_matrices = []
            for sigma in class_sigmas:
                precision = torch.eye(feature_dim, device=self.device) / (sigma ** 2)
                precision_matrices.append(precision)

            all_prototypes.append(class_prototypes)
            all_precision_matrices.append(torch.stack(precision_matrices))
            prototypes_per_class.append(n_protos)

            if verbose:
                print(f"类别 {class_label}: {n_protos} 个原型")

        if verbose:
            total_prototypes = sum(prototypes_per_class)
            avg_prototypes = total_prototypes / len(prototypes_per_class)
            print(f"\n总原型数: {total_prototypes}")
            print(f"平均每类: {avg_prototypes:.2f} 个原型")
            print(f"λ阈值: {self.imp_algo.estimate_lambda(features, labels):.4f}")

        return all_prototypes, all_precision_matrices, prototypes_per_class

    def update_sigma(self, new_sigma: float):
        """更新cluster方差（用于训练过程中的学习）"""
        self.imp_algo.sigma = new_sigma


class SupervisedIMPLoss(nn.Module):
    '\n    全监督IMP损失函数\n    \n    基于论文公式<configured>和<configured>：\n    <configured>. 找到每个类别最近的cluster\n    <configured>. 计算masked cross-entropy loss\n    \n    优势：\n    - 只对最近cluster计算损失，避免过度惩罚多模态\n    - 支持多原型的灵活表示\n    '

    def __init__(self, distance_metric: str = 'E'):
        """
        Args:
            distance_metric: 距离度量
                'E' - 欧氏距离
                'M' - 马氏距离
        """
        super().__init__()
        self.distance_metric = distance_metric

    def forward(self,
                query_features: torch.Tensor,
                all_prototypes: List[torch.Tensor],
                all_precision_matrices: List[torch.Tensor],
                query_labels: torch.Tensor,
                prototypes_per_class: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算IMP损失

        Args:
            query_features: [n_query, feature_dim]
            all_prototypes: List of [n_protos_c, feature_dim]
            all_precision_matrices: List of [n_protos_c, feature_dim, feature_dim]
            query_labels: [n_query]
            prototypes_per_class: List[int]

        Returns:
            loss: 标量损失
            class_probs: [n_query, n_classes] 类别概率
            min_distances: [n_query, n_classes] 每类最小距离
        """
        n_query = query_features.shape[0]
        n_classes = len(all_prototypes)
        device = query_features.device

        # 计算到每个类别最近cluster的距离
        min_distances = torch.zeros(n_query, n_classes, device=device)

        for class_idx in range(n_classes):
            class_prototypes = all_prototypes[class_idx]  # [n_protos_c, feature_dim]
            class_precisions = all_precision_matrices[class_idx]  # [n_protos_c, d, d]

            n_protos = class_prototypes.shape[0]

            # 计算到该类所有原型的距离
            class_distances = torch.full((n_query,), float('inf'), device=device)

            for proto_idx in range(n_protos):
                proto = class_prototypes[proto_idx]

                if self.distance_metric == 'E':
                    # 欧氏距离
                    distances = torch.sum((query_features - proto) ** 2, dim=1)
                else:  # 'M'
                    # 马氏距离
                    precision = class_precisions[proto_idx]
                    diff = query_features - proto  # [n_query, feature_dim]
                    distances = torch.sum(diff @ precision * diff, dim=1)

                # 保留最小距离
                class_distances = torch.minimum(class_distances, distances)

            min_distances[:, class_idx] = class_distances

        # 计算类别概率（负距离的softmax）
        log_probs = -min_distances
        class_probs = F.softmax(log_probs, dim=1)

        # 交叉熵损失
        loss = F.cross_entropy(log_probs, query_labels)

        return loss, class_probs, min_distances


# ============ 使用示例 ============

def usage_example():
    """
    全监督IMP使用示例
    """
    print("=== 全监督IMP使用示例 ===\n")

    # <configured>. 创建模拟数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_ways = _cfg_require('loss_IMP.py.usage_example.n_ways')
    k_shot = _cfg_require('loss_IMP.py.usage_example.k_shot')
    feature_dim = _cfg_require('loss_IMP.py.usage_example.feature_dim')
    n_query = _cfg_require('loss_IMP.py.usage_example.n_query')

    # Support set
    support_features = torch.randn(n_ways * k_shot, feature_dim, device=device)
    support_labels = torch.repeat_interleave(torch.arange(n_ways), k_shot).to(device)

    # Query set
    query_features = torch.randn(n_query, feature_dim, device=device)
    query_labels = torch.randint(0, n_ways, (n_query,), device=device)

    print(f"Support: {n_ways}-way {k_shot}-shot = {n_ways * k_shot} samples")
    print(f"Query: {n_query} samples\n")

    # <configured>. 初始化IMP生成器
    imp_generator = SupervisedIMPPrototypeGenerator(
        alpha=_cfg_require('loss_IMP.py.usage_example.alpha'),        # CRP浓度参数
        sigma_init=_cfg_require('loss_IMP.py.usage_example.sigma_init'),   # 初始方差
        device=device
    )

    # <configured>. 生成IMP原型
    print("--- 生成IMP原型 ---")
    all_prototypes, all_precision_matrices, prototypes_per_class = \
        imp_generator.compute_imp_prototypes(
            support_features,
            support_labels,
            n_ways,
            verbose=True
        )

    # <configured>. 计算损失
    print("\n--- 计算IMP损失 ---")
    imp_loss_fn = SupervisedIMPLoss(distance_metric='E')
    loss, probs, distances = imp_loss_fn(
        query_features,
        all_prototypes,
        all_precision_matrices,
        query_labels,
        prototypes_per_class
    )

    predictions = torch.argmax(probs, dim=1)
    accuracy = (predictions == query_labels).float().mean()

    print(f"损失值: {loss.item():.4f}")
    print(f"预测准确率: {accuracy.item():.4f}")

    # <configured>. 不同α值的影响
    print("\n--- α参数影响（控制原型数量）---")
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

    for alpha in alphas:
        imp_gen = SupervisedIMPPrototypeGenerator(alpha=alpha, device=device)
        _, _, counts = imp_gen.compute_imp_prototypes(
            support_features, support_labels, n_ways, verbose=False
        )
        total = sum(counts)
        avg = total / len(counts)
        print(f"  α={alpha:4.1f}: 总原型={total:2d}, 平均={avg:.1f}个/类")

    print("\n提示: α越大 → λ越大 → 越容易创建新原型")


def compare_with_standard_prototypes():
    """
    对比标准原型网络（单原型）vs IMP（多原型）
    """
    print("\n\n=== 单原型 vs 多原型对比 ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_ways = _cfg_require('loss_IMP.py.compare_with_standard_prototypes.n_ways')
    k_shot = _cfg_require('loss_IMP.py.compare_with_standard_prototypes.k_shot')
    feature_dim = _cfg_require('loss_IMP.py.compare_with_standard_prototypes.feature_dim')

    # 构造多模态数据：每类有两个子簇
    support_features = []
    support_labels = []

    for class_id in range(n_ways):
        # 子簇<configured>
        cluster1 = torch.randn(k_shot // 2, feature_dim, device=device) + class_id * 5
        # 子簇<configured>（距离较远）
        cluster2 = torch.randn(k_shot - k_shot // 2, feature_dim, device=device) + class_id * 5 + 3

        support_features.append(cluster1)
        support_features.append(cluster2)
        support_labels.extend([class_id] * k_shot)

    support_features = torch.cat(support_features, dim=0)
    support_labels = torch.tensor(support_labels, device=device)

    print("数据特点: 每类包含2个相距较远的子簇（多模态）\n")

    # 标准原型（单原型）
    print("【标准原型网络】")
    unique_labels = torch.unique(support_labels)
    standard_prototypes = []
    for label in unique_labels:
        class_mean = support_features[support_labels == label].mean(dim=0)
        standard_prototypes.append(class_mean)
        print(f"  类别 {label.item()}: 1 个原型（类均值）")

    # IMP原型（多原型）
    print("\n【IMP多原型】")
    imp_gen = SupervisedIMPPrototypeGenerator(alpha=_cfg_require('loss_IMP.py.compare_with_standard_prototypes.alpha'), device=device)
    _, _, counts = imp_gen.compute_imp_prototypes(
        support_features, support_labels, n_ways, verbose=True
    )

    print(f"\n结论: IMP自动识别出多模态结构，每类生成 {counts[0]:.0f} 个原型")


def theory_summary():
    """
    理论总结
    """
    print("\n\n" + "="*60)
    print("全监督IMP理论总结".center(60))
    print("="*60)

    print("\n核心思想：\n  通过Dirichlet Process自适应地为每个类别创建多个原型（cluster），\n  从而更好地拟合复杂的、多模态的类内分布。\n\n关键机制：\n\n  1. λ阈值（公式5）：\n     λ = 2σ log(α / (1 + ρ/σ)^(d/2))\n     - 控制创建新cluster的距离阈值\n     - α↑ → λ↑ → 更多cluster\n     \n  2. Cluster创建规则：\n     if min_distance(x_i, {μ_c : l_c = y_i}) > λ:\n         创建新cluster\n         \n  3. 软分配（高斯密度）：\n     z_{i,c} = N(x_i; μ_c, σ_c) / Σ_{c'} N(x_i; μ_{c'}, σ_{c'})\n     - 只对同类别cluster归一化\n     \n  4. 分类决策（公式6）：\n     p(y=n|x) ∝ exp(-d(x, μ_{c*_n}))\n     其中 c*_n = argmin_{c: l_c=n} d(x, μ_c)\n     \n  5. Masked Loss（公式7）：\n     只对每类最近的cluster计算损失\n     - 避免过度惩罚多模态\n\n论文实验结果：\n  - Omniglot字母识别：+25% 准确率 (vs 单原型)\n  - 自适应容量：简单类1个原型，复杂类多个原型\n  - 泛化能力：在超类到子类迁移中表现更好\n\n适用场景：\n  ✓ 类内分布复杂、多模态\n  ✓ 不同类别复杂度差异大\n  ✓ Few-shot学习（样本少，需要强先验）\n  ✓ 需要可解释的原型表示\n")
    print("="*60)


if __name__ == "__main__":
    usage_example()
    compare_with_standard_prototypes()
    theory_summary()
