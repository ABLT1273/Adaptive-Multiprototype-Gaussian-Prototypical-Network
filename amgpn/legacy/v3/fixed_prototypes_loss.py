from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiPrototypeGPNLoss(nn.Module):
    """
    多原型GPN损失 - 支持单类多子簇的马氏距离
    """
    def __init__(self, prototypes_per_class=None, use_multi=None):
        prototypes_per_class = _cfg_resolve('loss.py.MultiPrototypeGPNLoss.__init__.prototypes_per_class', prototypes_per_class)
        use_multi = _cfg_resolve('loss.py.MultiPrototypeGPNLoss.__init__.use_multi', use_multi)
        super().__init__()
        self.prototypes_per_class = prototypes_per_class
        self.distance_metric = MahalanobisDistance()
        self.use_multi=use_multi

    def forward(self, query_v, prototypes, precision_matrices, query_labels, prototype_assignments=None):
        """
        Args:
            query_v: [batch_size, feature_dim]
            prototypes: [n_ways * prototypes_per_class, feature_dim] - 展平的多原型
            precision_matrices: [n_ways * prototypes_per_class, feature_dim, feature_dim]
            query_labels: [batch_size]
            prototype_assignments: [n_ways, prototypes_per_class] - 每个类的原型分配掩码
        """
        if self.use_multi:
            batch_size = query_v.shape[0]
            n_ways = len(torch.unique(query_labels))

            # 计算到所有原型的距离 [batch_size, n_ways * prototypes_per_class]
            all_distances = self.distance_metric(query_v, prototypes,precision_matrices)

            # 重塑为 [batch_size, n_ways, prototypes_per_class]
            distances_reshaped = all_distances.view(batch_size, n_ways, self.prototypes_per_class)

            # 对每个类取最小距离（样本到该类最近原型的距离）
            min_distances, _ = torch.min(distances_reshaped, dim=2)  # [batch_size, n_ways]

            # 转换为logits（负距离）
            logits = -min_distances

            # 计算损失
            loss = F.cross_entropy(logits, query_labels)
            probabilities = F.softmax(logits, dim=1)

            return loss, probabilities, min_distances
        else:
            distances = self.distance_metric(query_v, prototypes, precision_matrices)
            logits = -distances
            probabilities = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, query_labels)
            return loss, probabilities, distances

class MahalanobisDistance(nn.Module):
    '\n    马氏距离度量 - 从GPNLoss提取的独立模块\n    \n    核心思想：\n    <configured>. 考虑特征空间的协方差结构\n    <configured>. 使用精度矩阵（协方差矩阵的逆）\n    <configured>. 计算二次型: sqrt((x-μ)^T P (x-μ))\n    \n    优势：\n    - 理论基础扎实\n    - 考虑特征间相关性\n    - 可解释性强\n    '
    def __init__(self):
        super().__init__()

    def to(self, device):
        """重写to方法确保所有组件都移动到正确设备"""
        super().to(device)
        return self

    def forward(self, v, prototypes, precision_matrices=None):
        """
        计算马氏距离

        Args:
            v: 查询样本特征 [batch_size, feature_dim]
            prototypes: 各类原型 [n_ways, feature_dim]
            precision_matrices: 各类精度矩阵 [n_ways, feature_dim, feature_dim]
                              如果为None，则使用单位矩阵（退化为欧氏距离）

        Returns:
            distances: 马氏距离 [batch_size, n_ways]
        """
        batch_size, feature_dim = v.shape
        n_ways = prototypes.shape[0]
        device = v.device

        # 扩维以便广播计算
        v_expanded = v.unsqueeze(1).expand(-1, n_ways, -1)  # [batch_size, n_ways, feature_dim]
        prototypes_expanded = prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_ways, feature_dim]

        # 计算差异
        diff = v_expanded - prototypes_expanded  # [batch_size, n_ways, feature_dim]

        distances = torch.zeros(batch_size, n_ways, device=device)


        for i in range(n_ways):
            diff_i = diff[:, i, :]  # [batch_size, feature_dim]
            P_i = precision_matrices[i]  # [feature_dim, feature_dim]

            # 计算二次型: (x-μ)^T P (x-μ)
            temp = torch.mm(diff_i, P_i)  # [batch_size, feature_dim]
            quadratic_form = torch.sum(diff_i * temp, dim=1)  # [batch_size]
            distances[:, i] = torch.sqrt(torch.clamp(quadratic_form, min=1e-8))

        return distances

class EuclideanDistance(nn.Module):
    """欧氏距离计算模块"""

    def __init__(self):
        super().__init__()

    def forward(self,
                v: torch.Tensor,
                prototypes: torch.Tensor,
                precision_matrices: torch.Tensor = None) -> torch.Tensor:
        """
        计算欧氏距离

        Args:
            v: [batch_size, feature_dim] 查询特征
            prototypes: [n_prototypes, feature_dim] 原型
            precision_matrices: 未使用（保持接口兼容）

        Returns:
            distances: [batch_size, n_prototypes] 欧氏距离
        """
        # 直接计算欧氏距离
        distances = torch.cdist(v, prototypes)
        return distances
