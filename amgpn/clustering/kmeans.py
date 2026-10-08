from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class KMeansPlusPlus:
    """
    完整的K-Means++聚类实现，包含初始化和迭代优化

    Features:
    - 高效的向量化K-Means++初始化
    - 完整的K-Means迭代算法
    - 支持GPU加速
    - 支持不同的距离度量（L2, 余弦相似度）
    """

    def __init__(self, k: int, max_iters: int = None, tol: float = None,
                 distance_metric: str = 'l2', device: str = None):
        """
        Args:
            k: 聚类数量
            max_iters: 最大迭代次数
            tol: 收敛阈值
            distance_metric: 距离度量 ('l2' 或 'cosine')
            device: 计算设备 ('cuda' 或 'cpu')
        """
        max_iters = _cfg_resolve('kmeans_plus_plus_complete.py.KMeansPlusPlus.__init__.max_iters', max_iters)
        tol = _cfg_resolve('kmeans_plus_plus_complete.py.KMeansPlusPlus.__init__.tol', tol)
        device = _cfg_resolve('kmeans_plus_plus_complete.py.KMeansPlusPlus.__init__.device', device)
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.distance_metric = distance_metric
        self.device = device

        self.centroids = None
        self.labels = None
        self.inertia_history = []

    def _compute_distances(self, data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        计算样本到聚类中心的距离（向量化实现）

        Args:
            data: [n_samples, d_features]
            centroids: [k, d_features]

        Returns:
            distances: [n_samples, k]
        """
        if self.distance_metric == 'l2':
            # 使用数学恒等式优化L2距离计算
            data_sq = (data ** 2).sum(dim=1, keepdim=True)  # [n, <configured>]
            centroids_sq = (centroids ** 2).sum(dim=1)  # [k]
            dot_product = torch.mm(data, centroids.t())  # [n, k]
            distances = torch.sqrt(torch.clamp(
                data_sq + centroids_sq - 2 * dot_product,
                min=1e-8
            ))
        elif self.distance_metric == 'cosine':
            # 余弦距离
            data_norm = F.normalize(data, p=2, dim=1)  # [n, d]
            centroids_norm = F.normalize(centroids, p=2, dim=1)  # [k, d]
            # 余弦相似度 = x·c / (||x|| * ||c||)
            similarity = torch.mm(data_norm, centroids_norm.t())  # [n, k]
            distances = 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def _kmeans_plusplus_init(self, data: torch.Tensor) -> torch.Tensor:
        '\n        K-Means++ 初始化：智能选择初始聚类中心\n        \n        算法流程：\n        <configured>. 随机选择第一个中心\n        <configured>. 对每个新中心：\n           - 计算每个样本到最近中心的距离D(x)\n           - 选择距离较远的点作为新中心（概率正比于D(x)^<configured>）\n        \n        Args:\n            data: [n_samples, d_features]\n            \n        Returns:\n            centroids: [k, d_features]\n        '
        n_samples = data.shape[0]

        if n_samples < self.k:
            raise ValueError(f"Number of samples ({n_samples}) < k ({self.k})")

        centroids = []

        # 步骤<configured>：随机选择第一个中心
        first_idx = torch.randint(0, n_samples, (1,), device=data.device).item()
        centroids.append(data[first_idx])

        # 步骤<configured>-k：依次选择其余k-<configured>个中心
        for _ in range(1, self.k):
            # 计算所有样本到已有中心的最小距离
            centroids_tensor = torch.stack(centroids)  # [num_centroids, d]
            distances = self._compute_distances(data, centroids_tensor)  # [n, num_centroids]
            min_distances = distances.min(dim=1)[0]  # [n]

            # 距离的平方用于概率计算（K-Means++的关键）
            distances_sq = min_distances ** 2
            probabilities = distances_sq / distances_sq.sum()

            # 根据概率选择下一个中心
            next_idx = torch.multinomial(probabilities, 1).item()
            centroids.append(data[next_idx])

        return torch.stack(centroids)  # [k, d]

    def _assign_clusters(self, data: torch.Tensor) -> torch.Tensor:
        """
        将每个样本分配到最近的聚类中心

        Args:
            data: [n_samples, d_features]

        Returns:
            labels: [n_samples] - 每个样本的聚类标签
        """
        distances = self._compute_distances(data, self.centroids)  # [n, k]
        labels = distances.argmin(dim=1)
        return labels

    def _update_centroids(self, data: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        根据聚类分配更新聚类中心

        Args:
            data: [n_samples, d_features]
            labels: [n_samples] - 每个样本的聚类标签

        Returns:
            new_centroids: [k, d_features]
            inertia: 类内距离和（用于判断收敛）
        """
        new_centroids = []
        inertia = 0.0

        for i in range(self.k):
            # 获取属于第i个聚类的样本
            mask = labels == i
            cluster_samples = data[mask]

            if cluster_samples.shape[0] == 0:
                # 如果某个聚类为空，随机选择一个样本作为中心
                new_centroids.append(self.centroids[i])
            else:
                # 计算聚类中心（平均值）
                new_center = cluster_samples.mean(dim=0)
                new_centroids.append(new_center)

                # 累加类内距离
                distances = torch.norm(cluster_samples - new_center, dim=1)
                inertia += (distances ** 2).sum().item()

        return torch.stack(new_centroids), inertia

    def fit(self, data: torch.Tensor) -> 'KMeansPlusPlus':
        """
        使用K-Means++算法进行聚类

        Args:
            data: [n_samples, d_features]

        Returns:
            self
        """
        # 确保数据在正确的设备上
        data = data.to(self.device)

        # K-Means++ 初始化
        self.centroids = self._kmeans_plusplus_init(data)

        # 迭代优化
        self.inertia_history = []

        for iteration in range(self.max_iters):
            # 分配聚类
            self.labels = self._assign_clusters(data)

            # 更新中心
            new_centroids, inertia = self._update_centroids(data, self.labels)
            self.inertia_history.append(inertia)

            # 计算中心变化量
            centroid_shift = torch.norm(new_centroids - self.centroids).item()


            # 检查收敛条件
            if centroid_shift < self.tol:
                break

            self.centroids = new_centroids

        return self

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        预测新数据的聚类标签

        Args:
            data: [n_samples, d_features]

        Returns:
            labels: [n_samples]
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        data = data.to(self.device)
        return self._assign_clusters(data)

    def fit_predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        拟合并预测

        Args:
            data: [n_samples, d_features]

        Returns:
            labels: [n_samples]
        """
        self.fit(data)
        return self.labels

    def get_centroids(self) -> torch.Tensor:
        """返回聚类中心"""
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet.")
        return self.centroids.cpu() if self.device == 'cuda' else self.centroids

    def get_inertia(self) -> float:
        """返回最后的inertia值"""
        if not self.inertia_history:
            raise ValueError("Model has not been fitted yet.")
        return self.inertia_history[-1]


# ============================================================================
# 优化的原始实现对比
# ============================================================================

def kmeans_plus_plus_original(data: torch.Tensor, k: int, max_iters: int = None) -> torch.Tensor:
    '\n    原始实现（存在的问题）\n    \n    问题：\n    <configured>. 嵌套循环导致O(n*k*k)复杂度 - 非常低效\n    <configured>. 没有数据检查\n    <configured>. 没有收敛判断\n    <configured>. 没有返回聚类中心，只返回初始化的中心\n    <configured>. 没有处理空聚类的情况\n    '
    max_iters = _cfg_resolve('kmeans_plus_plus_complete.py.kmeans_plus_plus_original.max_iters', max_iters)
    n_samples = data.shape[0]
    centroids = []

    # 第一个中心随机选择
    first_idx = torch.randint(0, n_samples, (1,))
    centroids.append(data[first_idx])

    for _ in range(1, k):
        # 问题：O(n*k)的嵌套循环
        distances = []
        for sample in data:
            min_dist = float('inf')
            for center in centroids:
                dist = torch.norm(sample - center)  # 每次计算一个距离
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)  # 问题：Python列表很慢

        # 问题：转换回张量多次
        distances = torch.tensor(distances)
        probabilities = distances / distances.sum()
        next_idx = torch.multinomial(probabilities, 1)
        centroids.append(data[next_idx])

    return torch.cat(centroids)


def kmeans_plus_plus_optimized(data: torch.Tensor, k: int, max_iters: int = None) -> torch.Tensor:
    '\n    优化版本（与KMeansPlusPlus类中的初始化方法一致）\n    \n    优化点：\n    <configured>. 向量化距离计算：O(n*k)\n    <configured>. 一次性返回张量\n    <configured>. 适当的device处理\n    <configured>. 直接使用torch操作避免Python循环\n    '
    max_iters = _cfg_resolve('kmeans_plus_plus_complete.py.kmeans_plus_plus_optimized.max_iters', max_iters)
    device = data.device
    n_samples = data.shape[0]

    if n_samples < k:
        raise ValueError(f"Number of samples ({n_samples}) < k ({k})")

    centroids = []

    # 第一个中心
    first_idx = torch.randint(0, n_samples, (1,), device=device).item()
    centroids.append(data[first_idx])

    for _ in range(1, k):
        # 向量化：一次计算所有距离
        centroids_stack = torch.stack(centroids)  # [num_centers, d]

        # 计算到所有中心的距离 [n, num_centers]
        data_expand = data.unsqueeze(1)  # [n, <configured>, d]
        centroids_expand = centroids_stack.unsqueeze(0)  # [<configured>, num_centers, d]
        distances = torch.norm(data_expand - centroids_expand, dim=2)  # [n, num_centers]

        # 最小距离
        min_distances = distances.min(dim=1)[0]  # [n]

        # K-Means++：按距离平方的概率选择
        distances_sq = min_distances ** 2
        probabilities = distances_sq / distances_sq.sum()
        next_idx = torch.multinomial(probabilities, 1).item()
        centroids.append(data[next_idx])

    return torch.stack(centroids)  # [k, d]


# ============================================================================
# 使用示例与性能对比
# ============================================================================

if __name__ == "__main__":
    import time

    # 创建测试数据
    n_samples = _cfg_require('kmeans_plus_plus_complete.py.module.size_or_budget')
    n_features = 64
    k = 5

    # CPU测试
    print("="*80)
    print("CPU Performance Comparison")
    print("="*80)
    data_cpu = torch.randn(n_samples, n_features)

    # 原始版本
    start = time.time()
    centroids_orig = kmeans_plus_plus_original(data_cpu, k=k)
    time_orig = time.time() - start
    print(f"Original implementation: {time_orig:.4f}s")
    print(f"  Centroids shape: {centroids_orig.shape}")

    # 优化版本
    start = time.time()
    centroids_opt = kmeans_plus_plus_optimized(data_cpu, k=k)
    time_opt = time.time() - start
    print(f"Optimized implementation: {time_opt:.4f}s")
    print(f"  Centroids shape: {centroids_opt.shape}")

    print(f"Speedup: {time_orig / time_opt:.2f}x\n")

    # 完整类版本（包含K-Means迭代）
    print("="*80)
    print("Full K-Means with Iteration")
    print("="*80)

    # CPU版本
    start = time.time()
    kmeans_cpu = KMeansPlusPlus(k=k, max_iters=_cfg_require('kmeans_plus_plus_complete.py.module.max_iters'), device=_cfg_require('kmeans_plus_plus_complete.py.module.device'))
    kmeans_cpu.fit(data_cpu)
    time_cpu = time.time() - start
    print(f"CPU K-Means: {time_cpu:.4f}s")
    print(f"Final Inertia: {kmeans_cpu.get_inertia():.4f}\n")

    # GPU版本（如果可用）
    if torch.cuda.is_available():
        data_gpu = data_cpu.cuda()
        start = time.time()
        kmeans_gpu = KMeansPlusPlus(k=k, max_iters=_cfg_require('kmeans_plus_plus_complete.py.module.max_iters__3'), device=_cfg_require('kmeans_plus_plus_complete.py.module.device__3'))
        kmeans_gpu.fit(data_gpu)
        time_gpu = time.time() - start
        print(f"GPU K-Means: {time_gpu:.4f}s")
        print(f"Final Inertia: {kmeans_gpu.get_inertia():.4f}")
        print(f"Speedup: {time_cpu / time_gpu:.2f}x\n")

    # 用例：预测新样本
    print("="*80)
    print("Prediction on New Data")
    print("="*80)
    new_data = torch.randn(100, n_features)
    labels = kmeans_cpu.predict(new_data)
    print(f"Predicted {new_data.shape[0]} samples")
    print(f"Label distribution: {torch.bincount(labels)}")
    print(f"Centroids shape: {kmeans_cpu.get_centroids().shape}")

    # 用例：针对GPN的原型初始化
    print("\n" + "="*80)
    print("Usage in GPN: Initialize Prototypes per Class")
    print("="*80)

    class_features = torch.randn(_cfg_require('kmeans_plus_plus_complete.py.module.size_or_budget__2'), 64)  # 某个类的特征
    n_prototypes = 4

    kmeans_proto = KMeansPlusPlus(k=n_prototypes, max_iters=_cfg_require('kmeans_plus_plus_complete.py.module.max_iters__2'), device=_cfg_require('kmeans_plus_plus_complete.py.module.device__2'))
    kmeans_proto.fit(class_features)
    prototypes = kmeans_proto.get_centroids()

    print(f"Generated {prototypes.shape[0]} prototypes from {class_features.shape[0]} features")
    print(f"Prototype shape: {prototypes.shape}")
    print(f"Prototype distances (should be well-separated):")
    for i in range(n_prototypes):
        for j in range(i+1, n_prototypes):
            dist = torch.norm(prototypes[i] - prototypes[j]).item()
            print(f"  Distance between proto {i} and {j}: {dist:.4f}")
