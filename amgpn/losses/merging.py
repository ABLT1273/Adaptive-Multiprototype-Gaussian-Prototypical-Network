from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
#在原基础上新增基线聚类方法
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ========== 新增导入 ==========
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
import numpy as np
# ==============================

class MultiPrototypeGPNLoss(nn.Module):
    "\n    多原型GPN损失 - 任务内动态原型合并（增强版，支持<configured>种合并策略）\n    \n    设计原则：\n    <configured>. 每个任务内独立合并原型\n    <configured>. 返回任务级别的prototype_mask供统计使用\n    <configured>. 不跨任务持久化状态\n    \n    支持的合并方法：\n    - 'connected': 连通分量方法（原始方法）\n    - 'pairwise': 成对合并方法\n    - 'hierarchical': 层次聚类方法\n    - 'dbscan': 基于密度的聚类方法\n    "
    def __init__(self, use_multi=None, use_Mdistance=None,
                 merge_threshold=None,
                 merge_method=None):
        """
        Args:
            use_multi: 是否使用多原型
            use_Mdistance: 是否使用马氏距离
            merge_threshold: 合并阈值
            merge_method: 合并方法 ('connected', 'pairwise', 'hierarchical', 'dbscan')
        """
        use_multi = _cfg_resolve('loss_ada_num_clean_clustering.py.MultiPrototypeGPNLoss.__init__.use_multi', use_multi)
        use_Mdistance = _cfg_resolve('loss_ada_num_clean_clustering.py.MultiPrototypeGPNLoss.__init__.use_Mdistance', use_Mdistance)
        merge_threshold = _cfg_resolve('loss_ada_num_clean_clustering.py.MultiPrototypeGPNLoss.__init__.merge_threshold', merge_threshold)
        merge_method = _cfg_resolve('loss_ada_num_clean_clustering.py.MultiPrototypeGPNLoss.__init__.merge_method', merge_method)
        super().__init__()
        if use_Mdistance:
            self.distance_metric = MahalanobisDistance()
        else:
            self.distance_metric = EuclideanDistance()

        self.use_multi = use_multi
        self.use_Mdistance = use_Mdistance

        # 原型合并参数
        self.merge_threshold = merge_threshold
        self.merge_method = merge_method  # 新增：合并方法选择

        # 特定方法的参数
        self.hierarchical_n_clusters = None      # hierarchical方法的聚类数
        self.hierarchical_linkage = 'ward'       # hierarchical的链接方法
        self.dbscan_eps = merge_threshold        # DBSCAN的邻域半径
        self.dbscan_min_samples = 2              # DBSCAN的最小样本数

        # 当前epoch（从外部设置）
        self.current_epoch = 0

        # 统计信息（全局累计）
        self.global_merge_count = 0
        self.task_count = 0

    def _create_task_prototype_mask(self, n_ways, k_shot, device):
        """
        为单个任务创建原型掩码

        Args:
            n_ways: 类别数
            k_shot: 每类样本数
            device: 设备

        Returns:
            mask: [n_ways, k_shot] 全True的掩码
        """
        return torch.ones(n_ways, k_shot, dtype=torch.bool, device=device)

    def _merge_prototypes_in_task_optimized(self, prototypes, precision_matrices,
                                      prototype_mask, n_ways, k_shot):
        """
        使用指定方法合并原型（支持多种策略）

        Args:
            prototypes: [n_ways * k_shot, feature_dim]
            precision_matrices: [n_ways * k_shot, feature_dim, feature_dim]
            prototype_mask: [n_ways, k_shot] 当前激活状态
            n_ways, k_shot: 任务参数

        Returns:
            updated_prototypes, updated_precision, updated_mask, merge_count
        """
        feature_dim = prototypes.shape[1]
        prototypes_reshaped = prototypes.view(n_ways, k_shot, feature_dim)
        precision_reshaped = precision_matrices.view(n_ways, k_shot,
                                                     feature_dim, feature_dim)

        total_merge_count = 0

        # 遍历每个类别，应用选定的合并策略
        for class_idx in range(n_ways):
            active_indices = torch.where(prototype_mask[class_idx])[0]
            n_active = len(active_indices)

            if n_active <= 1:
                continue

            # 根据merge_method选择不同的合并策略
            if self.merge_method == 'connected':
                # 原始的连通分量方法
                active_prototypes = prototypes_reshaped[class_idx, active_indices]
                active_precisions = precision_reshaped[class_idx, active_indices]

                adj_matrix = self._build_connectivity_graph(
                    active_prototypes, active_precisions, self.merge_threshold
                )
                components = self._find_connected_components(adj_matrix)

                for component in components:
                    if len(component) > 1:
                        global_indices = [active_indices[i].item() for i in component]
                        self._merge_component(
                            prototypes_reshaped, precision_reshaped,
                            prototype_mask, class_idx, global_indices
                        )
                        total_merge_count += (len(component) - 1)

            elif self.merge_method == 'pairwise':
                # 成对合并方法
                count = self._pairwise_merging_in_class(
                    prototypes_reshaped, precision_reshaped, prototype_mask,
                    class_idx, active_indices, self.merge_threshold
                )
                total_merge_count += count

            elif self.merge_method == 'hierarchical':
                # 层次聚类方法
                count = self._hierarchical_clustering_in_class(
                    prototypes_reshaped, precision_reshaped, prototype_mask,
                    class_idx, active_indices
                )
                total_merge_count += count

            elif self.merge_method == 'dbscan':
                # DBSCAN聚类方法
                count = self._dbscan_clustering_in_class(
                    prototypes_reshaped, precision_reshaped, prototype_mask,
                    class_idx, active_indices
                )
                total_merge_count += count

            else:
                raise ValueError(f"Unknown merge method: {self.merge_method}")

        updated_prototypes = prototypes_reshaped.view(-1, feature_dim)
        updated_precision = precision_reshaped.view(-1, feature_dim, feature_dim)

        return updated_prototypes, updated_precision, prototype_mask, total_merge_count

    def _build_connectivity_graph(self, prototypes, precisions, threshold):
        """
        构建连通图：节点i和j相连当且仅当它们的马氏距离小于阈值

        Args:
            prototypes: [n_active, feature_dim] 激活原型
            precisions: [n_active, feature_dim, feature_dim] 对应的精度矩阵
            threshold: 合并阈值

        Returns:
            adj_matrix: [n_active, n_active] 邻接矩阵
        """
        n_active = len(prototypes)
        adj_matrix = torch.zeros(n_active, n_active, dtype=torch.bool, device=prototypes.device)

        # 预计算所有原型对之间的马氏距离
        for i in range(n_active):
            for j in range(i + 1, n_active):
                if self.use_Mdistance:
                    # 使用马氏距离
                    dist = self._compute_pairwise_mahalanobis(
                        prototypes[i], prototypes[j], precisions[i], precisions[j]
                    )
                else:
                    # 使用欧氏距离
                    dist = torch.norm(prototypes[i] - prototypes[j])
                if dist <= threshold:
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True

        return adj_matrix

    def _compute_pairwise_mahalanobis(self, proto_i, proto_j, precision_i, precision_j):
        """
        计算两个原型之间的马氏距离

        Args:
            proto_i, proto_j: [feature_dim] 原型向量
            precision_i, precision_j: [feature_dim, feature_dim] 精度矩阵

        Returns:
            distance: 标量马氏距离
        """
        # 使用两个精度矩阵的平均
        avg_precision = (precision_i + precision_j) / 2

        # 计算差异向量
        diff = proto_i - proto_j

        dist_squared = torch.dot(diff, torch.mv(avg_precision, diff))

        # 确保数值稳定性
        return torch.sqrt(torch.clamp(dist_squared, min=1e-10))

    def _find_connected_components(self, adj_matrix):
        """
        使用BFS找到图的连通分量

        Args:
            adj_matrix: [n, n] 邻接矩阵

        Returns:
            components: 连通分量列表，每个分量是节点索引列表
        """
        n = adj_matrix.shape[0]
        visited = [False] * n
        components = []

        for i in range(n):
            if not visited[i]:
                # 新的连通分量
                component = []
                stack = [i]
                visited[i] = True

                # BFS遍历
                while stack:
                    node = stack.pop()
                    component.append(node)

                    # 添加所有未访问的邻居
                    neighbors = torch.where(adj_matrix[node])[0]
                    for neighbor in neighbors:
                        neighbor_idx = neighbor.item()
                        if not visited[neighbor_idx]:
                            visited[neighbor_idx] = True
                            stack.append(neighbor_idx)

                components.append(component)

        return components

    def _merge_component(self, prototypes, precisions, mask, class_idx, indices):
        """
        合并一个连通分量内的所有原型

        Args:
            prototypes: [n_ways, k_shot, feature_dim] 原型张量
            precisions: [n_ways, k_shot, feature_dim, feature_dim] 精度矩阵张量
            mask: [n_ways, k_shot] 掩码张量
            class_idx: 当前类别索引
            indices: 要合并的原型索引列表
        """
        if len(indices) <= 1:
            return

        # 计算合并后的原型和精度矩阵
        component_prototypes = prototypes[class_idx, indices]
        component_precisions = precisions[class_idx, indices]

        merged_prototype = torch.mean(component_prototypes, dim=0)
        merged_precision = torch.mean(component_precisions, dim=0)

        # 将合并结果保存到第一个位置
        target_idx = indices[0]
        prototypes[class_idx, target_idx] = merged_prototype
        precisions[class_idx, target_idx] = merged_precision
        mask[class_idx, target_idx] = True  # 确保目标位置激活

        # 将其他位置标记为非激活
        for idx in indices[1:]:
            mask[class_idx, idx] = False

    # ==================== 新增：三种额外的合并方法 ====================

    def _pairwise_merging_in_class(self, prototypes, precisions, mask,
                                    class_idx, active_indices, threshold):
        """
        在单个类内使用成对合并策略
        迭代地合并距离最近的两个原型

        Args:
            prototypes: [n_ways, k_shot, feature_dim]
            precisions: [n_ways, k_shot, feature_dim, feature_dim]
            mask: [n_ways, k_shot]
            class_idx: 当前类索引
            active_indices: 激活的原型索引
            threshold: 合并阈值

        Returns:
            merge_count: 本次合并的原型数量
        """
        feature_dim = prototypes.shape[2]
        n_active = len(active_indices)
        merge_count = 0

        if n_active <= 1:
            return merge_count

        # 提取当前激活的原型
        current_prototypes = prototypes[class_idx, active_indices].clone()
        current_precisions = precisions[class_idx, active_indices].clone()
        local_mask = torch.ones(n_active, dtype=torch.bool, device=mask.device)

        while True:
            # 计算当前激活原型的成对距离
            active_local = torch.where(local_mask)[0]
            if len(active_local) <= 1:
                break

            # 构建距离矩阵
            n_curr = len(active_local)
            dist_matrix = torch.full((n_curr, n_curr), float('inf'),
                                    device=prototypes.device)

            for i in range(n_curr):
                for j in range(i + 1, n_curr):
                    idx_i = active_local[i]
                    idx_j = active_local[j]

                    if self.use_Mdistance:
                        dist = self._compute_pairwise_mahalanobis(
                            current_prototypes[idx_i],
                            current_prototypes[idx_j],
                            current_precisions[idx_i],
                            current_precisions[idx_j]
                        )
                    else:
                        dist = torch.norm(current_prototypes[idx_i] -
                                        current_prototypes[idx_j])

                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist

            # 找到最小距离
            min_dist = dist_matrix.min()

            if min_dist > threshold:
                break

            # 找到最小距离的索引
            min_idx = torch.argmin(dist_matrix)
            i_local = min_idx // n_curr
            j_local = min_idx % n_curr

            idx_i = active_local[i_local]
            idx_j = active_local[j_local]

            # 合并两个原型
            merged_proto = (current_prototypes[idx_i] + current_prototypes[idx_j]) / 2
            merged_prec = (current_precisions[idx_i] + current_precisions[idx_j]) / 2

            # 更新到第一个位置
            current_prototypes[idx_i] = merged_proto
            current_precisions[idx_i] = merged_prec
            local_mask[idx_j] = False

            merge_count += 1

        # 将结果写回原始张量
        for local_idx, global_idx in enumerate(active_indices):
            if local_mask[local_idx]:
                prototypes[class_idx, global_idx] = current_prototypes[local_idx]
                precisions[class_idx, global_idx] = current_precisions[local_idx]
                mask[class_idx, global_idx] = True
            else:
                mask[class_idx, global_idx] = False

        return merge_count

    def _hierarchical_clustering_in_class(self, prototypes, precisions, mask,
                                         class_idx, active_indices):
        """
        在单个类内使用层次聚类

        Args:
            prototypes: [n_ways, k_shot, feature_dim]
            precisions: [n_ways, k_shot, feature_dim, feature_dim]
            mask: [n_ways, k_shot]
            class_idx: 当前类索引
            active_indices: 激活的原型索引

        Returns:
            merge_count: 本次合并的原型数量
        """
        n_active = len(active_indices)

        if n_active <= 1:
            return 0

        # 提取激活的原型
        active_protos = prototypes[class_idx, active_indices].detach().cpu().numpy()

        # 执行层次聚类
        Z = linkage(active_protos, method=self.hierarchical_linkage)

        # 确定聚类标签
        if self.hierarchical_n_clusters is not None:
            labels = fcluster(Z, self.hierarchical_n_clusters, criterion='maxclust')
        else:
            labels = fcluster(Z, self.merge_threshold, criterion='distance')

        # 按照聚类结果合并
        unique_labels = np.unique(labels)
        merge_count = 0

        # 先将当前类的所有原型标记为非激活
        mask[class_idx, :] = False

        for label in unique_labels:
            cluster_local_indices = np.where(labels == label)[0]
            cluster_global_indices = [active_indices[i].item()
                                     for i in cluster_local_indices]

            if len(cluster_global_indices) > 1:
                # 合并这个聚类
                self._merge_component(prototypes, precisions, mask,
                                     class_idx, cluster_global_indices)
                merge_count += (len(cluster_global_indices) - 1)
            else:
                # 单个原型，保持激活
                mask[class_idx, cluster_global_indices[0]] = True

        return merge_count

    def _dbscan_clustering_in_class(self, prototypes, precisions, mask,
                                    class_idx, active_indices):
        """
        在单个类内使用DBSCAN聚类

        Args:
            prototypes: [n_ways, k_shot, feature_dim]
            precisions: [n_ways, k_shot, feature_dim, feature_dim]
            mask: [n_ways, k_shot]
            class_idx: 当前类索引
            active_indices: 激活的原型索引

        Returns:
            merge_count: 本次合并的原型数量
        """
        n_active = len(active_indices)

        if n_active <= 1:
            return 0

        # 提取激活的原型
        active_protos = prototypes[class_idx, active_indices].detach().cpu().numpy()

        # 执行DBSCAN聚类
        metric = 'euclidean' if not self.use_Mdistance else 'euclidean'
        db = DBSCAN(eps=self.dbscan_eps,
                   min_samples=self.dbscan_min_samples,
                   metric=metric)
        labels = db.fit_predict(active_protos)

        # 按照聚类结果合并
        unique_labels = np.unique(labels)
        merge_count = 0

        # 先将当前类的所有原型标记为非激活
        mask[class_idx, :] = False

        for label in unique_labels:
            cluster_local_indices = np.where(labels == label)[0]
            cluster_global_indices = [active_indices[i].item()
                                     for i in cluster_local_indices]

            if label == -1:
                # 噪声点：每个单独保留
                for idx in cluster_global_indices:
                    mask[class_idx, idx] = True
            else:
                # 正常聚类
                if len(cluster_global_indices) > 1:
                    self._merge_component(prototypes, precisions, mask,
                                        class_idx, cluster_global_indices)
                    merge_count += (len(cluster_global_indices) - 1)
                else:
                    mask[class_idx, cluster_global_indices[0]] = True

        return merge_count

    # ==================== 原有的forward方法保持不变 ====================

    def forward(self, query_v, prototypes, precision_matrices, query_labels, epoch=None):
        """
        前向传播（任务内原型合并）

        Args:
            query_v: [batch_size, feature_dim]
            prototypes: [n_ways * k_shot, feature_dim]
            precision_matrices: [n_ways * k_shot, feature_dim, feature_dim]
            query_labels: [batch_size]
            epoch: 当前epoch（可选）

        Returns:
            dict: 包含loss, probabilities, distances, prototype_mask等信息
        """
        if epoch is not None:
            self.current_epoch = epoch

        batch_size = query_v.shape[0]
        n_ways = len(torch.unique(query_labels))
        k_shot = prototypes.shape[0] // n_ways

        # 为这个任务创建原型掩码
        prototype_mask = self._create_task_prototype_mask(n_ways, k_shot, prototypes.device)

        merge_info = None

        prototypes, precision_matrices, prototype_mask, merge_count = \
            self._merge_prototypes_in_task_optimized(
                prototypes, precision_matrices, prototype_mask, n_ways, k_shot
            )

        active_counts = prototype_mask.sum(dim=1).tolist()
        merge_info = {
            'merge_count': merge_count,
            'active_prototypes': active_counts,
            'total_active': sum(active_counts),
            'merge_method': self.merge_method  # 新增：记录使用的方法
        }

        self.global_merge_count += merge_count

        self.task_count += 1

        # 计算距离和loss
        if self.use_multi:
            all_distances = self.distance_metric(query_v, prototypes, precision_matrices)
            distances_reshaped = all_distances.view(batch_size, n_ways, k_shot)

            # 应用掩码
            mask_expanded = prototype_mask.unsqueeze(0).expand(batch_size, -1, -1)
            distances_reshaped = torch.where(
                mask_expanded,
                distances_reshaped,
                torch.tensor(float('inf'), device=distances_reshaped.device)
            )

            min_distances, closest_prototype_idx = torch.min(distances_reshaped, dim=2)
            logits = -min_distances

        else:
            distances = self.distance_metric(query_v, prototypes, precision_matrices)
            logits = -distances
            min_distances = distances
            closest_prototype_idx = None

        probabilities = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, query_labels)

        # ========== 关键：返回prototype_mask供统计使用 ==========
        output = {
            'loss': loss,
            'probabilities': probabilities,
            'distances': min_distances,
            'prototypes': prototypes,
            'precision_matrices': precision_matrices,
            'prototype_mask': prototype_mask,  # ← 返回任务级别的mask
            'n_ways': n_ways,
            'k_shot': k_shot
        }

        if closest_prototype_idx is not None:
            output['closest_prototype_idx'] = closest_prototype_idx

        if merge_info is not None:
            output['merge_info'] = merge_info

        return output


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
