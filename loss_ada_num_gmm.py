from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiPrototypeGPNLoss(nn.Module):
    '\n    多原型GPN损失 - 支持GMM混合高斯模式\n    \n    新增功能：\n    <configured>. 支持GMM模式：每个类用多个高斯分量的混合模型表示\n    <configured>. 原有的单原型和多原型独立模式仍然保留\n    '
    def __init__(self, prototypes_per_class=None, use_multi=None, use_Mdistance=None,
                 use_gmm=None, gmm_components=None,
                 merge_threshold=None, merge_start_epoch=None, merge_interval=None):
        """
        Args:
            prototypes_per_class: 每个类的最大原型数
            use_multi: 是否使用多原型独立模式
            use_Mdistance: 是否使用马氏距离
            use_gmm: 是否使用GMM混合高斯模式
            gmm_components: GMM中每个类的高斯分量数
            merge_threshold: 原型合并阈值
            merge_start_epoch: 开始合并的epoch
            merge_interval: 合并间隔
        """
        prototypes_per_class = _cfg_resolve('loss_ada_num_gmm.py.MultiPrototypeGPNLoss.__init__.prototypes_per_class', prototypes_per_class)
        use_multi = _cfg_resolve('loss_ada_num_gmm.py.MultiPrototypeGPNLoss.__init__.use_multi', use_multi)
        use_Mdistance = _cfg_resolve('loss_ada_num_gmm.py.MultiPrototypeGPNLoss.__init__.use_Mdistance', use_Mdistance)
        use_gmm = _cfg_resolve('loss_ada_num_gmm.py.MultiPrototypeGPNLoss.__init__.use_gmm', use_gmm)
        gmm_components = _cfg_resolve('loss_ada_num_gmm.py.MultiPrototypeGPNLoss.__init__.gmm_components', gmm_components)
        merge_threshold = _cfg_resolve('loss_ada_num_gmm.py.MultiPrototypeGPNLoss.__init__.merge_threshold', merge_threshold)
        merge_start_epoch = _cfg_resolve('loss_ada_num_gmm.py.MultiPrototypeGPNLoss.__init__.merge_start_epoch', merge_start_epoch)
        merge_interval = _cfg_resolve('loss_ada_num_gmm.py.MultiPrototypeGPNLoss.__init__.merge_interval', merge_interval)
        super().__init__()
        self.prototypes_per_class = prototypes_per_class
        self.use_multi = use_multi
        self.use_Mdistance = use_Mdistance
        self.use_gmm = use_gmm
        self.gmm_components = gmm_components

        # 选择距离度量
        if use_gmm:
            self.distance_metric = GMMDistance(n_components=gmm_components)
        elif use_Mdistance:
            self.distance_metric = MahalanobisDistance()
        else:
            self.distance_metric = EuclideanDistance()

        # 原型合并参数（仅用于非GMM模式）
        self.merge_threshold = merge_threshold
        self.merge_start_epoch = merge_start_epoch
        self.merge_interval = merge_interval

        # 当前epoch
        self.current_epoch = 0

        # 统计信息
        self.global_merge_count = 0
        self.task_count = 0

    def _create_task_prototype_mask(self, n_ways, k_shot, device):
        """为单个任务创建原型掩码"""
        return torch.ones(n_ways, k_shot, dtype=torch.bool, device=device)

    def _merge_prototypes_in_task_optimized(self, prototypes, precision_matrices,
                                      prototype_mask, n_ways, k_shot):
        """
        使用马氏距离和图连通分量方法一次性合并所有相似原型
        （仅用于非GMM模式）
        """
        feature_dim = prototypes.shape[1]
        prototypes_reshaped = prototypes.view(n_ways, k_shot, feature_dim)
        precision_reshaped = precision_matrices.view(n_ways, k_shot, feature_dim, feature_dim)

        total_merge_count = 0

        for class_idx in range(n_ways):
            # 获取当前类的激活原型索引
            active_indices = torch.where(prototype_mask[class_idx])[0]
            n_active = len(active_indices)

            if n_active <= 1:
                continue

            # 提取激活的原型和精度矩阵
            active_prototypes = prototypes_reshaped[class_idx, active_indices]
            active_precisions = precision_reshaped[class_idx, active_indices]

            # 构建连通图
            adj_matrix = self._build_connectivity_graph(
                active_prototypes, active_precisions, self.merge_threshold
            )

            # 找到所有连通分量
            components = self._find_connected_components(adj_matrix)

            # 合并每个连通分量
            for component in components:
                if len(component) > 1:
                    global_indices = [active_indices[i].item() for i in component]
                    self._merge_component(
                        prototypes_reshaped, precision_reshaped, prototype_mask,
                        class_idx, global_indices
                    )
                    total_merge_count += (len(component) - 1)

        updated_prototypes = prototypes_reshaped.view(-1, feature_dim)
        updated_precision = precision_reshaped.view(-1, feature_dim, feature_dim)

        return updated_prototypes, updated_precision, prototype_mask, total_merge_count

    def _build_connectivity_graph(self, prototypes, precisions, threshold):
        """构建连通图"""
        n_active = len(prototypes)
        adj_matrix = torch.zeros(n_active, n_active, dtype=torch.bool, device=prototypes.device)

        for i in range(n_active):
            for j in range(i + 1, n_active):
                if self.use_Mdistance:
                    dist = self._compute_pairwise_mahalanobis(
                        prototypes[i], prototypes[j], precisions[i], precisions[j]
                    )
                else:
                    dist = torch.norm(prototypes[i] - prototypes[j])
                if dist <= threshold:
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True

        return adj_matrix

    def _compute_pairwise_mahalanobis(self, proto_i, proto_j, precision_i, precision_j):
        """计算两个原型之间的马氏距离"""
        avg_precision = (precision_i + precision_j) / 2
        diff = proto_i - proto_j
        dist_squared = torch.dot(diff, torch.mv(avg_precision, diff))
        return torch.sqrt(torch.clamp(dist_squared, min=1e-10))

    def _find_connected_components(self, adj_matrix):
        """使用BFS找到图的连通分量"""
        n = adj_matrix.shape[0]
        visited = [False] * n
        components = []

        for i in range(n):
            if not visited[i]:
                component = []
                stack = [i]
                visited[i] = True

                while stack:
                    node = stack.pop()
                    component.append(node)

                    neighbors = torch.where(adj_matrix[node])[0]
                    for neighbor in neighbors:
                        neighbor_idx = neighbor.item()
                        if not visited[neighbor_idx]:
                            visited[neighbor_idx] = True
                            stack.append(neighbor_idx)

                components.append(component)

        return components

    def _merge_component(self, prototypes_reshaped, precision_reshaped,
                        prototype_mask, class_idx, global_indices):
        """合并一个连通分量中的所有原型"""
        if len(global_indices) < 2:
            return

        active_prototypes = prototypes_reshaped[class_idx, global_indices]
        active_precisions = precision_reshaped[class_idx, global_indices]

        # 使用加权平均合并
        merged_prototype = active_prototypes.mean(dim=0)
        merged_precision = active_precisions.mean(dim=0)

        # 更新第一个原型，禁用其他原型
        first_idx = global_indices[0]
        prototypes_reshaped[class_idx, first_idx] = merged_prototype
        precision_reshaped[class_idx, first_idx] = merged_precision

        for idx in global_indices[1:]:
            prototype_mask[class_idx, idx] = False

    def forward(self, query_v, prototypes, precision_matrices, query_labels,
                gmm_params=None, epoch=None):
        """
        前向传播

        Args:
            query_v: [batch_size, feature_dim] 查询特征
            prototypes: [n_ways * k_shot, feature_dim] 或 [n_ways, feature_dim]
            precision_matrices: 精度矩阵
            query_labels: [batch_size] 查询标签
            gmm_params: GMM参数字典 (仅在use_gmm=True时使用)
                - means: [n_ways, n_components, feature_dim]
                - covariances: [n_ways, n_components, feature_dim, feature_dim]
                - weights: [n_ways, n_components]
            epoch: 当前epoch

        Returns:
            dict: 包含loss, probabilities, distances等信息
        """
        if epoch is not None:
            self.current_epoch = epoch

        batch_size = query_v.shape[0]
        n_ways = len(torch.unique(query_labels))

        # GMM模式
        if self.use_gmm:
            if gmm_params is None:
                raise ValueError("GMM mode requires gmm_params")

            # 使用GMM距离度量
            distances = self.distance_metric(query_v, gmm_params)
            logits = -distances  # 负对数似然作为logits

            probabilities = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, query_labels)

            output = {
                'loss': loss,
                'probabilities': probabilities,
                'distances': distances,
                'gmm_params': gmm_params,
                'n_ways': n_ways,
            }

            return output

        # 原有的多原型或单原型模式
        k_shot = prototypes.shape[0] // n_ways
        prototype_mask = self._create_task_prototype_mask(n_ways, k_shot, prototypes.device)

        # 判断是否需要合并（仅用于多原型模式）
        should_merge = (
            self.use_multi and
            self.current_epoch >= self.merge_start_epoch
        )

        merge_info = None
        if should_merge:
            prototypes, precision_matrices, prototype_mask, merge_count = \
                self._merge_prototypes_in_task_optimized(
                    prototypes, precision_matrices, prototype_mask, n_ways, k_shot
                )

            active_counts = prototype_mask.sum(dim=1).tolist()
            merge_info = {
                'merge_count': merge_count,
                'active_prototypes': active_counts,
                'total_active': sum(active_counts)
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

        output = {
            'loss': loss,
            'probabilities': probabilities,
            'distances': min_distances,
            'prototypes': prototypes,
            'precision_matrices': precision_matrices,
            'prototype_mask': prototype_mask,
            'n_ways': n_ways,
            'k_shot': k_shot
        }

        if closest_prototype_idx is not None:
            output['closest_prototype_idx'] = closest_prototype_idx

        if merge_info is not None:
            output['merge_info'] = merge_info

        return output

    def get_global_statistics(self):
        """获取全局统计信息"""
        return {
            'total_merges': self.global_merge_count,
            'total_tasks': self.task_count,
            'avg_merges_per_task': self.global_merge_count / max(self.task_count, 1),
            'current_epoch': self.current_epoch
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.global_merge_count = 0
        self.task_count = 0
        self.current_epoch = 0


class GMMDistance(nn.Module):
    '\n    GMM (Gaussian Mixture Model) 距离度量\n    \n    核心思想：\n    <configured>. 每个类用多个高斯分量的混合模型表示\n    <configured>. 计算查询样本到每个类GMM的负对数似然作为距离\n    <configured>. GMM参数从支持集样本估计得到\n    \n    优势：\n    - 更灵活的类表示能力\n    - 可以建模复杂的类内分布\n    - 理论基础扎实（最大似然估计）\n    '
    def __init__(self, n_components=None):
        """
        Args:
            n_components: 每个类的高斯分量数
        """
        n_components = _cfg_resolve('loss_ada_num_gmm.py.GMMDistance.__init__.n_components', n_components)
        super().__init__()
        self.n_components = n_components

    def forward(self, query_v, gmm_params):
        """
        计算查询样本到各类GMM的负对数似然

        Args:
            query_v: [batch_size, feature_dim] 查询特征
            gmm_params: GMM参数字典
                - means: [n_ways, n_components, feature_dim] 各分量均值
                - covariances: [n_ways, n_components, feature_dim, feature_dim] 协方差矩阵
                - weights: [n_ways, n_components] 各分量权重（已归一化）

        Returns:
            distances: [batch_size, n_ways] 负对数似然（越小表示越相似）
        """
        means = gmm_params['means']  # [n_ways, n_components, feature_dim]
        covariances = gmm_params['covariances']  # [n_ways, n_components, feature_dim, feature_dim]
        weights = gmm_params['weights']  # [n_ways, n_components]

        batch_size = query_v.shape[0]
        n_ways = means.shape[0]
        n_components = means.shape[1]
        feature_dim = means.shape[2]
        device = query_v.device

        # 存储每个类的对数似然
        log_likelihoods = torch.zeros(batch_size, n_ways, device=device)

        for class_idx in range(n_ways):
            # 当前类的GMM参数
            class_means = means[class_idx]  # [n_components, feature_dim]
            class_covs = covariances[class_idx]  # [n_components, feature_dim, feature_dim]
            class_weights = weights[class_idx]  # [n_components]

            # 计算每个分量的对数概率密度
            component_log_probs = torch.zeros(batch_size, n_components, device=device)

            for comp_idx in range(n_components):
                mean = class_means[comp_idx]  # [feature_dim]
                cov = class_covs[comp_idx]  # [feature_dim, feature_dim]

                # 计算多元高斯分布的对数概率密度

                # 添加数值稳定性：确保协方差矩阵正定
                cov = cov + torch.eye(feature_dim, device=device) * 1e-6

                # 计算差异
                diff = query_v - mean.unsqueeze(0)  # [batch_size, feature_dim]

                # 计算精度矩阵（协方差的逆）
                try:
                    precision = torch.linalg.inv(cov)
                    # 计算行列式
                    sign, logdet = torch.linalg.slogdet(cov)
                    if sign <= 0:
                        logdet = torch.tensor(0.0, device=device)
                except:
                    # 如果求逆失败，使用单位矩阵
                    precision = torch.eye(feature_dim, device=device)
                    logdet = torch.tensor(0.0, device=device)

                # 计算马氏距离的平方
                mahalanobis_sq = torch.sum(diff @ precision * diff, dim=1)  # [batch_size]

                # 计算对数概率密度
                log_prob = -0.5 * (mahalanobis_sq + logdet + feature_dim * math.log(2 * math.pi))

                component_log_probs[:, comp_idx] = log_prob

            # 使用log-sum-exp技巧计算混合模型的对数似然
            # log p(x|class) = log Σ_k w_k * p_k(x) = log Σ_k exp(log w_k + log p_k(x))
            log_weights = torch.log(class_weights + 1e-10)  # [n_components]
            weighted_log_probs = component_log_probs + log_weights.unsqueeze(0)  # [batch_size, n_components]

            # log-sum-exp
            max_log_prob = torch.max(weighted_log_probs, dim=1, keepdim=True)[0]
            log_likelihood = max_log_prob.squeeze(1) + torch.log(
                torch.sum(torch.exp(weighted_log_probs - max_log_prob), dim=1)
            )

            log_likelihoods[:, class_idx] = log_likelihood

        # 返回负对数似然作为距离（越小越好）
        distances = -log_likelihoods

        return distances


class MahalanobisDistance(nn.Module):
    """
    马氏距离度量
    """
    def __init__(self):
        super().__init__()

    def to(self, device):
        super().to(device)
        return self

    def forward(self, v, prototypes, precision_matrices=None):
        """
        计算马氏距离

        Args:
            v: [batch_size, feature_dim]
            prototypes: [n_ways, feature_dim]
            precision_matrices: [n_ways, feature_dim, feature_dim]

        Returns:
            distances: [batch_size, n_ways]
        """
        batch_size, feature_dim = v.shape
        n_ways = prototypes.shape[0]
        device = v.device

        v_expanded = v.unsqueeze(1).expand(-1, n_ways, -1)
        prototypes_expanded = prototypes.unsqueeze(0).expand(batch_size, -1, -1)

        diff = v_expanded - prototypes_expanded
        distances = torch.zeros(batch_size, n_ways, device=device)

        for i in range(n_ways):
            diff_i = diff[:, i, :]
            P_i = precision_matrices[i]

            temp = torch.mm(diff_i, P_i)
            quadratic_form = torch.sum(diff_i * temp, dim=1)
            distances[:, i] = torch.sqrt(torch.clamp(quadratic_form, min=1e-8))

        return distances


class EuclideanDistance(nn.Module):
    """欧氏距离计算模块"""

    def __init__(self):
        super().__init__()

    def forward(self, v, prototypes, precision_matrices=None):
        """
        计算欧氏距离

        Args:
            v: [batch_size, feature_dim]
            prototypes: [n_prototypes, feature_dim]
            precision_matrices: 未使用（保持接口兼容）

        Returns:
            distances: [batch_size, n_prototypes]
        """
        distances = torch.cdist(v, prototypes)
        return distances
