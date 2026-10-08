from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _inverse_softplus_scalar(x: float) -> float:
    """
    用于初始化 raw 参数，使 softplus(raw) ≈ x
    """
    return math.log(math.exp(x) - 1.0)


class UNEMGaussianHead(nn.Module):
    """
    最小 UNEM-Gaussian transductive head。

    输入：
        query_v: [Q, F]
        support_prototypes: [n_ways * k_shot, F]
        support_precision_matrices: [n_ways * k_shot, F, F]

    输出：
        logits: [Q, n_ways]
        probabilities: [Q, n_ways]
        distances: [Q, n_ways]
        mu: [n_ways, F]
        precision_matrices: [n_ways, F, F]
        resp: [Q, n_ways]

    注意：
        这是 transductive 推理，query batch 内样本会共同参与 EM 展开。
    """
    def __init__(
        self,
        n_layers=None,
        init_lambda=None,
        init_temperature=None,
        eps=1e-6,
        use_diag_precision=None,
    ):
        n_layers = _cfg_resolve('loss_ada_num_clean_UNEM.py.UNEMGaussianHead.__init__.n_layers', n_layers)
        init_lambda = _cfg_resolve('loss_ada_num_clean_UNEM.py.UNEMGaussianHead.__init__.init_lambda', init_lambda)
        init_temperature = _cfg_resolve('loss_ada_num_clean_UNEM.py.UNEMGaussianHead.__init__.init_temperature', init_temperature)
        use_diag_precision = _cfg_resolve('loss_ada_num_clean_UNEM.py.UNEMGaussianHead.__init__.use_diag_precision', use_diag_precision)
        super().__init__()
        self.n_layers = n_layers
        self.eps = eps
        self.use_diag_precision = use_diag_precision

        self.raw_lambda = nn.Parameter(
            torch.full((n_layers,), _inverse_softplus_scalar(init_lambda))
        )

        self.raw_temperature = nn.Parameter(
            torch.full(
                (n_layers,),
                _inverse_softplus_scalar(max(init_temperature - 1.0, 1e-4))
            )
        )

    def _dist_logits(self, query_v, mu, precision_diag=None):
        '\n        Gaussian logit，默认使用单位协方差：\n            logit = -<configured> * ||x - mu||^<configured>\n\n        如果 use_diag_precision=True：\n            logit = -<configured> * (x - mu)^T P (x - mu)\n        '
        diff = query_v.unsqueeze(1) - mu.unsqueeze(0)  # [Q, C, F]

        if self.use_diag_precision and precision_diag is not None:
            dist2 = (diff.pow(2) * precision_diag.unsqueeze(0)).sum(dim=-1)
        else:
            dist2 = diff.pow(2).sum(dim=-1)

        return -0.5 * dist2

    def forward(
        self,
        query_v,
        support_prototypes,
        support_precision_matrices,
        n_ways,
        k_shot,
    ):
        device = query_v.device
        dtype = query_v.dtype
        feature_dim = query_v.shape[-1]

        # [C*K, F] -> [C, K, F]
        support = support_prototypes.reshape(n_ways, k_shot, feature_dim)

        # 初始化每类 Gaussian 均值：support 类均值
        mu = support.mean(dim=1)  # [C, F]

        # 默认不使用 GPN precision，更贴近普通 UNEM-Gaussian identity covariance
        precision_diag = None
        if self.use_diag_precision and support_precision_matrices is not None:
            P = torch.diagonal(
                support_precision_matrices,
                dim1=-2,
                dim2=-1
            )
            P = P.reshape(n_ways, k_shot, feature_dim).clamp_min(self.eps)
            precision_diag = P.mean(dim=1)  # [C, F]

        # query assignment 初始化
        logits = self._dist_logits(query_v, mu, precision_diag)
        resp = F.softmax(logits, dim=1)  # [Q, C]

        # query batch 类比例
        pi_q = torch.full(
            (n_ways,),
            1.0 / n_ways,
            device=device,
            dtype=dtype
        )

        # support hard-label contribution
        support_sum = support.sum(dim=1)  # [C, F]
        support_count = torch.full(
            (n_ways, 1),
            float(k_shot),
            device=device,
            dtype=dtype
        )

        lambdas = F.softplus(self.raw_lambda)                  # [L]
        temperatures = 1.0 + F.softplus(self.raw_temperature)  # [L]

        for layer in range(self.n_layers):
            lam = lambdas[layer]
            temp = temperatures[layer]

            # M-step：support hard labels + query soft labels 更新均值
            query_weight_sum = resp.t() @ query_v  # [C, F]
            query_mass = resp.sum(dim=0, keepdim=True).t()  # [C, <configured>]

            class_mass = support_count + query_mass
            mu = (support_sum + query_weight_sum) / class_mass.clamp_min(self.eps)

            # 更新 query batch 类比例
            pi_q = resp.mean(dim=0).clamp_min(self.eps)
            pi_q = pi_q / pi_q.sum().clamp_min(self.eps)

            # E-step：带 class-balance correction 和 temperature
            logits = self._dist_logits(query_v, mu, precision_diag)
            logits = (
                logits - lam * torch.log(pi_q.unsqueeze(0))
            ) / temp.clamp_min(self.eps)

            resp = F.softmax(logits, dim=1)

        final_logits = self._dist_logits(query_v, mu, precision_diag)
        final_logits = (
            final_logits - lambdas[-1] * torch.log(pi_q.unsqueeze(0))
        ) / temperatures[-1].clamp_min(self.eps)

        probabilities = F.softmax(final_logits, dim=1)
        distances = -final_logits

        # 为了兼容原输出结构，返回 [C, F, F] precision
        precision_matrices = torch.eye(
            feature_dim,
            device=device,
            dtype=dtype
        ).unsqueeze(0).repeat(n_ways, 1, 1)

        return final_logits, probabilities, distances, mu, precision_matrices, resp


class UNEMClassGMMHead(nn.Module):
    '\n    每类一个 GMM 的 UNEM-style transductive head。\n\n    每个类别 c 拥有 M 个 Gaussian components:\n        GMM_c = {mu[c, <configured>], ..., mu[c, M-<configured>]}\n\n    推理流程：\n        <configured>. support 初始化每类 GMM components\n        <configured>. query batch 参与 transductive EM\n        <configured>. 每层执行 soft assignment + M-step 更新\n        <configured>. 最终同一类的 M 个 component 用 logsumexp 聚合成类别 logits\n\n    输入：\n        query_v: [Q, F]\n        support_prototypes: [C * K, F]\n        support_precision_matrices: [C * K, F, F]\n\n    输出：\n        logits: [Q, C]\n        probabilities: [Q, C]\n        distances: [Q, C]\n        final_mu: [C * M, F]\n        final_precision_matrices: [C * M, F, F]\n        component_resp: [Q, C, M]\n        M\n    '
    def __init__(
        self,
        n_layers=None,
        gmm_components=None,
        init_lambda=None,
        init_temperature=None,
        eps=1e-6,
        use_diag_precision=None,
    ):
        n_layers = _cfg_resolve('loss_ada_num_clean_UNEM.py.UNEMClassGMMHead.__init__.n_layers', n_layers)
        init_lambda = _cfg_resolve('loss_ada_num_clean_UNEM.py.UNEMClassGMMHead.__init__.init_lambda', init_lambda)
        init_temperature = _cfg_resolve('loss_ada_num_clean_UNEM.py.UNEMClassGMMHead.__init__.init_temperature', init_temperature)
        use_diag_precision = _cfg_resolve('loss_ada_num_clean_UNEM.py.UNEMClassGMMHead.__init__.use_diag_precision', use_diag_precision)
        super().__init__()
        self.n_layers = n_layers
        self.gmm_components = gmm_components
        self.eps = eps
        self.use_diag_precision = use_diag_precision

        self.raw_lambda = nn.Parameter(
            torch.full((n_layers,), _inverse_softplus_scalar(init_lambda))
        )

        self.raw_temperature = nn.Parameter(
            torch.full(
                (n_layers,),
                _inverse_softplus_scalar(max(init_temperature - 1.0, 1e-4))
            )
        )

    def _init_components(self, support, support_precision_diag=None):
        """
        support: [C, K, F]
        return:
            mu: [C, M, F]
            precision_diag: [C, M, F] or None
        """
        C, K, F_dim = support.shape

        if self.gmm_components is None:
            M = K
        else:
            M = min(self.gmm_components, K)

        # 为了比“取前 M 个”更稳，这里用均匀下标初始化
        if M == K:
            indices = torch.arange(K, device=support.device)
        else:
            indices = torch.linspace(
                0, K - 1, steps=M, device=support.device
            ).long()

        mu = support[:, indices, :].clone()

        precision_diag = None
        if self.use_diag_precision and support_precision_diag is not None:
            precision_diag = support_precision_diag[:, indices, :].clone()

        return mu, precision_diag, M

    def _gaussian_component_logits(self, query_v, mu, precision_diag=None):
        """
        query_v: [Q, F]
        mu: [C, M, F]
        precision_diag: [C, M, F] or None

        return:
            component_logits: [Q, C, M]
        """
        diff = query_v[:, None, None, :] - mu[None, :, :, :]  # [Q, C, M, F]

        if self.use_diag_precision and precision_diag is not None:
            dist2 = (diff.pow(2) * precision_diag[None, :, :, :]).sum(dim=-1)
        else:
            dist2 = diff.pow(2).sum(dim=-1)

        return -0.5 * dist2

    def forward(
        self,
        query_v,
        support_prototypes,
        support_precision_matrices,
        n_ways,
        k_shot,
    ):
        device = query_v.device
        dtype = query_v.dtype
        Q, feature_dim = query_v.shape

        support = support_prototypes.reshape(n_ways, k_shot, feature_dim)

        support_precision_diag = None
        if self.use_diag_precision and support_precision_matrices is not None:
            support_precision_diag = torch.diagonal(
                support_precision_matrices,
                dim1=-2,
                dim2=-1
            ).reshape(n_ways, k_shot, feature_dim).clamp_min(self.eps)

        # 初始化每类 GMM components
        mu, precision_diag, M = self._init_components(
            support,
            support_precision_diag
        )

        # 每类内部 component mixture weight: [C, M]
        pi_cm = torch.full(
            (n_ways, M),
            1.0 / M,
            device=device,
            dtype=dtype
        )

        lambdas = F.softplus(self.raw_lambda)
        temperatures = 1.0 + F.softplus(self.raw_temperature)

        # 初始化 query responsibilities
        component_logits = self._gaussian_component_logits(
            query_v,
            mu,
            precision_diag
        ) + torch.log(pi_cm[None, :, :].clamp_min(self.eps))

        class_logits = torch.logsumexp(component_logits, dim=2)  # [Q, C]
        class_resp = F.softmax(class_logits, dim=1)              # [Q, C]

        component_resp = F.softmax(
            component_logits.reshape(Q, -1),
            dim=1
        ).reshape(Q, n_ways, M)

        for layer in range(self.n_layers):
            lam = lambdas[layer]
            temp = temperatures[layer]

            # =========================
            # <configured>. Support component assignment
            # =========================
            # support 只在本类内部给 M 个 component 做软分配
            support_diff = support[:, :, None, :] - mu[:, None, :, :]  # [C, K, M, F]

            if self.use_diag_precision and precision_diag is not None:
                support_dist2 = (
                    support_diff.pow(2) * precision_diag[:, None, :, :]
                ).sum(dim=-1)
            else:
                support_dist2 = support_diff.pow(2).sum(dim=-1)

            support_comp_logits = (
                -0.5 * support_dist2
                + torch.log(pi_cm[:, None, :].clamp_min(self.eps))
            )  # [C, K, M]

            support_resp = F.softmax(support_comp_logits, dim=2)  # [C, K, M]

            # =========================
            # <configured>. Query component assignment
            # =========================
            component_logits = self._gaussian_component_logits(
                query_v,
                mu,
                precision_diag
            ) + torch.log(pi_cm[None, :, :].clamp_min(self.eps))

            class_logits = torch.logsumexp(component_logits, dim=2)  # [Q, C]

            # Correct class balance.
            class_prior = class_resp.mean(dim=0).clamp_min(self.eps)
            class_prior = class_prior / class_prior.sum().clamp_min(self.eps)

            balanced_class_logits = (
                class_logits - lam * torch.log(class_prior[None, :])
            ) / temp.clamp_min(self.eps)

            class_resp = F.softmax(balanced_class_logits, dim=1)  # [Q, C]

            # component posterior = p(c|x) * p(m|x,c)
            comp_given_class = F.softmax(component_logits, dim=2)  # [Q, C, M]
            component_resp = class_resp[:, :, None] * comp_given_class

            # =========================
            # <configured>. M-step 更新每类 GMM components
            # =========================
            # support contribution
            support_weight = support_resp  # [C, K, M]
            support_mass = support_weight.sum(dim=1)  # [C, M]
            support_sum = torch.einsum(
                "ckm,ckf->cmf",
                support_weight,
                support
            )

            # query contribution
            query_mass = component_resp.sum(dim=0)  # [C, M]
            query_sum = torch.einsum(
                "qcm,qf->cmf",
                component_resp,
                query_v
            )

            total_mass = support_mass + query_mass  # [C, M]
            mu = (support_sum + query_sum) / total_mass[:, :, None].clamp_min(self.eps)

            # 更新 mixture weights
            pi_cm = total_mass / total_mass.sum(dim=1, keepdim=True).clamp_min(self.eps)
            pi_cm = pi_cm.clamp_min(self.eps)
            pi_cm = pi_cm / pi_cm.sum(dim=1, keepdim=True).clamp_min(self.eps)

            # 可选：更新 diagonal precision
            if self.use_diag_precision:
                # 用 support + query 的加权方差估计对角 precision
                support_var_sum = torch.einsum(
                    "ckm,ckmf->cmf",
                    support_weight,
                    (support[:, :, None, :] - mu[:, None, :, :]).pow(2)
                )

                query_var_sum = torch.einsum(
                    "qcm,qcmf->cmf",
                    component_resp,
                    (query_v[:, None, None, :] - mu[None, :, :, :]).pow(2)
                )

                var = (support_var_sum + query_var_sum) / total_mass[:, :, None].clamp_min(self.eps)
                precision_diag = 1.0 / var.clamp_min(self.eps)

        # =========================
        # 最终类别 logits
        # =========================
        final_component_logits = self._gaussian_component_logits(
            query_v,
            mu,
            precision_diag
        ) + torch.log(pi_cm[None, :, :].clamp_min(self.eps))

        final_class_logits = torch.logsumexp(final_component_logits, dim=2)

        final_class_prior = class_resp.mean(dim=0).clamp_min(self.eps)
        final_class_prior = final_class_prior / final_class_prior.sum().clamp_min(self.eps)

        final_logits = (
            final_class_logits - lambdas[-1] * torch.log(final_class_prior[None, :])
        ) / temperatures[-1].clamp_min(self.eps)

        probabilities = F.softmax(final_logits, dim=1)
        distances = -final_logits

        # 兼容原框架：输出 [C*M, F]
        final_mu = mu.reshape(n_ways * M, feature_dim)

        if precision_diag is not None:
            final_precision_matrices = torch.diag_embed(
                precision_diag.reshape(n_ways * M, feature_dim)
            )
        else:
            final_precision_matrices = torch.eye(
                feature_dim,
                device=device,
                dtype=dtype
            ).unsqueeze(0).repeat(n_ways * M, 1, 1)

        return (
            final_logits,
            probabilities,
            distances,
            final_mu,
            final_precision_matrices,
            component_resp,
            M,
            pi_cm,
        )

class MultiPrototypeGPNLoss(nn.Module):
    '\n    多原型GPN损失 - 任务内动态原型合并（返回mask版本）\n    \n    设计原则：\n    <configured>. 每个任务内独立合并原型\n    <configured>. 返回任务级别的prototype_mask供统计使用\n    <configured>. 不跨任务持久化状态\n    '
    def __init__(
        self,
        use_multi=None,
        use_Mdistance=None,
        merge_threshold=None,
        use_unem=None,
        unem_layers=None,
        unem_use_diag_precision=None,
        unem_gmm_components=None,
    ):
        use_multi = _cfg_resolve('loss_ada_num_clean_UNEM.py.MultiPrototypeGPNLoss.__init__.use_multi', use_multi)
        use_Mdistance = _cfg_resolve('loss_ada_num_clean_UNEM.py.MultiPrototypeGPNLoss.__init__.use_Mdistance', use_Mdistance)
        merge_threshold = _cfg_resolve('loss_ada_num_clean_UNEM.py.MultiPrototypeGPNLoss.__init__.merge_threshold', merge_threshold)
        use_unem = _cfg_resolve('loss_ada_num_clean_UNEM.py.MultiPrototypeGPNLoss.__init__.use_unem', use_unem)
        unem_layers = _cfg_resolve('loss_ada_num_clean_UNEM.py.MultiPrototypeGPNLoss.__init__.unem_layers', unem_layers)
        unem_use_diag_precision = _cfg_resolve('loss_ada_num_clean_UNEM.py.MultiPrototypeGPNLoss.__init__.unem_use_diag_precision', unem_use_diag_precision)
        super().__init__()

        if use_Mdistance:
            self.distance_metric = MahalanobisDistance()
        else:
            self.distance_metric = EuclideanDistance()

        self.use_multi = use_multi
        self.use_Mdistance = use_Mdistance

        # UNEM-GMM 开关
        self.use_unem = use_unem
        self.unem = UNEMClassGMMHead(
            n_layers=unem_layers,
            gmm_components=unem_gmm_components,
            use_diag_precision=unem_use_diag_precision,
        ) if use_unem else None

        # 原型合并参数：保留，用于非 UNEM 模式
        self.merge_threshold = merge_threshold

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
        使用马氏距离和图连通分量方法一次性合并所有相似原型

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

            # 构建连通图：使用马氏距离判断连接
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
    def forward(
        self,
        query_v,
        prototypes,
        precision_matrices,
        query_labels,
        epoch=None,
        n_ways=None,
        k_shot=None,
    ):
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

        # 不再从 query_labels 推断 n_ways/k_shot，优先使用 trainer 传入的任务参数
        if n_ways is None:
            n_ways = len(torch.unique(query_labels))

        if k_shot is None:
            k_shot = prototypes.shape[0] // n_ways

        # ==================== UNEM-GMM 分支 ====================
        if self.use_unem:
            (
                logits,
                probabilities,
                min_distances,
                gmm_prototypes,
                gmm_precisions,
                component_resp,
                gmm_components,
                pi_cm,
            ) = self.unem(
                query_v=query_v,
                support_prototypes=prototypes,
                support_precision_matrices=precision_matrices,
                n_ways=n_ways,
                k_shot=k_shot,
            )

            loss = F.cross_entropy(logits, query_labels)

            # 每类有 gmm_components 个 Gaussian components
            prototype_mask = torch.ones(
                n_ways,
                gmm_components,
                dtype=torch.bool,
                device=prototypes.device
            )

            # component_resp: [Q, C, M]
            closest_component_idx = torch.argmax(component_resp, dim=2)  # [Q, C]

            merge_info = {
                'merge_count': n_ways * max(k_shot - gmm_components, 0),
                'active_prototypes': [gmm_components] * n_ways,
                'total_active': n_ways * gmm_components,
                'mode': 'UNEM-Class-GMM',
                'gmm_components': gmm_components,
            }

            self.task_count += 1

            return {
                'loss': loss,
                'probabilities': probabilities,
                'distances': min_distances,
                'prototypes': gmm_prototypes,
                'precision_matrices': gmm_precisions,
                'prototype_mask': prototype_mask,
                'n_ways': n_ways,
                'k_shot': gmm_components,
                'closest_prototype_idx': closest_component_idx,
                'component_resp': component_resp,
                'mixture_weights': pi_cm,
                'merge_info': merge_info,
            }
        # ==================== UNEM 分支结束 ====================
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
            'total_active': sum(active_counts)
        }

        self.global_merge_count += merge_count

        self.task_count += 1

        # 计算距离和loss
        if self.use_multi:
            all_distances = self.distance_metric(query_v, prototypes, precision_matrices)

            proto_per_class = prototype_mask.shape[1]
            distances_reshaped = all_distances.view(batch_size, n_ways, proto_per_class)

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
