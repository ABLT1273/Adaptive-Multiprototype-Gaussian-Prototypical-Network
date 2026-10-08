from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPNLoss(nn.Module):
    """
    GPN损失函数 - 马氏距离 + 交叉熵
    """
    def __init__(self):
        super().__init__()

    def compute_mahalanobis_distance(self, v, prototypes, precision_matrices):
        """
        计算马氏距离

        Args:
            v: 查询样本特征 [batch_size, feature_dim]
            prototypes: 各类原型 [n_ways, feature_dim]
            precision_matrices: 各类精度矩阵 [n_ways, feature_dim, feature_dim]

        Returns:
            distances: 马氏距离 [batch_size, n_ways]
        """
        batch_size, feature_dim = v.shape
        n_ways = prototypes.shape[0]

        # 扩维以便广播计算
        v_expanded = v.unsqueeze(1).expand(-1, n_ways, -1)  # [batch_size, n_ways, feature_dim]
        prototypes_expanded = prototypes.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_ways, feature_dim]

        # 计算差异
        diff = v_expanded - prototypes_expanded  # [batch_size, n_ways, feature_dim]

        distances = torch.zeros(batch_size, n_ways, device=v.device)

        for i in range(n_ways):
            diff_i = diff[:, i, :]  # [batch_size, feature_dim]
            P_i = precision_matrices[i]  # [feature_dim, feature_dim]

            # 计算二次型: (x-μ)^T P (x-μ)
            # 使用批矩阵乘法提高效率
            temp = torch.mm(diff_i, P_i)  # [batch_size, feature_dim]
            quadratic_form = torch.sum(diff_i * temp, dim=1)  # [batch_size]
            distances[:, i] = torch.sqrt(torch.clamp(quadratic_form, min=1e-8))

        return distances

    def forward(self, query_v, prototypes, precision_matrices, query_labels):
        """
        计算损失

        Args:
            query_v: 查询样本特征 [batch_size, feature_dim]
            prototypes: 各类原型 [n_ways, feature_dim]
            precision_matrices: 各类精度矩阵 [n_ways, feature_dim, feature_dim]
            query_labels: 查询集标签 [batch_size]
        """
        # 计算马氏距离
        distances = self.compute_mahalanobis_distance(query_v, prototypes, precision_matrices)

        # 将距离转换为logits（负距离）
        logits = -distances

        # 计算softmax概率
        probabilities = F.softmax(logits, dim=1)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, query_labels)

        return loss, probabilities, distances


class LearnableMetricDistance(nn.Module):
    '\n    可学习度量距离 - 自适应学习最优距离函数\n    \n    核心思想：\n    <configured>. 用神经网络学习距离度量（而非手工设计）\n    <configured>. 端到端训练，自动适应数据分布\n    <configured>. 结合多种距离的优势\n    \n    优于马氏距离：\n    - 非线性变换能力\n    - 自适应特征重要性\n    - 学习复杂的特征交互\n    '
    def __init__(self, feature_dim, hidden_dim=None):
        hidden_dim = _cfg_resolve('loss_Copy1.py.LearnableMetricDistance.__init__.hidden_dim', hidden_dim)
        super().__init__()

        # 特征变换网络（学习更好的度量空间）
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

        # 可学习的特征权重（替代固定精度矩阵）
        self.feature_weights = nn.Parameter(torch.ones(feature_dim))

        # 距离融合参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 欧氏距离权重
        self.beta = nn.Parameter(torch.tensor(0.3))   # 余弦距离权重
        self.gamma = nn.Parameter(torch.tensor(0.2))  # 相关性权重

    def to(self, device):
        """重写to方法确保所有组件都移动到正确设备"""
        super().to(device)
        return self

    def get_branch_weights(self):
        """获取各分支的融合权重（用于分析）"""

        weights_sum = torch.abs(self.alpha) + torch.abs(self.beta) + torch.abs(self.gamma)
        return {
            'euclidean': (torch.abs(self.alpha) / weights_sum).item(),
            'cosine': (torch.abs(self.beta) / weights_sum).item(),
            'correlation': (torch.abs(self.gamma) / weights_sum).item()
        }
    def forward(self, v, prototypes):
        """
        Args:
            v: [batch_size, feature_dim]
            prototypes: [n_ways, feature_dim]
        Returns:
            distances: [batch_size, n_ways]
        """
        batch_size = v.shape[0]
        n_ways = prototypes.shape[0]

        # <configured>. 特征空间变换（学习更好的表示）
        v_trans = self.transform(v)  # [batch_size, feature_dim]
        proto_trans = self.transform(prototypes)  # [n_ways, feature_dim]

        # <configured>. 加权欧氏距离
        weights = F.softplus(self.feature_weights).unsqueeze(0).unsqueeze(0)  # [<configured>, <configured>, feature_dim]
        diff = v_trans.unsqueeze(1) - proto_trans.unsqueeze(0)  # [batch, ways, dim]
        weighted_diff = diff * weights
        euclidean_dist = torch.sqrt((weighted_diff ** 2).sum(-1) + 1e-8)  # [batch, ways]

        # <configured>. 余弦距离（捕获方向信息）
        v_norm = F.normalize(v_trans, p=2, dim=1)
        proto_norm = F.normalize(proto_trans, p=2, dim=1)
        cosine_sim = torch.mm(v_norm, proto_norm.t())  # [batch, ways]
        cosine_dist = 1 - cosine_sim

        # <configured>. 特征相关性距离（捕获高阶交互）
        correlation_dist = self._correlation_distance(v_trans, proto_trans)

        # <configured>. 自适应融合（权重归一化）
        weights_sum = torch.abs(self.alpha) + torch.abs(self.beta) + torch.abs(self.gamma)
        w1 = torch.abs(self.alpha) / weights_sum
        w2 = torch.abs(self.beta) / weights_sum
        w3 = torch.abs(self.gamma) / weights_sum

        distances = w1 * euclidean_dist + w2 * cosine_dist + w3 * correlation_dist

        return distances

    def _correlation_distance(self, v, prototypes):
        """计算特征相关性距离"""
        batch_size = v.shape[0]
        n_ways = prototypes.shape[0]
        device = v.device  # 获取设备信息

        # 计算特征间的协方差
        v_centered = v - v.mean(dim=1, keepdim=True)
        proto_centered = prototypes - prototypes.mean(dim=1, keepdim=True)

        distances = torch.zeros(batch_size, n_ways, device=device)  # 在正确设备上创建
        for i in range(n_ways):
            # 皮尔逊相关系数
            corr = F.cosine_similarity(v_centered, proto_centered[i:i+1], dim=1)
            distances[:, i] = 1 - torch.abs(corr)

        return distances


class AdaptiveKernelDistance(nn.Module):
    '\n    自适应核距离 - 基于核方法的非线性距离\n    \n    核心思想：\n    <configured>. 用RBF核隐式映射到高维空间\n    <configured>. 自适应调整核带宽\n    <configured>. 多核融合提升表达能力\n    \n    优于马氏距离：\n    - 捕获非线性关系\n    - 自适应局部结构\n    - 多尺度特征融合\n    '
    def __init__(self, feature_dim, n_kernels=None):
        n_kernels = _cfg_resolve('loss_Copy1.py.AdaptiveKernelDistance.__init__.n_kernels', n_kernels)
        super().__init__()

        # 多个可学习的核带宽
        self.bandwidths = nn.Parameter(torch.linspace(0.1, 2.0, n_kernels))

        # 核权重
        self.kernel_weights = nn.Parameter(torch.ones(n_kernels) / n_kernels)

    def to(self, device):
        """重写to方法确保所有组件都移动到正确设备"""
        super().to(device)
        return self

    def forward(self, v, prototypes):
        """
        Args:
            v: [batch_size, feature_dim]
            prototypes: [n_ways, feature_dim]
        Returns:
            distances: [batch_size, n_ways]
        """
        batch_size = v.shape[0]
        n_ways = prototypes.shape[0]
        device = v.device  # 获取设备信息

        # 计算欧氏距离平方
        v_expanded = v.unsqueeze(1)  # [batch, <configured>, dim]
        proto_expanded = prototypes.unsqueeze(0)  # [<configured>, ways, dim]
        sq_dist = ((v_expanded - proto_expanded) ** 2).sum(-1)  # [batch, ways]

        # 多核RBF距离
        distances = torch.zeros_like(sq_dist, device=device)  # 在正确设备上创建
        weights_norm = F.softmax(self.kernel_weights, dim=0)

        for i, (bw, weight) in enumerate(zip(self.bandwidths, weights_norm)):
            kernel_sim = torch.exp(-sq_dist / (2 * bw ** 2 + 1e-8))
            kernel_dist = torch.sqrt(2 - 2 * kernel_sim + 1e-8)
            distances += weight * kernel_dist

        return distances


class BilinearSimilarityDistance(nn.Module):
    '\n    双线性相似度距离 - 学习特征间的交互矩阵\n    \n    核心思想：\n    <configured>. 学习特征维度之间的交互权重\n    <configured>. 双线性形式：sim(x,y) = x^T W y\n    <configured>. 比内积更强的表达能力\n    \n    优于马氏距离：\n    - 学习非对称特征交互\n    - 端到端优化\n    - 低秩分解降低参数量\n    '
    def __init__(self, feature_dim, rank=None):
        rank = _cfg_resolve('loss_Copy1.py.BilinearSimilarityDistance.__init__.rank', rank)
        super().__init__()

        # 低秩双线性矩阵: W = U @ V^T
        self.U = nn.Parameter(torch.randn(feature_dim, rank) / math.sqrt(rank))
        self.V = nn.Parameter(torch.randn(feature_dim, rank) / math.sqrt(rank))

        # 可选的特征变换
        self.use_transform = True
        if self.use_transform:
            self.transform = nn.Linear(feature_dim, feature_dim, bias=False)

    def to(self, device):
        """重写to方法确保所有组件都移动到正确设备"""
        super().to(device)
        return self

    def forward(self, v, prototypes):
        """
        Args:
            v: [batch_size, feature_dim]
            prototypes: [n_ways, feature_dim]
        Returns:
            distances: [batch_size, n_ways]
        """
        # 可选特征变换
        if self.use_transform:
            v = self.transform(v)
            prototypes = self.transform(prototypes)

        # 双线性相似度: x^T (U @ V^T) y
        # = (x^T U) @ (V^T y)
        v_proj = torch.mm(v, self.U)  # [batch, rank]
        proto_proj = torch.mm(prototypes, self.V)  # [ways, rank]

        # 计算相似度
        similarity = torch.mm(v_proj, proto_proj.t())  # [batch, ways]

        # 转换为距离（归一化后）
        # 使用负相似度作为距离
        distances = -similarity

        # 归一化到正值范围
        distances = distances - distances.min()

        return distances


class AttentionBasedDistance(nn.Module):
    '\n    注意力距离 - 基于注意力机制的自适应距离\n    \n    核心思想：\n    <configured>. 查询-原型注意力权重\n    <configured>. 特征维度的动态重要性\n    <configured>. 上下文感知的距离计算\n    \n    优于马氏距离：\n    - 动态适应不同样本\n    - 自动发现关键特征\n    - 多头注意力捕获多方面相似性\n    '
    def __init__(self, feature_dim, num_heads=None):
        num_heads = _cfg_resolve('loss_Copy1.py.AttentionBasedDistance.__init__.num_heads', num_heads)
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        # 多头注意力
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # 距离预测网络
        self.distance_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, _cfg_require('loss_Copy1.py.AttentionBasedDistance.__init__.Linear_arg1'))
        )

    def to(self, device):
        """重写to方法确保所有组件都移动到正确设备"""
        super().to(device)
        return self

    def forward(self, v, prototypes):
        """
        Args:
            v: [batch_size, feature_dim]
            prototypes: [n_ways, feature_dim]
        Returns:
            distances: [batch_size, n_ways]
        """
        batch_size = v.shape[0]
        n_ways = prototypes.shape[0]
        feature_dim = v.shape[1]
        device = v.device  # 获取设备信息

        # Query, Key, Value投影
        Q = self.q_proj(v)  # [batch, feature_dim]
        K = self.k_proj(prototypes)  # [ways, feature_dim]
        V = self.v_proj(prototypes)  # [ways, feature_dim]

        # 重塑为多头
        Q = Q.view(batch_size, self.num_heads, self.head_dim)  # [batch, heads, head_dim]
        K = K.view(n_ways, self.num_heads, self.head_dim)  # [ways, heads, head_dim]
        V = V.view(n_ways, self.num_heads, self.head_dim)

        # 计算注意力分数
        scores = torch.einsum('bhd,whd->bhw', Q, K) / math.sqrt(self.head_dim)  # [batch, heads, ways]
        attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, ways]

        # 加权聚合
        attended = torch.einsum('bhw,whd->bhd', attn_weights, V)  # [batch, heads, head_dim]
        attended = attended.reshape(batch_size, feature_dim)  # [batch, feature_dim]

        # 计算距离（通过神经网络）
        distances = torch.zeros(batch_size, n_ways, device=device)  # 在正确设备上创建
        for i in range(n_ways):
            # 拼接特征差异
            diff = v - prototypes[i:i+1]
            concat_feat = torch.cat([diff, attended, v * prototypes[i:i+1]], dim=1)

            # 通过网络预测距离
            dist = self.distance_net(diff).squeeze(-1)
            distances[:, i] = torch.abs(dist)

        return distances


class HybridMetricDistance(nn.Module):
    '\n    混合度量距离 - 融合多种先进距离的集大成者\n    \n    结合：\n    <configured>. 可学习度量（非线性变换）\n    <configured>. 自适应核（多尺度）\n    <configured>. 双线性相似度（特征交互）\n    <configured>. 注意力机制（动态权重）\n    \n    这是目前最强的距离度量方案\n    '
    def __init__(self, feature_dim, use_learnable=None, use_kernel=None,
                 use_bilinear=None, use_attention=None):
        use_learnable = _cfg_resolve('loss_Copy1.py.HybridMetricDistance.__init__.use_learnable', use_learnable)
        use_kernel = _cfg_resolve('loss_Copy1.py.HybridMetricDistance.__init__.use_kernel', use_kernel)
        use_bilinear = _cfg_resolve('loss_Copy1.py.HybridMetricDistance.__init__.use_bilinear', use_bilinear)
        use_attention = _cfg_resolve('loss_Copy1.py.HybridMetricDistance.__init__.use_attention', use_attention)
        super().__init__()

        self.use_learnable = use_learnable
        self.use_kernel = use_kernel
        self.use_bilinear = use_bilinear
        self.use_attention = use_attention

        # 各个组件
        if use_learnable:
            self.learnable = LearnableMetricDistance(feature_dim)
        if use_kernel:
            self.kernel = AdaptiveKernelDistance(feature_dim, n_kernels=_cfg_require('loss_Copy1.py.HybridMetricDistance.__init__.n_kernels'))
        if use_bilinear:
            self.bilinear = BilinearSimilarityDistance(feature_dim, rank=_cfg_require('loss_Copy1.py.HybridMetricDistance.__init__.rank'))
        if use_attention:
            self.attention = AttentionBasedDistance(feature_dim, num_heads=_cfg_require('loss_Copy1.py.HybridMetricDistance.__init__.num_heads'))

        # 融合权重（可学习）
        n_components = sum([use_learnable, use_kernel, use_bilinear, use_attention])
        self.fusion_weights = nn.Parameter(torch.ones(n_components) / n_components)

    def to(self, device):
        """重写to方法确保所有组件都移动到正确设备"""
        super().to(device)
        return self

    def forward(self, v, prototypes):
        """
        Args:
            v: [batch_size, feature_dim]
            prototypes: [n_ways, feature_dim]
        Returns:
            distances: [batch_size, n_ways]
        """
        device = v.device  # 获取设备信息
        distances_list = []

        if self.use_learnable:
            distances_list.append(self.learnable(v, prototypes))
        if self.use_kernel:
            distances_list.append(self.kernel(v, prototypes))
        if self.use_bilinear:
            distances_list.append(self.bilinear(v, prototypes))
        if self.use_attention:
            distances_list.append(self.attention(v, prototypes))

        # 堆叠所有距离
        distances_stack = torch.stack(distances_list, dim=0)  # [n_components, batch, ways]

        # 归一化融合权重
        weights = F.softmax(self.fusion_weights, dim=0).view(-1, 1, 1)

        # 加权融合
        distances = (distances_stack * weights).sum(dim=0)  # [batch, ways]

        return distances


class GPNLoss_Advanced(nn.Module):
    """
    高级GPN损失 - 使用先进的距离度量
    """
    def __init__(self, loss_feature_dim, metric_type=None):
        metric_type = _cfg_resolve('loss_Copy1.py.GPNLoss_Advanced.__init__.metric_type', metric_type)
        super().__init__()

        # 选择距离度量
        if metric_type == 'learnable':
            self.distance_metric = LearnableMetricDistance(loss_feature_dim)
        elif metric_type == 'kernel':
            self.distance_metric = AdaptiveKernelDistance(loss_feature_dim)
        elif metric_type == 'bilinear':
            self.distance_metric = BilinearSimilarityDistance(loss_feature_dim)
        elif metric_type == 'attention':
            self.distance_metric = AttentionBasedDistance(loss_feature_dim)
        elif metric_type == 'hybrid':
            self.distance_metric = HybridMetricDistance(loss_feature_dim)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")


    def compute_distance(self, query_v, prototypes, precision_matrices=None):
        """
        计算距离的统一接口

        Args:
            query_v: [batch_size, loss_feature_dim]
            prototypes: [n_ways, loss_feature_dim]
            precision_matrices: [n_ways, loss_feature_dim, loss_feature_dim]
                              (可选，仅马氏距离需要)
        Returns:
            distances: [batch_size, n_ways]
        """
        return self.distance_metric(query_v, prototypes)

    def forward(self, query_v, prototypes, precision_matrices, query_labels):
        """
        Args:
            query_v: [batch_size, loss_feature_dim]
            prototypes: [n_ways, loss_feature_dim]
            precision_matrices: [n_ways, loss_feature_dim, loss_feature_dim]
                              (仅当使用马氏距离时需要)
            query_labels: [batch_size]
        Returns:
            loss: 交叉熵损失
            probabilities: softmax概率 [batch_size, n_ways]
            distances: 距离矩阵 [batch_size, n_ways]
        """
        # 计算距离（根据是否需要精度矩阵）
        distances = self.compute_distance(query_v, prototypes, precision_matrices)

        # 转换为logits（负距离）
        logits = -distances

        # 计算概率和损失
        probabilities = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, query_labels)

        return loss, probabilities, distances
