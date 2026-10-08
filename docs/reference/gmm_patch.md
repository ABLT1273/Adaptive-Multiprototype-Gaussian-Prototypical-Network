# Historical GMM integration patch

This is integration reference material, not an importable module or runnable experiment.
The integrated implementation lives in `amgpn/legacy/v4/gmm.py`.

```python
# GPN_V4_adaMulti GMM修改补丁
# 此文件包含需要添加到GPN_V4_adaMulti.py中的GMM相关代码

'\n============================================\n第一部分：修改导入语句\n============================================\n将第<configured>行的导入语句从：\n    from amgpn.legacy.v4.adaptive_loss import MultiPrototypeGPNLoss\n改为：\n    from amgpn.legacy.v4.gmm_loss import MultiPrototypeGPNLoss\n'

'\n============================================\n第二部分：修改GPNTrainer.__init__方法\n============================================\n在__init__方法的参数列表中添加GMM相关参数（约第<configured>-<configured>行）：\n'
from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format

def __init__(self, model, device, use_Mdistance=None, use_multi=None, save_middle=None, mlr=None,
             prototypes_per_class=None,
             use_edge_corner=None, lambda_L=None, lambda_H=None,
             cut_ratio=None, resize=None,
             # 新增：GMM相关参数
             use_gmm=None, gmm_components=None,
             # 原型合并相关参数
             merge_threshold=None, merge_start_epoch=None, merge_interval=None):
    """
    Args:
        model: GPN骨干网络
        device: 训练设备
        use_multi: 是否使用多原型模式
        use_Mdistance: 是否使用马氏距离
        use_gmm: 是否使用GMM混合高斯模式
        gmm_components: GMM中每个类的高斯分量数
        save_middle: 是否中途保存最优模型
        mlr: 模型学习率
        ... (其他参数说明)
    """
    # 在原有代码基础上添加：
    use_Mdistance = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.use_Mdistance', use_Mdistance)
    use_multi = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.use_multi', use_multi)
    save_middle = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.save_middle', save_middle)
    mlr = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.mlr', mlr)
    prototypes_per_class = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.prototypes_per_class', prototypes_per_class)
    use_edge_corner = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.use_edge_corner', use_edge_corner)
    lambda_L = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.lambda_L', lambda_L)
    lambda_H = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.lambda_H', lambda_H)
    cut_ratio = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.cut_ratio', cut_ratio)
    resize = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.resize', resize)
    use_gmm = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.use_gmm', use_gmm)
    gmm_components = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.gmm_components', gmm_components)
    merge_threshold = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.merge_threshold', merge_threshold)
    merge_start_epoch = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.merge_start_epoch', merge_start_epoch)
    merge_interval = _cfg_resolve('GMM/GPN_V4_GMM_patch.py.__init__.merge_interval', merge_interval)
    self.use_gmm = use_gmm
    self.gmm_components = gmm_components

    # 修改loss_fn初始化（约第<configured>-<configured>行）：
    self.loss_fn = MultiPrototypeGPNLoss(
        prototypes_per_class=prototypes_per_class,
        use_multi=self.use_multi,
        use_Mdistance=self.use_Mdistance,
        use_gmm=self.use_gmm,  # 新增
        gmm_components=self.gmm_components,  # 新增
        merge_threshold=merge_threshold,
        merge_start_epoch=merge_start_epoch,
    )

    # 在初始化结束时添加提示（约第<configured>行）：
    if self.use_gmm:
        print(f"GMM混合高斯模式启用: 每类{gmm_components}个高斯分量")


'\n============================================\n第三部分：添加GMM参数估计方法\n============================================\n在compute_gaussian_prototypes方法之后（约第<configured>行）添加以下新方法：\n'

def compute_gmm_parameters(self, support_v, support_s, support_labels, n_ways):
    """
    从支持集样本估计GMM参数

    使用K-Means++初始化 + EM算法估计GMM参数

    Args:
        support_v: [n_samples, feature_dim] 特征向量
        support_s: [n_samples, feature_dim] 不确定性特征
        support_labels: [n_samples] 标签
        n_ways: 类别数

    Returns:
        gmm_params: 字典包含
            - means: [n_ways, n_components, feature_dim]
            - covariances: [n_ways, n_components, feature_dim, feature_dim]
            - weights: [n_ways, n_components]
    """
    feature_dim = support_v.shape[1]
    n_components = self.gmm_components

    means = torch.zeros(n_ways, n_components, feature_dim, device=self.device)
    covariances = torch.zeros(n_ways, n_components, feature_dim, feature_dim, device=self.device)
    weights = torch.zeros(n_ways, n_components, device=self.device)

    for class_idx in range(n_ways):
        # 获取该类的所有样本
        class_mask = (support_labels == class_idx)
        class_features = support_v[class_mask]  # [n_class_samples, feature_dim]
        n_class_samples = class_features.shape[0]

        # 如果样本数少于分量数，调整分量数
        actual_components = min(n_components, n_class_samples)

        if actual_components == 1:
            # 只有一个分量，使用均值和协方差
            means[class_idx, 0] = class_features.mean(dim=0)
            if n_class_samples > 1:
                cov = torch.cov(class_features.t())
                covariances[class_idx, 0] = cov + torch.eye(feature_dim, device=self.device) * 1e-6
            else:
                covariances[class_idx, 0] = torch.eye(feature_dim, device=self.device)
            weights[class_idx, 0] = 1.0

        else:
            # 使用K-Means++初始化
            init_means = self._kmeans_plusplus_init(class_features, actual_components)

            # EM算法估计GMM参数
            comp_means, comp_covs, comp_weights = self._em_algorithm(
                class_features, init_means, max_iter=_cfg_require('GMM/GPN_V4_GMM_patch.py.compute_gmm_parameters.max_iter')
            )

            # 保存参数
            means[class_idx, :actual_components] = comp_means
            covariances[class_idx, :actual_components] = comp_covs
            weights[class_idx, :actual_components] = comp_weights

            # 如果实际分量数少于设定值，用最后一个分量填充
            if actual_components < n_components:
                for i in range(actual_components, n_components):
                    means[class_idx, i] = comp_means[-1]
                    covariances[class_idx, i] = comp_covs[-1]
                    weights[class_idx, i] = 0.0

    return {
        'means': means,
        'covariances': covariances,
        'weights': weights
    }

def _kmeans_plusplus_init(self, data, n_clusters):
    """
    K-Means++初始化

    Args:
        data: [n_samples, feature_dim]
        n_clusters: 簇的数量

    Returns:
        centers: [n_clusters, feature_dim]
    """
    n_samples, feature_dim = data.shape
    centers = torch.zeros(n_clusters, feature_dim, device=self.device)

    # 随机选择第一个中心
    idx = torch.randint(0, n_samples, (1,)).item()
    centers[0] = data[idx]

    # 选择剩余的中心
    for i in range(1, n_clusters):
        # 计算每个点到最近中心的距离
        distances = torch.cdist(data, centers[:i])  # [n_samples, i]
        min_distances = torch.min(distances, dim=1)[0]  # [n_samples]

        # 根据距离的平方选择下一个中心（概率与距离成正比）
        probs = min_distances ** 2
        probs = probs / probs.sum()

        # 根据概率选择
        idx = torch.multinomial(probs, 1).item()
        centers[i] = data[idx]

    return centers

def _em_algorithm(self, data, init_means, max_iter=None, tol=None):
    """
    EM算法估计GMM参数

    Args:
        data: [n_samples, feature_dim]
        init_means: [n_components, feature_dim] 初始均值
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        means: [n_components, feature_dim]
        covariances: [n_components, feature_dim, feature_dim]
        weights: [n_components]
    """
    max_iter = _cfg_resolve('GMM/GPN_V4_GMM_patch.py._em_algorithm.max_iter', max_iter)
    tol = _cfg_resolve('GMM/GPN_V4_GMM_patch.py._em_algorithm.tol', tol)
    n_samples, feature_dim = data.shape
    n_components = init_means.shape[0]

    # 初始化参数
    means = init_means.clone()
    covariances = torch.stack([
        torch.eye(feature_dim, device=self.device) for _ in range(n_components)
    ])
    weights = torch.ones(n_components, device=self.device) / n_components

    prev_log_likelihood = float('-inf')

    for iteration in range(max_iter):
        # E步：计算责任度（posterior probabilities）
        responsibilities = self._compute_responsibilities(
            data, means, covariances, weights
        )  # [n_samples, n_components]

        # M步：更新参数
        Nk = responsibilities.sum(dim=0)  # [n_components]

        # 更新权重
        weights = Nk / n_samples

        # 更新均值
        for k in range(n_components):
            means[k] = (responsibilities[:, k:k+1] * data).sum(dim=0) / (Nk[k] + 1e-10)

        # 更新协方差
        for k in range(n_components):
            diff = data - means[k:k+1]  # [n_samples, feature_dim]
            weighted_diff = responsibilities[:, k:k+1] * diff  # [n_samples, feature_dim]
            cov = (weighted_diff.t() @ diff) / (Nk[k] + 1e-10)
            # 添加正则化确保正定
            covariances[k] = cov + torch.eye(feature_dim, device=self.device) * 1e-6

        # 计算对数似然（用于检查收敛）
        log_likelihood = self._compute_log_likelihood(data, means, covariances, weights)

        # 检查收敛
        if abs(log_likelihood - prev_log_likelihood) < tol:
            break

        prev_log_likelihood = log_likelihood

    return means, covariances, weights

def _compute_responsibilities(self, data, means, covariances, weights):
    """
    计算责任度（E步）

    Args:
        data: [n_samples, feature_dim]
        means: [n_components, feature_dim]
        covariances: [n_components, feature_dim, feature_dim]
        weights: [n_components]

    Returns:
        responsibilities: [n_samples, n_components]
    """
    n_samples = data.shape[0]
    n_components = means.shape[0]

    # 计算每个分量的概率密度
    log_probs = torch.zeros(n_samples, n_components, device=self.device)

    for k in range(n_components):
        diff = data - means[k:k+1]  # [n_samples, feature_dim]

        # 计算精度矩阵
        try:
            precision = torch.linalg.inv(covariances[k])
            sign, logdet = torch.linalg.slogdet(covariances[k])
            if sign <= 0:
                logdet = torch.tensor(0.0, device=self.device)
        except:
            precision = torch.eye(data.shape[1], device=self.device)
            logdet = torch.tensor(0.0, device=self.device)

        # 计算马氏距离
        mahalanobis_sq = torch.sum(diff @ precision * diff, dim=1)

        # 对数概率密度
        log_prob = -0.5 * (mahalanobis_sq + logdet + data.shape[1] * torch.log(torch.tensor(2 * 3.14159265359)))
        log_probs[:, k] = log_prob + torch.log(weights[k] + 1e-10)

    # 使用log-sum-exp技巧计算责任度
    log_sum = torch.logsumexp(log_probs, dim=1, keepdim=True)
    log_responsibilities = log_probs - log_sum
    responsibilities = torch.exp(log_responsibilities)

    return responsibilities

def _compute_log_likelihood(self, data, means, covariances, weights):
    """
    计算数据的对数似然

    Args:
        data: [n_samples, feature_dim]
        means: [n_components, feature_dim]
        covariances: [n_components, feature_dim, feature_dim]
        weights: [n_components]

    Returns:
        log_likelihood: 标量
    """
    n_samples = data.shape[0]
    n_components = means.shape[0]

    log_probs = torch.zeros(n_samples, n_components, device=self.device)

    for k in range(n_components):
        diff = data - means[k:k+1]

        try:
            precision = torch.linalg.inv(covariances[k])
            sign, logdet = torch.linalg.slogdet(covariances[k])
            if sign <= 0:
                logdet = torch.tensor(0.0, device=self.device)
        except:
            precision = torch.eye(data.shape[1], device=self.device)
            logdet = torch.tensor(0.0, device=self.device)

        mahalanobis_sq = torch.sum(diff @ precision * diff, dim=1)
        log_prob = -0.5 * (mahalanobis_sq + logdet + data.shape[1] * torch.log(torch.tensor(2 * 3.14159265359)))
        log_probs[:, k] = log_prob + torch.log(weights[k] + 1e-10)

    # 计算总对数似然
    log_likelihood = torch.logsumexp(log_probs, dim=1).sum()

    return log_likelihood


'\n============================================\n第四部分：修改_single_task_forward方法\n============================================\n在_single_task_forward方法中（约第<configured>-<configured>行），添加GMM模式的处理：\n'

def _single_task_forward(self, support_signals, support_labels, query_signals, query_labels):
    """单个任务的前向传播（支持GMM模式）"""
    # ... (前面的预处理代码保持不变)

    # 标签重映射（这部分保持不变）
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

    # ========== 关键修改：根据模式选择不同的原型计算方式 ==========
    if self.use_gmm:
        # GMM模式：估计每个类的GMM参数
        gmm_params = self.compute_gmm_parameters(
            support_v, support_s, remapped_support_labels, n_ways
        )

        # 调用损失函数
        output = self.loss_fn(
            query_v,
            prototypes=None,  # GMM模式不需要传统原型
            precision_matrices=None,
            query_labels=remapped_query_labels,
            gmm_params=gmm_params,
            epoch=self.current_epoch
        )

    elif self.use_multi:
        # 多原型模式：使用所有support样本作为原型
        k_shot = len(support_labels) // n_ways
        prototypes = support_v
        precision_matrices = self.compute_precision_matrices_from_support(support_s)

        output = self.loss_fn(
            query_v,
            prototypes,
            precision_matrices,
            remapped_query_labels,
            epoch=self.current_epoch
        )
    else:
        # 单原型模式：聚合每类的support样本
        k_shot = len(support_labels) // n_ways
        prototypes, precision_matrices = self.compute_gaussian_prototypes(
            support_v, support_s, remapped_support_labels, n_ways
        )

        distances = self.loss_fn.distance_metric(query_v, prototypes, precision_matrices)
        logits = -distances
        probabilities = F.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, remapped_query_labels)

        output = {
            'loss': loss,
            'probabilities': probabilities,
            'distances': distances,
            'prototypes': prototypes,
            'precision_matrices': precision_matrices,
        }

    loss = output['loss']
    probabilities = output['probabilities']

    # 计算准确率
    predictions = torch.argmax(probabilities, dim=1)
    accuracy = (predictions == remapped_query_labels).float().mean()

    return loss, accuracy.item(), output


'\n============================================\n第五部分：修改evaluate方法\n============================================\n在evaluate方法中（约第<configured>-<configured>行），添加GMM模式的处理。\n主要修改在"根据use_multi选择不同的原型计算方式"部分：\n'

# 在evaluate方法中，找到以下代码块并修改：
# ========== 关键修复：根据模式选择不同的原型计算方式 ==========
if self.use_gmm:
    # GMM模式
    gmm_params = self.compute_gmm_parameters(
        support_v, support_s, remapped_support_labels, n_ways
    )

    output = self.loss_fn(
        query_v,
        prototypes=None,
        precision_matrices=None,
        query_labels=remapped_query_labels,
        gmm_params=gmm_params,
        epoch=self.current_epoch
    )

    logits = -output['distances']
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == remapped_query_labels).float().mean()
    total_acc += accuracy.item()

elif self.use_multi:
    # 多原型模式（原有代码保持不变）
    k_shot = len(support_labels) // n_ways
    prototypes = support_v
    precision_matrices = self.compute_precision_matrices_from_support(support_s)

    output = self.loss_fn(
        query_v,
        prototypes,
        precision_matrices,
        remapped_query_labels,
        epoch=self.current_epoch
    )

    prototype_mask = output['prototype_mask']
    logits = -output['distances']
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == remapped_query_labels).float().mean()
    total_acc += accuracy.item()

    # ... (原型统计代码保持不变)

else:
    # 单原型模式（原有代码保持不变）
    prototypes, precision_matrices = self.compute_gaussian_prototypes(
        support_v, support_s, remapped_support_labels, n_ways
    )

    distances = self.loss_fn.distance_metric(query_v, prototypes, precision_matrices)
    logits = -distances

    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == remapped_query_labels).float().mean()
    total_acc += accuracy.item()


'\n============================================\n使用说明\n============================================\n\n将上述代码添加到GPN_V4_adaMulti.py后，可以通过以下方式使用GMM模式：\n\n在GPN_V3.ipynb中：\n\n# 创建GMM模式的trainer\nmeta_learner = GPNTrainer(\n    model,\n    device=device,\n    mlr=BLR,\n    use_gmm=True,              # 启用GMM模式\n    gmm_components=<configured>,          # 每个类<configured>个高斯分量\n    prototypes_per_class=<configured>,\n    cut_ratio=<configured>,\n    resize=True\n)\n\n# 或者使用马氏距离+GMM\nmeta_learner = GPNTrainer(\n    model,\n    device=device,\n    mlr=BLR,\n    use_gmm=True,\n    use_Mdistance=True,        # GMM内部会使用马氏距离\n    gmm_components=<configured>,          # 调整分量数\n    prototypes_per_class=<configured>,\n    cut_ratio=<configured>,\n    resize=True\n)\n\n参数说明：\n- use_gmm: 是否启用GMM混合高斯模式\n- gmm_components: 每个类的高斯分量数（建议范围：<configured>-<configured>）\n- use_Mdistance: 在GMM中是否使用马氏距离（GMM本身已经考虑协方差）\n- use_multi: GMM模式下此参数会被忽略\n\n注意事项：\n<configured>. use_gmm, use_multi, 单原型模式是互斥的，只能选择其中一种\n<configured>. gmm_components不应大于每类的样本数（k_shot）\n<configured>. GMM模式计算量较大，训练速度会比单原型模式慢\n<configured>. 建议从较小的gmm_components开始尝试（如<configured>-<configured>）\n'
```
