from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
#在V3基础上修改，用于GMM训练与测试
import sys
sys.path.append("..") #相对路径或绝对路径

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from pathlib import Path
from tqdm import tqdm
from data_loader import *
from loss_ada_num_gmm import MultiPrototypeGPNLoss
from feature_extracter import *
import gc
from data_feature_show import crop_and_rescale_symmetric
from sklearn.metrics import f1_score, precision_score, recall_score
import time

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
    '\n    GPN模型训练器 - 适配动态原型损失函数\n    \n    核心改动：\n    <configured>. 删除所有原型生成逻辑（K-Means、EM聚类等）\n    <configured>. 支持集的每个样本直接作为初始原型\n    <configured>. 由MultiPrototypeGPNLoss自动推断和合并原型\n    '
    def __init__(self, model, device,use_Mdistance=None, use_multi=None,save_middle=None,mlr=None,
                 prototypes_per_class=None,  # 保留参数用于向后兼容
                 use_edge_corner=None, lambda_L=None, lambda_H=None,
                 crop_ratio_h=None,crop_ratio_l=None, resize=None,
                 # 新增：GMM相关参数
                 use_gmm=None, gmm_components=None,
                 # 新增：原型合并相关参数
                 merge_threshold=None, merge_start_epoch=None, merge_interval=None):
        """
        Args:
            model: GPN骨干网络
            device: 训练设备
            use_multi: 是否使用多原型模式
            save_middle: 是否中途保存最优模型
            mlr: 模型学习率
            prototypes_per_class: 最大原型数（向后兼容，实际由支持集大小决定）
            use_edge_corner: 是否使用边缘和角点特征增强
            lambda_L, lambda_H: 噪声不敏感特征参数
            crop_ratio: 裁剪比例
            resize: 是否resize
            use_gmm: 是否使用GMM混合高斯模式
            gmm_components: GMM中每个类的高斯分量数
            merge_threshold: 原型合并的L2距离阈值
            merge_start_epoch: 开始合并的epoch
            merge_interval: 合并检查间隔
        """
        use_Mdistance = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.use_Mdistance', use_Mdistance)
        use_multi = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.use_multi', use_multi)
        save_middle = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.save_middle', save_middle)
        mlr = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.mlr', mlr)
        prototypes_per_class = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.prototypes_per_class', prototypes_per_class)
        use_edge_corner = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.use_edge_corner', use_edge_corner)
        lambda_L = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.lambda_L', lambda_L)
        lambda_H = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.lambda_H', lambda_H)
        crop_ratio_h = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.crop_ratio_h', crop_ratio_h)
        crop_ratio_l = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.crop_ratio_l', crop_ratio_l)
        resize = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.resize', resize)
        use_gmm = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.use_gmm', use_gmm)
        gmm_components = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.gmm_components', gmm_components)
        merge_threshold = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.merge_threshold', merge_threshold)
        merge_start_epoch = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.merge_start_epoch', merge_start_epoch)
        merge_interval = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.__init__.merge_interval', merge_interval)
        self.device = device
        self.use_edge_corner = use_edge_corner
        self.crop_ratio_h = crop_ratio_h
        self.crop_ratio_l=crop_ratio_l
        self.resize = resize
        self.use_gmm = use_gmm
        self.gmm_components = gmm_components

        # 初始化模型
        self.model = model
        self.model.to(device)
        self.use_multi = use_multi
        self.use_Mdistance=use_Mdistance
        self.save_middle=save_middle

        # 初始化损失函数（支持动态原型合并）
        self.loss_fn = MultiPrototypeGPNLoss(
        prototypes_per_class=prototypes_per_class,
        use_multi=self.use_multi,
        use_Mdistance=self.use_Mdistance,
        use_gmm=self.use_gmm,  # 新增
        gmm_components=self.gmm_components,  # 新增
        merge_threshold=merge_threshold,
        merge_start_epoch=merge_start_epoch,
    )
        self.loss_fn.to(device)

        self.lambda_L = lambda_L
        self.lambda_H = lambda_H

        # 初始化特征提取器
        if self.use_edge_corner:
            self.feature_extractor = NoiseInsensitiveFeatureExtractor(
                lambda_L=self.lambda_L,
                lambda_H=self.lambda_H,
                delta=_cfg_require('GPN_V4_GMM.py.GPNTrainer.__init__.delta'),
                gamma=_cfg_require('GPN_V4_GMM.py.GPNTrainer.__init__.gamma'),
                eta_percentile=_cfg_require('GPN_V4_GMM.py.GPNTrainer.__init__.eta_percentile'),
                gaussian_window_size=_cfg_require('GPN_V4_GMM.py.GPNTrainer.__init__.gaussian_window_size')
            )
            # 修改模型第一层
            self._adapt_model_input()

        self.model_params = list(self.model.parameters())
        self.mlr = mlr
        self.optimizer_model = torch.optim.Adam(self.model_params, lr=self.mlr)

        # 训练统计
        self.current_epoch = 0
        self.merge_statistics = []

        if self.use_edge_corner:
            print(f"噪声不敏感特征提取器已启用")
        print(f"GPN模型初始化完成，设备: {self.device}")
        if self.use_multi:
            print(f"多原型模式启用: 初始原型=支持集样本数, 合并阈值={merge_threshold}")
        if self.use_gmm:
            print(f"GMM混合高斯模式启用: 每类{gmm_components}个高斯分量")

    def _adapt_model_input(self):
        '\n        修改模型第一层以接受<configured>通道输入\n        '
        old_conv = self.model.conv1

        # 创建新的<configured>通道卷积层
        new_conv = nn.Conv2d(
            in_channels=_cfg_require('GPN_V4_GMM.py.GPNTrainer._adapt_model_input.in_channels'),
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
        save_path = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.save_model.save_path', save_path)
        filtered_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if not ("total_ops" in k or "total_params" in k)
        }
        checkpoint = {
            'model_state_dict': filtered_state_dict,
            'current_epoch': self.current_epoch,
        }

        if hasattr(self, 'loss_fn') and hasattr(self.loss_fn, 'state_dict'):
            checkpoint['loss_fn_state_dict'] = self.loss_fn.state_dict()
            # 保存原型掩码状态
            if hasattr(self.loss_fn, 'prototype_active_mask') and self.loss_fn.prototype_active_mask is not None:
                checkpoint['prototype_active_mask'] = self.loss_fn.prototype_active_mask
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

        # 加载epoch信息
        if 'current_epoch' in checkpoint:
            self.current_epoch = checkpoint['current_epoch']

        if 'loss_fn_state_dict' in checkpoint:
            if hasattr(self, 'loss_fn') and hasattr(self.loss_fn, 'load_state_dict'):
                self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
                # 恢复原型掩码
                if 'prototype_active_mask' in checkpoint:
                    self.loss_fn.prototype_active_mask = checkpoint['prototype_active_mask']
                print(f"Loss function parameters loaded")
            else:
                print("Warning: Checkpoint contains loss_fn but current trainer doesn't have compatible loss_fn")

        print(f"Model loaded from {model_path}")

    def compute_precision_matrices_from_support(self, support_s):
        """
        从支持集的不确定性特征计算精度矩阵

        Args:
            support_s: [n_ways * k_shot, feature_dim]

        Returns:
            precision_matrices: [n_ways * k_shot, feature_dim, feature_dim]
        """
        sigma = support_s
        n_samples = sigma.shape[0]
        feature_dim = sigma.shape[1]

        precision_matrices = torch.zeros(n_samples, feature_dim, feature_dim, device=self.device)
        for i in range(n_samples):
            precision_matrices[i] = torch.diag(sigma[i])

        return precision_matrices

# ========== 修复后的 _single_task_forward 方法 ==========

    def _single_task_forward(self, support_signals, support_labels, query_signals, query_labels):
        """
        单个任务的前向传播（任务内动态原型合并）

        关键修复：区分use_multi的两种情况
        """
        # 移动到设备
        support_signals = support_signals.to(self.device).float().unsqueeze(1)
        support_labels = support_labels.to(self.device)
        query_signals = query_signals.to(self.device).float().unsqueeze(1)
        query_labels = query_labels.to(self.device)

        # 数据增强
        if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
            processed_support_list = []
            for signal in support_signals.unbind(0):
                processed_support_list.append(crop_and_rescale_symmetric(signal, self.crop_ratio_h,self.crop_ratio_l, self.resize))
            support_signals = torch.stack(processed_support_list, dim=0)

            processed_query_list = []
            for signal in query_signals.unbind(0):
                processed_query_list.append(crop_and_rescale_symmetric(signal, self.crop_ratio_h,self.crop_ratio_l, self.resize))
            query_signals = torch.stack(processed_query_list, dim=0)

        # 边缘特征提取
        if self.use_edge_corner:
            support_signals = self.feature_extractor.forward(support_signals)
            query_signals = self.feature_extractor.forward(query_signals)

        # 前向传播
        support_v, support_s = self.model(support_signals)
        query_v, query_s = self.model(query_signals)

        # 标签重映射
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
        k_shot = len(support_labels) // n_ways

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


    # ========== 修复后的 evaluate 方法 ==========

    def evaluate(self, test_loader, n_ways, show_progress=True,
             show_error_stats=False,    # 改为False
             return_stats=False,
             all_result=False,          # 新增
             return_features=False,     # 新增
             return_predictions=False): # 新增
        '\n        评估模型性能（使用任务级别的prototype_mask统计）\n\n        关键改进：\n        <configured>. 从loss函数的返回值中获取prototype_mask\n        <configured>. 每个任务的mask独立统计\n        <configured>. 不依赖持久化的状态\n        <configured>. 修复了use_multi=False时的索引越界问题\n\n        Args:\n            test_loader: 数据加载器\n            n_ways: 类别数\n            show_progress: 是否显示进度条\n            show_error_stats: 是否显示错误统计信息\n            return_stats: 是否返回详细统计信息\n\n        Returns:\n            avg_acc: 平均准确率\n            eval_stats: 评估统计信息（如果return_stats=True）\n        '
        self.model.eval()

        all_task_metrics = []
        collected_features = []
        collected_labels = []
        collected_preds = []
        collected_targets = []

        total_acc = 0

        # 错误统计结构
        error_details = {}
        global_error_stats = {}
        class_stats = {}

        # 原型统计（使用返回的mask）
        prototype_stats = {
            'per_class': {},      # 每个类的原型统计
            'collapse_metrics': {  # 原型坍缩指标
                'class_distances': {},
                'class_similarities': {},
                'class_variance': {},
            }
        }

        iterator = tqdm(test_loader, desc="Evaluating") if show_progress else test_loader

        with torch.no_grad():
            for task_id, meta_task in enumerate(iterator):

                task_start_time = time.time()

                support_signals, support_labels, query_signals, query_labels,selected_classes = meta_task
                # 移动到设备
                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                # 数据增强：裁剪
                if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
                    processed_support_list = []
                    for signal in support_signals.unbind(0):
                        processed_support_list.append(
                            crop_and_rescale_symmetric(signal, self.crop_ratio_h,self.crop_ratio_l, self.resize)
                        )
                    support_signals = torch.stack(processed_support_list, dim=0)

                    processed_query_list = []
                    for signal in query_signals.unbind(0):
                        processed_query_list.append(
                            crop_and_rescale_symmetric(signal, self.crop_ratio_h,self.crop_ratio_l, self.resize)
                        )
                    query_signals = torch.stack(processed_query_list, dim=0)

                torch.cuda.synchronize()
                task_start_time1 = time.time()
                # 前向传播
                support_v, support_s = self.model(support_signals)
                query_v, query_s = self.model(query_signals)

                torch.cuda.synchronize()
                task_end_time1= time.time()

                original_class_ids_list = selected_classes
                if not isinstance(original_class_ids_list, list):
                    original_class_ids_list = [original_class_ids_list]

                remapped_to_original = {
                    i: original_id
                    for i, original_id in enumerate(original_class_ids_list)
                }

                query_local_indices_np = query_labels.cpu().numpy()
                original_query_labels = np.array([
                    remapped_to_original[idx]
                    for idx in query_local_indices_np
                ])

                remapped_support_labels = support_labels
                remapped_query_labels = query_labels

                torch.cuda.synchronize()
                task_start_time2 = time.time()
                # ========== 关键修复：根据use_multi选择不同的原型计算方式 ==========
                k_shot = len(support_labels) // n_ways

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
                    torch.cuda.synchronize()
                    task_end_time2 = time.time()
                    for i, original_class in enumerate(original_class_ids_list):
                        # GMM 的组件数即为原型数
                        n_components = gmm_params['weights'][i].shape[0]

                        if original_class not in prototype_stats['per_class']:
                            prototype_stats['per_class'][original_class] = {
                                'total_tasks': 0,
                                'active_sum': 0,
                                'initial_prototypes': k_shot
                            }

                        prototype_stats['per_class'][original_class]['total_tasks'] += 1
                        prototype_stats['per_class'][original_class]['active_sum'] += n_components

                    torch.cuda.synchronize()
                    task_start_time3 = time.time()
                    logits = -output['distances']
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == remapped_query_labels).float().mean()
                    total_acc += accuracy.item()

                    torch.cuda.synchronize()
                    task_end_time3 = time.time()

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

                    # ========== 使用返回的mask进行统计 ==========
                    for i, original_class in enumerate(original_class_ids_list):

                        # 统计激活原型数量（从返回的mask中获取）
                        active_count = prototype_mask[i].sum().item()

                        if original_class not in prototype_stats['per_class']:
                            prototype_stats['per_class'][original_class] = {
                                'total_tasks': 0,
                                'active_sum': 0,
                                'initial_prototypes': k_shot
                            }

                        prototype_stats['per_class'][original_class]['total_tasks'] += 1
                        prototype_stats['per_class'][original_class]['active_sum'] += active_count

                        # ========== 计算原型坍缩指标 ==========
                        # 获取该类的原型
                        start_idx = i * k_shot
                        end_idx = start_idx + k_shot
                        class_prototypes = prototypes[start_idx:end_idx]  # [k_shot, feature_dim]

                        # 只考虑激活的原型
                        class_active_mask = prototype_mask[i]
                        if class_active_mask.sum() > 1:  # 至少<configured>个原型
                            active_protos = class_prototypes[class_active_mask]

                            # <configured>. 原型间距离
                            proto_distances = self._compute_pairwise_distances(active_protos)

                            # <configured>. 余弦相似度
                            proto_similarities = self._compute_pairwise_similarities(active_protos)

                            # <configured>. 方差
                            proto_variance = torch.var(active_protos, dim=0).mean().item()

                            # 保存统计
                            metrics = prototype_stats['collapse_metrics']
                            if original_class not in metrics['class_distances']:
                                metrics['class_distances'][original_class] = []
                                metrics['class_similarities'][original_class] = []
                                metrics['class_variance'][original_class] = []

                            metrics['class_distances'][original_class].extend(proto_distances)
                            metrics['class_similarities'][original_class].extend(proto_similarities)
                            metrics['class_variance'][original_class].append(proto_variance)

                else:
                    # ========== 单原型模式：聚合每类的原型 ==========
                    prototypes, precision_matrices = self.compute_gaussian_prototypes(
                        support_v, support_s, remapped_support_labels, n_ways
                    )
                    # 现在 prototypes 的形状是 [n_ways, feature_dim]

                    # 计算距离
                    distances = self.loss_fn.distance_metric(query_v, prototypes, precision_matrices)
                    logits = -distances  # [batch_size, n_ways]

                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == remapped_query_labels).float().mean()
                    total_acc += accuracy.item()


                # 收集特征
                if return_features:
                    collected_features.append(query_v.cpu().numpy())
                    collected_labels.append(original_query_labels.flatten())

                # 收集预测
                if return_predictions:
                    preds_cpu = predictions.cpu().numpy()
                    real_preds = [remapped_to_original[p] for p in preds_cpu]
                    real_targets = [remapped_to_original[t] for t in query_local_indices_np]
                    collected_preds.extend(real_preds)
                    collected_targets.extend(real_targets)

                # 收集任务指标
                if all_result:
                    predictions_cpu = predictions.cpu().numpy()
                    labels_cpu = remapped_query_labels.cpu().numpy()
                    task_acc = accuracy.item()
                    task_f1 = f1_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)
                    task_precision = precision_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)
                    task_recall = recall_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)
                    task_duration_ms = (task_end_time1 - task_start_time1+task_end_time2-task_start_time2+task_end_time3-task_start_time3) * _cfg_require('GPN_V4_GMM.py.GPNTrainer.evaluate.size_or_budget')
                    all_task_metrics.append({
                        'acc': task_acc,
                        'f1_macro': task_f1,
                        'precision_macro': task_precision,
                        'recall_macro': task_recall,
                        'time_ms': task_duration_ms,
                        'task_id': task_id,
                    })
                # ========== 错误统计（现在predictions的索引范围一定正确）==========
                if show_error_stats:
                    errors = (predictions != remapped_query_labels)

                    if errors.any():
                        error_indices = torch.where(errors)[0]

                        for idx in error_indices:
                            true_class = remapped_to_original[remapped_query_labels[idx].item()]
                            pred_idx = predictions[idx].item()

                            # 安全检查：确保索引有效
                            if pred_idx not in remapped_to_original:
                                print(f"Warning: Invalid prediction index {pred_idx}, skipping...")
                                continue

                            pred_class = remapped_to_original[pred_idx]

                            key = (true_class, pred_class)
                            if key not in global_error_stats:
                                global_error_stats[key] = 0
                            global_error_stats[key] += 1

                            if true_class not in class_stats:
                                class_stats[true_class] = {
                                    'total': 0,
                                    'errors': 0,
                                    'misclassified_as': {}
                                }

                            class_stats[true_class]['total'] += 1
                            class_stats[true_class]['errors'] += 1

                            if pred_class not in class_stats[true_class]['misclassified_as']:
                                class_stats[true_class]['misclassified_as'][pred_class] = 0
                            class_stats[true_class]['misclassified_as'][pred_class] += 1

                    correct_indices = torch.where(~errors)[0]
                    for idx in correct_indices:
                        true_class = remapped_to_original[remapped_query_labels[idx].item()]
                        if true_class not in class_stats:
                            class_stats[true_class] = {
                                'total': 0,
                                'errors': 0,
                                'misclassified_as': {}
                            }
                        class_stats[true_class]['total'] += 1

        # 计算平均准确率
        avg_acc = total_acc / len(test_loader)

        # ========== 显示错误统计 ==========
        if show_error_stats and global_error_stats:
            print("\n" + "=" * 60)
            print("错误识别统计")
            print("=" * 60)

            sorted_errors = sorted(
                global_error_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )

            print(f"\n最常见的错误识别模式（真实类别 -> 预测类别）:")
            for (true_cls, pred_cls), count in sorted_errors[:10]:
                print(f"  类别 {true_cls} -> 类别 {pred_cls}: {count} 次")

            print(f"\n各类别准确率:")
            for class_id in sorted(class_stats.keys()):
                stats = class_stats[class_id]
                acc = (stats['total'] - stats['errors']) / stats['total'] if stats['total'] > 0 else 0
                print(f"  类别 {class_id}: {acc:.2%} ({stats['total'] - stats['errors']}/{stats['total']})")

                if stats['misclassified_as']:
                    top_errors = sorted(
                        stats['misclassified_as'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    print(f"    最常误判为: {', '.join([f'{cls}({cnt}次)' for cls, cnt in top_errors])}")

            print("=" * 60)

        # ========== 显示原型统计 ==========
        if self.use_multi and show_error_stats:
            self._print_prototype_collapse_analysis(prototype_stats)

        result_pack = {'avg_acc': avg_acc}

        if return_stats:
            eval_stats = {
                'accuracy': avg_acc,
                'error_stats': global_error_stats,
                'class_stats': class_stats,
                'prototype_stats': prototype_stats
            }
            result_pack['stats'] = eval_stats

        if all_result:
            result_pack['all_task_metrics'] = all_task_metrics
            # 新增：计算平均指标
            result_pack['avg_f1_macro'] = np.mean([m['f1_macro'] for m in all_task_metrics])
            result_pack['avg_precision_macro'] = np.mean([m['precision_macro'] for m in all_task_metrics])
            result_pack['avg_recall_macro'] = np.mean([m['recall_macro'] for m in all_task_metrics])
            result_pack['avg_time_ms'] = np.mean([m['time_ms'] for m in all_task_metrics])

        if return_features:
            result_pack['features'] = np.concatenate(collected_features, axis=0)
            result_pack['feature_labels'] = np.concatenate(collected_labels, axis=0)

        if return_predictions:
            result_pack['all_predictions'] = collected_preds
            result_pack['all_targets'] = collected_targets
        if self.use_multi:
            # 计算原型统计摘要
            prototype_summary = {
                'per_class_avg_active': {},  # 每个类的平均激活原型数
                'global_avg_active': 0,      # 全局平均激活原型数
                'global_avg_initial': 0,     # 全局平均初始原型数
                'merge_ratio': 0,            # 原型合并率
                'raw_stats': prototype_stats # 原始统计数据
            }

            total_active = 0
            total_initial = 0
            total_tasks = 0

            for class_id, stats in prototype_stats['per_class'].items():
                avg_active = stats['active_sum'] / stats['total_tasks']
                prototype_summary['per_class_avg_active'][class_id] = avg_active

                total_active += stats['active_sum']
                total_initial += stats['initial_prototypes'] * stats['total_tasks']
                total_tasks += stats['total_tasks']

            if total_tasks > 0:
                prototype_summary['global_avg_active'] = total_active / total_tasks
                prototype_summary['global_avg_initial'] = total_initial / total_tasks
                prototype_summary['merge_ratio'] = 1 - (total_active / total_initial)

            result_pack['prototype_summary'] = prototype_summary
        if not any([all_result, return_features, return_predictions, return_stats]):
            return avg_acc
        else:
            return result_pack

    # ========== 必需的辅助方法 ==========

    def compute_gaussian_prototypes(self, support_v, support_s, support_labels, n_ways):
        """
        计算高斯原型（单原型模式）

        Args:
            support_v: [n_samples, feature_dim] 特征向量
            support_s: [n_samples, feature_dim] 不确定性特征
            support_labels: [n_samples] 标签
            n_ways: 类别数

        Returns:
            prototypes: [n_ways, feature_dim] 每类的原型
            precision_matrices: [n_ways, feature_dim, feature_dim] 每类的精度矩阵
        """
        feature_dim = support_v.shape[1]
        prototypes = torch.zeros(n_ways, feature_dim, device=self.device)
        precision_matrices = torch.zeros(n_ways, feature_dim, feature_dim, device=self.device)

        for class_idx in range(n_ways):
            # 获取该类的所有样本
            class_mask = (support_labels == class_idx)
            class_features = support_v[class_mask].detach()  # [n_class_samples, feature_dim]
            class_uncertainties = support_s[class_mask]  # [n_class_samples, feature_dim]

            # 计算原型（均值）
            prototypes[class_idx] = class_features.mean(dim=0)

            # 计算精度矩阵
            # 从不确定性特征计算sigma
            sigma = class_uncertainties.mean(dim=0)
            precision_matrices[class_idx] = torch.diag(sigma)

        return prototypes, precision_matrices

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
            class_features = support_v[class_mask].detach()  # [n_class_samples, feature_dim]
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
                    class_features, init_means, max_iter=_cfg_require('GPN_V4_GMM.py.GPNTrainer.compute_gmm_parameters.max_iter')
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
        max_iter = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer._em_algorithm.max_iter', max_iter)
        tol = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer._em_algorithm.tol', tol)
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

    def train_step_batch(self, meta_batch, step_index):
        """
        批处理训练步骤

        Returns:
            avg_loss, avg_acc, merge_summary
        """
        self.model.train()

        support_signals, support_labels, query_signals, query_labels,selected_classes = meta_batch
        batch_size = support_signals.size(0)

        total_loss = 0
        total_acc = 0
        batch_merge_info = []

        self.optimizer_model.zero_grad()

        for i in range(batch_size):
            loss, acc, output = self._single_task_forward(
                support_signals[i], support_labels[i],
                query_signals[i], query_labels[i]
            )

            (loss / batch_size).backward()

            total_loss += loss.item()
            total_acc += acc

            if 'merge_info' in output:
                batch_merge_info.append(output['merge_info'])

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('GPN_V4_GMM.py.GPNTrainer.train_step_batch.max_norm'))
        self.optimizer_model.step()

        merge_summary = None
        if batch_merge_info:
            merge_summary = {
                'total_merges': sum(info['merge_count'] for info in batch_merge_info),
                'avg_active_prototypes': sum(info['total_active'] for info in batch_merge_info) / len(batch_merge_info)
            }

        return total_loss / batch_size, total_acc / batch_size, merge_summary


    def train(self, train_loader, test_loader, epochs=None,
          n_ways=None, save_path=None):
        """
        改进的元学习训练循环（适配动态原型）

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            n_ways: 类别数（用于测试）
            save_path: 模型保存路径
        """
        epochs = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.train.epochs', epochs)
        n_ways = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.train.n_ways', n_ways)
        save_path = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.train.save_path', save_path)
        self.model.train()

        scheduler_model = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model,
            step_size=_cfg_require('GPN_V4_GMM.py.GPNTrainer.train.step_size'),
            gamma=_cfg_require('GPN_V4_GMM.py.GPNTrainer.train.gamma')
        )

        print(f"开始元学习训练: {epochs} epochs")
        print(f"每个 epoch 有 {len(train_loader)} 个批次")
        print(f"实际 batch_size: {train_loader.batch_size}")
        print(f"每个 epoch 训练 {len(train_loader) * train_loader.batch_size} 个任务")

        if self.use_multi:
            print(f"多原型模式: 初始原型=支持集大小, 合并阈值={self.loss_fn.merge_threshold}")

        best_acc = 0
        early_stop = 0

        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            processed_batches = 0
            epoch_merge_count = 0  # 新增：统计epoch内的合并次数

            self.current_epoch+=1

            # 使用 tqdm 显示进度
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, meta_batch in enumerate(pbar):
                    # 批处理训练步骤（返回<configured>个值）
                    loss, acc, merge_info = self.train_step_batch(meta_batch, batch_idx)

                    total_loss += loss
                    total_acc += acc
                    processed_batches += 1

                    # 新增：收集合并信息
                    if merge_info is not None:
                        epoch_merge_count += merge_info['total_merges']

                    # 更新进度条
                    postfix_dict = {
                        'loss': f'{loss:.4f}',
                        'accuracy': f'{acc:.4f}',
                        'lr': f'{self.optimizer_model.param_groups[0]["lr"]:.6f}'
                    }

                    # 新增：显示原型信息
                    if self.use_multi and merge_info is not None:
                        postfix_dict['active_proto'] = f"{merge_info['avg_active_prototypes']:.1f}"

                    pbar.set_postfix(postfix_dict)

            # 学习率调度
            scheduler_model.step()

            # 计算平均指标
            avg_loss = total_loss / processed_batches
            avg_acc = total_acc / processed_batches

            print(f"Epoch {epoch+1} 训练完成: avg_loss: {avg_loss:.4f}, avg_acc: {avg_acc:.4f}")
            # 定期评估
            if (epoch + 1) % 5 == 0:
                print(f"\n开始测试评估...")
                test_acc, eval_stats = self.evaluate(
                    test_loader=test_loader,
                    n_ways=n_ways,
                    show_progress=True,
                    show_error_stats=True,
                    return_stats=True  # 新增：返回详细统计
                )
                print(f"测试准确率: {test_acc:.4f}")

                # 新增：显示原型统计
                if self.use_multi and 'prototype_stats' in eval_stats:
                    self._print_prototype_evaluation_summary(eval_stats['prototype_stats'])

                if test_acc > best_acc:
                    best_acc = test_acc
                    early_stop = 0
                    if self.save_middle:
                        self.save_model(save_path)#中途保存

                else:
                    early_stop += 1
                    print(f"未提升（连续 {early_stop} 次）")
                    if early_stop >= 2:
                        print(f"\n早停触发，训练结束")
                        break

        # 保存最终模型
        self.save_model(save_path)

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"\n训练完成！最佳测试准确率: {best_acc:.4f}")

    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """完整的评估协议（符合论文）"""
        n_trials = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('GPN_V4_GMM.py.GPNTrainer.full_evaluation.q_query', q_query)
        results = {}

        for n_way in [8,7,6,5,4,3]:
            for k_shot in [10, 5, 1]:
                print(f"\n{'='*50}")
                print(f"Evaluating {n_way}-way {k_shot}-shot")
                print(f"{'='*50}")

                accuracies = []

                # 使用tqdm显示trials进度，position参数避免嵌套冲突
                pbar = tqdm(range(n_trials),
                           desc=f"{n_way}w{k_shot}s",
                           position=0,
                           leave=True)
                i=0#该框架的warm-up策略在full-evaluate这里实现（本处因为每轮evaluate仅一个任务）
                avg_time_ms=0

                for trial in pbar:
                    meta_test = MetaDataset(
                        test_dataset, n_way, k_shot, q_query=q_query,
                        num_tasks=_cfg_require('GPN_V4_GMM.py.GPNTrainer.full_evaluation.num_tasks'),seed=random.randint(0, _cfg_require('GPN_V4_GMM.py.GPNTrainer.full_evaluation.size_or_budget'))
                    )
                    loader = DataLoader(meta_test, batch_size=_cfg_require('GPN_V4_GMM.py.GPNTrainer.full_evaluation.batch_size'))

                    # 关闭内层进度条
                    result_pack=self.evaluate(loader, n_way, show_progress=False, show_error_stats=False,all_result=True)
                    if i!=0:
                        avg_time_ms+=result_pack['avg_time_ms']
                    i+=1
                    acc=result_pack['avg_acc']
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

                avg_time_ms=avg_time_ms/(i-1)
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                results[f'{n_way}w{k_shot}s'] = (mean_acc, std_acc)
                print('avg_time_ms',avg_time_ms)
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


    def _print_prototype_evaluation_summary(self, prototype_stats):
        """
        打印评估时的原型统计摘要

        Args:
            prototype_stats: 原型统计信息字典
        """
        print("\n" + "=" * 60)
        print("原型统计摘要")
        print("=" * 60)

        per_class = prototype_stats.get('per_class', {})

        if per_class:
            print(f"\n各类别平均激活原型数:")
            for class_id in sorted(per_class.keys()):
                stats = per_class[class_id]
                avg_active = stats['active_sum'] / stats['total_tasks']
                initial = stats['initial_prototypes']
                retention = (avg_active / initial * 100) if initial > 0 else 0

                print(f'  类别 {class_id}: {avg_active:.2f}/{initial} (保留率: {retention:.1f}%)')

        print("=" * 60)


    def _print_prototype_collapse_analysis(self, prototype_stats):
        """
        打印原型坍缩分析

        Args:
            prototype_stats: 原型统计信息字典
        """
        print("\n" + "=" * 60)
        print("原型坍缩分析")
        print("=" * 60)

        metrics = prototype_stats.get('collapse_metrics', {})
        per_class = prototype_stats.get('per_class', {})

        if not metrics['class_distances']:
            print("没有足够的数据进行坍缩分析")
            print("=" * 60)
            return

        print(f"\n{'类别':<6} {'平均距离':<12} {'平均相似度':<12} {'方差':<10} {'激活原型':<10}")
        print("-" * 60)

        for class_id in sorted(metrics['class_distances'].keys()):
            distances = metrics['class_distances'][class_id]
            similarities = metrics['class_similarities'][class_id]
            variances = metrics['class_variance'][class_id]

            avg_dist = np.mean(distances) if distances else 0
            avg_sim = np.mean(similarities) if similarities else 0
            avg_var = np.mean(variances) if variances else 0

            # 获取平均激活原型数
            avg_active = "N/A"
            if class_id in per_class:
                stats = per_class[class_id]
                avg_active = f"{stats['active_sum'] / stats['total_tasks']:.1f}"

            # 判断是否坍缩（根据阈值）
            is_collapsed = avg_dist < 0.5 or avg_sim > 0.9
            marker = "⚠️" if is_collapsed else ""

            print(f'{class_id:<6} {avg_dist:<12.4f} {avg_sim:<12.4f} {avg_var:<10.4f} {avg_active:<10} {marker}')

        print("\n说明:")
        print("  - 平均距离: 原型间的平均欧氏距离（越大越好）")
        print("  - 平均相似度: 原型间的平均余弦相似度（越小越好）")
        print("  - 方差: 原型特征的方差（越大说明越分散）")
        print("  - ⚠️ 标记: 可能存在原型坍缩")
        print("=" * 60)


    def _compute_pairwise_distances(self, prototypes):
        """
        计算原型间的成对欧氏距离

        Args:
            prototypes: [n_prototypes, feature_dim]

        Returns:
            distances: 所有成对距离的列表
        """
        n = prototypes.shape[0]
        distances = []

        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.norm(prototypes[i] - prototypes[j], p=2).item()
                distances.append(dist)

        return distances


    def _compute_pairwise_similarities(self, prototypes):
        """
        计算原型间的余弦相似度

        Args:
            prototypes: [n_prototypes, feature_dim]

        Returns:
            similarities: 所有成对相似度的列表
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


    def get_training_statistics(self):
        """
        获取训练统计信息

        Returns:
            dict: 包含全局合并统计等信息
        """
        stats = self.loss_fn.get_global_statistics()
        stats['current_training_epoch'] = self.current_epoch
        return stats

    def reset_training_statistics(self):
        """
        重置训练统计信息（开始新的训练session时调用）
        """
        self.loss_fn.reset_statistics()
        self.current_epoch = 0
        print("训练统计信息已重置")


class DecoupledAxisAttention(nn.Module):
    """
    解耦的轴注意力 - 针对RF信号时频特性

    核心思想：
    - 频率轴：强注意力（物理相关）
    - 时间轴：弱注意力（随机性高）
    """
    def __init__(self, channels, freq_time_ratio=None):
        freq_time_ratio = _cfg_resolve('GPN_V4_GMM.py.DecoupledAxisAttention.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()
        self.freq_weight = freq_time_ratio
        self.time_weight = 1

        # 频率轴注意力（沿时间维度池化，保留频率信息）
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),  # [B, C, H, <configured>]
            nn.Conv2d(channels, channels, kernel_size=_cfg_require('GPN_V4_GMM.py.DecoupledAxisAttention.__init__.kernel_size'), padding=_cfg_require('GPN_V4_GMM.py.DecoupledAxisAttention.__init__.padding')),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # 时间轴注意力（沿频率维度池化，保留时间信息）
        self.time_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # [B, C, <configured>, W]
            nn.Conv2d(channels, channels, kernel_size=_cfg_require('GPN_V4_GMM.py.DecoupledAxisAttention.__init__.kernel_size__2'), padding=_cfg_require('GPN_V4_GMM.py.DecoupledAxisAttention.__init__.padding__2')),
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


class DropPath(nn.Module):
    """DropPath正则化"""
    def __init__(self, drop_prob=None):
        drop_prob = _cfg_resolve('GPN_V4_GMM.py.DropPath.__init__.drop_prob', drop_prob)
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


class CBAM(nn.Module):
    def __init__(self, channels, reduction=None):
        reduction = _cfg_resolve('GPN_V4_GMM.py.CBAM.__init__.reduction', reduction)
        super().__init__()
        # 两个子模块：通道注意力 + 空间注意力
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 顺序执行：先通道 → 后空间
        x = self.channel_attention(x)  # 通道重加权
        x = self.spatial_attention(x)  # 空间重加权
        return x
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=None):
        reduction = _cfg_resolve('GPN_V4_GMM.py.ChannelAttention.__init__.reduction', reduction)
        super().__init__()
        # 使用平均池化和最大池化的双路径
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, _cfg_require('GPN_V4_GMM.py.ChannelAttention.__init__.Conv2d_arg2'), bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, _cfg_require('GPN_V4_GMM.py.ChannelAttention.__init__.Conv2d_arg2__2'), bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入: [B, C, H, W]

        # 双路径池化
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, <configured>, <configured>]
        max_out = self.mlp(self.max_pool(x))  # [B, C, <configured>, <configured>]

        # 融合并生成通道权重
        channel_weights = self.sigmoid(avg_out + max_out)  # [B, C, <configured>, <configured>]

        return x * channel_weights  # 通道级乘法
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=None):
        kernel_size = _cfg_resolve('GPN_V4_GMM.py.SpatialAttention.__init__.kernel_size', kernel_size)
        super().__init__()
        # 使用通道维度的平均和最大池化
        self.conv = nn.Conv2d(_cfg_require('GPN_V4_GMM.py.SpatialAttention.__init__.Conv2d_arg0'), _cfg_require('GPN_V4_GMM.py.SpatialAttention.__init__.Conv2d_arg1'), kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入: [B, C, H, W]

        # 沿通道维度聚合
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, <configured>, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B, <configured>, H, W]

        # 拼接通道统计信息
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, <configured>, H, W]

        # 生成空间权重图
        spatial_weights = self.sigmoid(self.conv(spatial_input))  # [B, <configured>, H, W]

        return x * spatial_weights  # 空间级乘法
class RelativePositionAttention2D(nn.Module):
    """
    相对位置注意力 - 2D版本(可插入ResNeXt)
    将序列展平后应用相对位置注意力,适用于时频图等2D数据
    """
    def __init__(self, channels, reduction=None, max_relative_position=None):
        '\n        Args:\n            channels: 输入通道数\n            reduction: 降维比例(用于减少计算量)\n            max_relative_position: 最大相对位置(由外部配置提供)\n        '
        reduction = _cfg_resolve('GPN_V4_GMM.py.RelativePositionAttention2D.__init__.reduction', reduction)
        max_relative_position = _cfg_resolve('GPN_V4_GMM.py.RelativePositionAttention2D.__init__.max_relative_position', max_relative_position)
        super().__init__()
        self.channels = channels
        self.max_relative_position = max_relative_position

        # 降维以减少计算量
        inner_dim = max(channels // reduction, 32)
        self.heads = _cfg_require('GPN_V4_GMM.py.RelativePositionAttention2D.__init__.heads')  # 多头注意力
        self.head_dim = inner_dim // self.heads

        # 相对位置嵌入表
        self.relative_position_bias = nn.Parameter(
            torch.randn(2 * max_relative_position - 1, self.heads)
        )

        # 通道降维
        self.reduce = nn.Conv2d(channels, inner_dim, _cfg_require('GPN_V4_GMM.py.RelativePositionAttention2D.__init__.Conv2d_arg2'))

        # QKV投影
        self.to_qkv = nn.Conv2d(inner_dim, inner_dim * 3, _cfg_require('GPN_V4_GMM.py.RelativePositionAttention2D.__init__.Conv2d_arg2__2'), bias=False)

        # 输出投影(恢复通道数)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, _cfg_require('GPN_V4_GMM.py.RelativePositionAttention2D.__init__.Conv2d_arg2__3')),
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
        reduction = _cfg_resolve('GPN_V4_GMM.py.RFRelativePositionAttention2D.__init__.reduction', reduction)
        heads = _cfg_resolve('GPN_V4_GMM.py.RFRelativePositionAttention2D.__init__.heads', heads)
        super().__init__()
        self.channels = channels
        self.heads = heads

        # 降维以减少计算量
        inner_dim = max(channels // reduction, 32)
        self.inner_dim = inner_dim

        # 动态相对位置编码(支持不同尺寸的输入)
        self.max_freq = _cfg_require('GPN_V4_GMM.py.RFRelativePositionAttention2D.__init__.size_or_budget')  # 最大频率bins
        self.max_time = _cfg_require('GPN_V4_GMM.py.RFRelativePositionAttention2D.__init__.size_or_budget__2')  # 最大时间steps

        self.freq_relative_bias = nn.Parameter(
            torch.randn(2 * self.max_freq - 1, heads)
        )
        self.time_relative_bias = nn.Parameter(
            torch.randn(2 * self.max_time - 1, heads)
        )

        # 通道降维
        self.reduce = nn.Conv2d(channels, inner_dim, _cfg_require('GPN_V4_GMM.py.RFRelativePositionAttention2D.__init__.Conv2d_arg2'))

        # QKV投影
        self.to_qkv = nn.Conv2d(inner_dim, inner_dim * 3, _cfg_require('GPN_V4_GMM.py.RFRelativePositionAttention2D.__init__.Conv2d_arg2__2'), bias=False)

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, _cfg_require('GPN_V4_GMM.py.RFRelativePositionAttention2D.__init__.Conv2d_arg2__3')),
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
        reduction = _cfg_resolve('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.reduction', reduction)
        heads = _cfg_resolve('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.heads', heads)
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
        max_freq = freq_bins if freq_bins else _cfg_require('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.size_or_budget')
        max_time = time_steps if time_steps else _cfg_require('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.size_or_budget__2')

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
        self.reduce = nn.Conv2d(channels, inner_dim, _cfg_require('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2'))

        self.to_q = nn.Conv2d(inner_dim, inner_dim, _cfg_require('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2__2'), bias=False)
        self.to_k = nn.Conv2d(inner_dim, inner_dim, _cfg_require('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2__3'), bias=False)
        self.to_v = nn.Conv2d(inner_dim, inner_dim, _cfg_require('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2__4'), bias=False)

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, _cfg_require('GPN_V4_GMM.py.OptimizedRFRelativeAttention.__init__.Conv2d_arg2__5')),
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
        reduction = _cfg_resolve('GPN_V4_GMM.py.FlashStyleRFAttention.__init__.reduction', reduction)
        heads = _cfg_resolve('GPN_V4_GMM.py.FlashStyleRFAttention.__init__.heads', heads)
        freq_chunk_size = _cfg_resolve('GPN_V4_GMM.py.FlashStyleRFAttention.__init__.freq_chunk_size', freq_chunk_size)
        time_chunk_size = _cfg_resolve('GPN_V4_GMM.py.FlashStyleRFAttention.__init__.time_chunk_size', time_chunk_size)
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

        self.reduce = nn.Conv2d(channels, inner_dim, _cfg_require('GPN_V4_GMM.py.FlashStyleRFAttention.__init__.Conv2d_arg2'))
        self.to_qkv = nn.Conv2d(inner_dim, inner_dim * 3, _cfg_require('GPN_V4_GMM.py.FlashStyleRFAttention.__init__.Conv2d_arg2__2'), bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, _cfg_require('GPN_V4_GMM.py.FlashStyleRFAttention.__init__.Conv2d_arg2__3')),
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
                 freq_time_ratio=None,):
        stride = _cfg_resolve('GPN_V4_GMM.py.ResNeXtBlock.__init__.stride', stride)
        cardinality = _cfg_resolve('GPN_V4_GMM.py.ResNeXtBlock.__init__.cardinality', cardinality)
        attention_type = _cfg_resolve('GPN_V4_GMM.py.ResNeXtBlock.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V4_GMM.py.ResNeXtBlock.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V4_GMM.py.ResNeXtBlock.__init__.freq_time_ratio', freq_time_ratio)
        super().__init__()

        mid_channels = out_channels

        # 1x1 扩展卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=_cfg_require('GPN_V4_GMM.py.ResNeXtBlock.__init__.kernel_size'), bias=False,)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 分组卷积
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=_cfg_require('GPN_V4_GMM.py.ResNeXtBlock.__init__.kernel_size__2'), stride=stride,
                              padding=_cfg_require('GPN_V4_GMM.py.ResNeXtBlock.__init__.padding'), groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 压缩卷积
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                              kernel_size=_cfg_require('GPN_V4_GMM.py.ResNeXtBlock.__init__.kernel_size__3'), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.SiLU(inplace=True)

        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=_cfg_require('GPN_V4_GMM.py.ResNeXtBlock.__init__.kernel_size__4'), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 注意力机制选择
        if attention_type == 's':
            self.attention = SEModule(out_channels, reduction=reduction)
        elif attention_type == 'c':
            # 新增:CBAM注意力
            self.attention = CBAM(out_channels, reduction=reduction)
        elif attention_type == 'd':
            self.attention = DecoupledAxisAttention(out_channels, freq_time_ratio)
        elif attention_type == 'f':
            self.attention = FrequencyPriorityCA(out_channels, reduction, freq_time_ratio)
        elif attention_type == 'r':
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

        # 注意力模块(在Add之前应用)
        if self.attention is not None:
            out = self.attention(out)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

class GPN_Optimized(nn.Module):
    "\n    优化的GPN模型 - 完全兼容现有训练器\n    \n    架构特点：\n    - Stage <configured>-<configured>: 可选RepVGG（浅层，训练稳定）\n    - Stage <configured>-<configured>: 可选ConvNeXt（深层，大感受野）\n    - 全局：统一的注意力机制（SE / decoupled / freq_priority）\n    - 输出：v (embedding), s (precision) - 与原模型一致\n    \n    参数说明：\n        attention_type: 注意力类型\n            - 'se': 标准SE模块（默认）\n            - 'decoupled': 解耦时频注意力\n            - 'freq_priority': 频率优先CA\n            - None: 不使用注意力\n        reduction: SE模块的reduction ratio（由外部配置提供）\n    "
    def __init__(self,
                 attention_type=None,
                 reduction=None,
                 freq_time_ratio=None,
                 ca_input_channels=None):
        attention_type = _cfg_resolve('GPN_V4_GMM.py.GPN_Optimized.__init__.attention_type', attention_type)
        reduction = _cfg_resolve('GPN_V4_GMM.py.GPN_Optimized.__init__.reduction', reduction)
        freq_time_ratio = _cfg_resolve('GPN_V4_GMM.py.GPN_Optimized.__init__.freq_time_ratio', freq_time_ratio)
        ca_input_channels = _cfg_resolve('GPN_V4_GMM.py.GPN_Optimized.__init__.ca_input_channels', ca_input_channels)
        super().__init__()

        self.attention_type = attention_type
        self.use_ca = _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.use_ca')#留待对比测试？
        self.ca_position = 'input'

        if self.use_ca and self.ca_position == 'input':
            # 先用小卷积提升通道数（保持频率-时间结构）
            self.input_proj = nn.Sequential(
                nn.Conv2d(_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.Conv2d_arg0__2'), ca_input_channels, kernel_size=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.kernel_size__8'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__12'),
                         padding=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.padding__4'), bias=False),  # <configured>×<configured>小卷积，局部混合
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
            self.conv1 = nn.Conv2d(ca_input_channels, _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.Conv2d_arg1'), kernel_size=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.kernel_size__6'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__10'),
                                  padding=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.padding__2'), bias=False)
        else:
            # 原始Stem
            self.conv1 = nn.Conv2d(_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.Conv2d_arg0'), _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.Conv2d_arg1__2'), kernel_size=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.kernel_size__7'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__11'),
                                  padding=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.padding__3'), bias=False)

        self.bn1 = nn.BatchNorm2d(_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.BatchNorm2d_arg0'))
        self.relu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.kernel_size'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride'), padding=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.padding'))
        # 输出: [B, <configured>, <configured>, <configured>]


        self.block1 = ResNeXtBlock(
            _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.ResNeXtBlock_arg0'), _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.ResNeXtBlock_arg1'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__2'), cardinality=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.cardinality'),
            attention_type=attention_type[0],
            reduction=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.reduction__2'),
            freq_time_ratio=freq_time_ratio,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.kernel_size__2'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__3'))
        # 输出: [B, <configured>, <configured>, <configured>]


        self.block2 = ResNeXtBlock(
            _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__2'), _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__2'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__4'), cardinality=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.cardinality__2'),
            attention_type=attention_type[1],
            reduction=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.reduction__3'),
            freq_time_ratio=freq_time_ratio
        )
        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.kernel_size__3'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__5'))
        # 输出: [B, <configured>, <configured>, <configured>]


        self.block3 = ResNeXtBlock(
            _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__3'), _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__3'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__6'), cardinality=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.cardinality__3'),
            attention_type=attention_type[2],
            reduction=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.reduction__4'),
            freq_time_ratio=freq_time_ratio
        )
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.kernel_size__4'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__7'))
        # 输出: [B, <configured>, <configured>, <configured>]

        self.block4 = ResNeXtBlock(
            _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.ResNeXtBlock_arg0__4'), _cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.ResNeXtBlock_arg1__4'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__8'), cardinality=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.cardinality__4'),
            attention_type=attention_type[3],
            reduction=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.reduction__5'),
            freq_time_ratio=freq_time_ratio
        )
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.kernel_size__5'), stride=_cfg_require('GPN_V4_GMM.py.GPN_Optimized.__init__.stride__9'))
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
        v_features = x[:, :_cfg_require('GPN_V4_GMM.py.GPN_Optimized.forward.size_or_budget'), :, :]  # [B, <configured>, <configured>, <configured>]
        s_features = x[:, _cfg_require('GPN_V4_GMM.py.GPN_Optimized.forward.size_or_budget__2'):, :, :]  # [B, <configured>, <configured>, <configured>]

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
