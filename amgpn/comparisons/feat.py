from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
# Derived by source editing from the AMGPN trainer; no Python subclass relationship.
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from pathlib import Path
from tqdm import tqdm
from amgpn.data.episodes import *
from amgpn.losses.adaptive import MultiPrototypeGPNLoss
from amgpn.data.feature_maps import *
import gc
from amgpn.data.preprocessing import crop_and_rescale_symmetric

#新增FEATAdaptor
class FEATAdaptor(nn.Module):
    """
    FEAT 适配层
    保证：Query 只能单向关注 Support，Query 之间互不影响，且 Query 绝不改变 Support 的特征
    """
    def __init__(self, feature_dim=None, n_heads=None):
        feature_dim = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.FEATAdaptor.__init__.feature_dim', feature_dim)
        n_heads = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.FEATAdaptor.__init__.n_heads', n_heads)
        super().__init__()
        # 支持集内部交互的自注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=n_heads, dim_feedforward=feature_dim*2, batch_first=True
        )
        self.support_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=_cfg_require('GPN_V5_adaMulti_clean_FEAT.py.FEATAdaptor.__init__.num_layers'))

        # 查询集单向关注支持集的交叉注意力投影
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.scale = 1.0 / (feature_dim ** 0.5)

    def forward(self, support_v, query_v):
        # support_v: [S, <configured>], query_v: [Q, <configured>]

        # <configured>. 支持集独自进行自注意力强化，完全不看 Query
        s_input = support_v.unsqueeze(0) # [<configured>, S, <configured>]
        adapted_support = self.support_self_attn(s_input).squeeze(0) # [S, <configured>]

        # <configured>. 查询集单向对强化后的支持集做 Cross-Attention
        # Query 映射自原始 query_v，Key/Value 映射自 adapted_support
        Q_mat = self.query_transform(query_v)        # [Q, <configured>]
        K_mat = self.key_transform(adapted_support)  # [S, <configured>]
        V_mat = self.value_transform(adapted_support)# [S, <configured>]

        # 计算注意力权重 [Q, S]
        attn_scores = torch.mm(Q_mat, K_mat.t()) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 查询集融入支持集上下文后的特征
        adapted_query = torch.mm(attn_weights, V_mat) # [Q, <configured>]

        # 残差连接
        adapted_query = query_v + adapted_query

        # 返回的 adapted_support 完全没有受到 query_v 的任何逆向影响
        return adapted_support, adapted_query

    def query_transform(self, x): return self.q_proj(x)
    def key_transform(self, x): return self.k_proj(x)
    def value_transform(self, x): return self.v_proj(x)

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
    GPN模型训练器 - 适配动态原型损失函数

    支持集的每个样本直接作为初始原型
    由MultiPrototypeGPNLoss自动推断和合并原型
    """
    def __init__(self, model, device,use_Mdistance=None, use_multi=None,save_middle=None,mlr=None,
                 crop_ratio_h=None,crop_ratio_l=None, resize=None,
                 # 新增：原型合并相关参数
                 merge_threshold=None,
                 FEAT_heads=None):
        """
        Args:
            model: GPN骨干网络
            device: 训练设备
            use_multi: 是否使用多原型模式
            save_middle: 是否中途保存最优模型
            mlr: 模型学习率
            crop_ratio: 裁剪比例
            resize: 是否resize
            merge_threshold: 原型合并的L2距离阈值
        """
        use_Mdistance = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.use_Mdistance', use_Mdistance)
        use_multi = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.use_multi', use_multi)
        save_middle = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.save_middle', save_middle)
        mlr = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.mlr', mlr)
        crop_ratio_h = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.crop_ratio_h', crop_ratio_h)
        crop_ratio_l = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.crop_ratio_l', crop_ratio_l)
        resize = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.resize', resize)
        merge_threshold = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.merge_threshold', merge_threshold)
        FEAT_heads = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.FEAT_heads', FEAT_heads)
        self.device = device
        self.crop_ratio_h = crop_ratio_h
        self.crop_ratio_l=crop_ratio_l
        self.resize = resize

        # 初始化模型
        self.model = model
        self.model.to(device)
        self.use_multi = use_multi
        self.use_Mdistance=use_Mdistance
        self.save_middle=save_middle
        self.feat_adaptor=FEATAdaptor(feature_dim=_cfg_require('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.feature_dim'),n_heads=FEAT_heads)
        self.feat_adaptor.to(device)

        # 初始化损失函数（支持动态原型合并）
        self.loss_fn = MultiPrototypeGPNLoss(
            use_multi=self.use_multi,
            use_Mdistance=self.use_Mdistance,
            merge_threshold=merge_threshold,
        )
        self.loss_fn.to(device)

        self.model_params = list(self.model.parameters())
        self.feat_params = list(self.feat_adaptor.parameters())

        self.trainable_params = (
            self.model_params
            + self.feat_params
        )
        self.mlr = mlr
        self.optimizer_model = torch.optim.Adam(self.trainable_params, lr=self.mlr)

        # 训练统计
        self.current_epoch = 0
        self.merge_statistics = []

        if self.use_multi:
            print(f"多原型模式启用: 初始原型=支持集样本数, 合并阈值={merge_threshold}")

    def save_model(self, save_path=None):
        """
        保存模型和损失函数的所有可学习参数
        """
        save_path = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.save_model.save_path', save_path)
        filtered_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if not ("total_ops" in k or "total_params" in k)
        }
        checkpoint = {
            'model_state_dict': filtered_state_dict,
            'current_epoch': self.current_epoch,
        }

        if hasattr(self, 'feat_adaptor'):
            checkpoint['feat_adaptor_state_dict'] = self.feat_adaptor.state_dict()
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
        if 'feat_adaptor_state_dict' in checkpoint:
            self.feat_adaptor.load_state_dict(checkpoint['feat_adaptor_state_dict'])


        if 'loss_fn_state_dict' in checkpoint:

            if hasattr(self, 'loss_fn') and hasattr(self.loss_fn, 'load_state_dict'):

                self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
                print(f"Loss function parameters loaded")
            else:
                print("Warning: Checkpoint contains loss_fn but current trainer doesn't have compatible loss_fn")
        else:
            print("Warning: Checkpoint doesn't contain loss_fn parameters")
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

    def _single_task_forward(self, support_signals, support_labels, query_signals, query_labels):
        """
        单个任务的前向传播（任务内动态原型合并）

        区分use_multi的两种情况
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

        # 前向传播
        support_v, support_s = self.model(support_signals)
        query_v, query_s = self.model(query_signals)
        #FEAT发挥作用
        support_v, query_v = self.feat_adaptor(support_v, query_v)

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

        # ========== 根据use_multi选择不同的原型准备方式 ==========
        if self.use_multi:
            # 多原型模式：使用所有support样本作为原型
            prototypes = support_v  # [n_ways * k_shot, feature_dim]
            precision_matrices = self.compute_precision_matrices_from_support(support_s)

            # 调用损失函数（任务内合并）
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
            # prototypes: [n_ways, feature_dim]

            # 直接计算距离（不需要通过loss_fn的forward）
            distances = self.loss_fn.distance_metric(query_v, prototypes, precision_matrices)
            logits = -distances  # [batch_size, n_ways]
            probabilities = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, remapped_query_labels)

            # 构造与多原型模式兼容的输出格式
            predictions = torch.argmax(probabilities, dim=1)
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
                 show_error_stats=True, return_stats=False):
        '\n        评估模型性能（使用任务级别的prototype_mask统计）\n\n        关键改进：\n        <configured>. 从loss函数的返回值中获取prototype_mask\n        <configured>. 每个任务的mask独立统计\n        <configured>. 不依赖持久化的状态\n        <configured>. 修复了use_multi=False时的索引越界问题\n\n        Args:\n            test_loader: 数据加载器\n            n_ways: 类别数\n            show_progress: 是否显示进度条\n            show_error_stats: 是否显示错误统计信息\n            return_stats: 是否返回详细统计信息\n\n        Returns:\n            avg_acc: 平均准确率\n            eval_stats: 评估统计信息（如果return_stats=True）\n        '
        self.model.eval()
        total_acc = 0

        if hasattr(self, "feat_adaptor"):
            self.feat_adaptor.eval()

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
                support_signals, support_labels, query_signals, query_labels, *extra = meta_task

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

                # 前向传播
                support_v, support_s = self.model(support_signals)
                query_v, query_s = self.model(query_signals)

                #FEAT发挥作用
                support_v, query_v = self.feat_adaptor(support_v, query_v)

                # 重新映射标签
                unique_labels = torch.unique(support_labels)
                label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
                remapped_to_original = {i: label.item() for i, label in enumerate(unique_labels)}

                remapped_support_labels = torch.tensor(
                    [label_mapping[label.item()] for label in support_labels],
                    device=self.device
                )
                remapped_query_labels = torch.tensor(
                    [label_mapping[label.item()] for label in query_labels],
                    device=self.device
                )
                original_query_labels = query_labels.cpu().numpy()

                # ========== 关键修复：根据use_multi选择不同的原型计算方式 ==========
                if self.use_multi:
                    # 多原型模式：使用所有support样本作为原型
                    k_shot = len(support_labels) // n_ways
                    prototypes = support_v  # [n_ways * k_shot, feature_dim]
                    precision_matrices = self.compute_precision_matrices_from_support(support_s)

                    # 调用损失函数（返回包含mask的字典）
                    output = self.loss_fn(
                        query_v,
                        prototypes,
                        precision_matrices,
                        remapped_query_labels,
                        epoch=self.current_epoch
                    )

                    # 从返回值中获取prototype_mask
                    prototype_mask = output['prototype_mask']  # [n_ways, k_shot]

                    # 提取结果
                    logits = -output['distances']
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == remapped_query_labels).float().mean()
                    total_acc += accuracy.item()

                    # ========== 使用返回的mask进行统计 ==========
                    for i, class_label in enumerate(unique_labels):
                        original_class = class_label.item()

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

        # 返回结果
        if return_stats:
            eval_stats = {
                'accuracy': avg_acc,
                'error_stats': global_error_stats,
                'class_stats': class_stats,
                'prototype_stats': prototype_stats
            }
            return avg_acc, eval_stats
        else:
            return avg_acc


    # ========== 必需的辅助方法 ==========

    def compute_gaussian_prototypes(self, support_v, support_s, support_labels, n_ways):
        """
        计算高斯原型（单原型模式）（简单平均）

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
            class_features = support_v[class_mask]  # [n_class_samples, feature_dim]
            class_uncertainties = support_s[class_mask]  # [n_class_samples, feature_dim]

            # 计算原型（均值）
            prototypes[class_idx] = class_features.mean(dim=0)

            # 计算精度矩阵
            # 从不确定性特征计算sigma
            sigma = class_uncertainties.mean(dim=0)
            precision_matrices[class_idx] = torch.diag(sigma)

        return prototypes, precision_matrices


    def train_step_batch(self, meta_batch, step_index):
        """
        批处理训练步骤

        Returns:
            avg_loss, avg_acc, merge_summary
        """


        self.model.train()
        self.feat_adaptor.train()
        support_signals, support_labels, query_signals, query_labels, *extra = meta_batch
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

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.train_step_batch.max_norm'))
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
        epochs = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.train.epochs', epochs)
        n_ways = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.train.n_ways', n_ways)
        save_path = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.train.save_path', save_path)
        self.model.train()
        self.feat_adaptor.train()

        scheduler_model = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model,
            step_size=_cfg_require('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.train.step_size'),
            gamma=_cfg_require('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.train.gamma')
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
                    if early_stop >= 4:
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
        n_trials = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.full_evaluation.q_query', q_query)
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

                for trial in pbar:
                    meta_test = MetaDataset(
                        test_dataset, n_way, k_shot, q_query=q_query,
                        num_tasks=_cfg_require('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.full_evaluation.num_tasks'),seed=random.randint(0, _cfg_require('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.full_evaluation.size_or_budget'))
                    )
                    loader = DataLoader(meta_test, batch_size=_cfg_require('GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.full_evaluation.batch_size'))

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

# ==================== SE模块 ====================


# ============ 更新ResNeXtBlock支持新的注意力机制 ============

from amgpn.models.backbone import SEModule, ResNeXtBlock, configured_backbone
GPN_Optimized = configured_backbone('GPN_V5_adaMulti_clean_FEAT.py')
