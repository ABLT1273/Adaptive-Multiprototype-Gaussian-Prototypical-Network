'\nAMGPN 综合评估模块\n==================\n用于生成论文所需的所有实验数据和图表\n\n图表清单:\n- Table <configured>: Main Results (扩充版，含F1-score)\n- Table <configured>: Initialization Strategy Comparison\n- Table <configured>: Merging Strategy Comparison\n- Fig <configured>: K-shot Performance Curves\n- Fig <configured>: t-SNE Embedding Visualization\n- Fig <configured>: Confusion Matrix\n- Fig <configured>: Per-class Accuracy Comparison\n- Fig <configured>: Threshold Sensitivity Analysis\n- Fig <configured>: Adaptive Prototype Distribution\n'
from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import gc
import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveEvaluator:
    '\n    综合评估器 - 生成论文所需的所有实验数据\n    \n    功能:\n    <configured>. 多指标评估 (Accuracy, F1, Precision, Recall)\n    <configured>. 不同初始化策略对比\n    <configured>. 不同合并策略对比\n    <configured>. 超参数敏感性分析\n    <configured>. 嵌入可视化\n    <configured>. 混淆矩阵生成\n    <configured>. Per-class分析\n    '

    def __init__(self, trainer, device=None):
        """
        Args:
            trainer: GPNTrainer实例
            device: 计算设备
        """
        device = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.__init__.device', device)
        self.trainer = trainer
        self.device = device
        self.results_cache = {}

    # ==================== 核心评估方法 ====================

    def evaluate_with_metrics(self,model, test_loader, n_ways, return_embeddings=False):
        """
        带多指标的评估方法

        Returns:
            dict: {
                'accuracy': float,
                'f1_macro': float,
                'f1_weighted': float,
                'precision': float,
                'recall': float,
                'per_class_acc': dict,
                'confusion_matrix': np.array,
                'all_predictions': list,
                'all_labels': list,
                'embeddings': dict (if return_embeddings=True)
            }
        """
        model.eval()

        all_predictions = []
        all_labels = []
        all_query_embeddings = []
        all_support_embeddings = []
        all_support_labels = []
        all_query_labels = []
        all_prototypes = []
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        prototype_counts_per_class = defaultdict(list)

        with torch.no_grad():
            for meta_task in tqdm(test_loader, desc="Evaluating"):
                support_signals, support_labels, query_signals, query_labels = meta_task

                # 数据预处理
                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                # 数据增强
                if self.trainer.crop_ratio_h != 0 or self.trainer.crop_ratio_l != 0:
                    from amgpn.data.preprocessing import crop_and_rescale_symmetric
                    support_signals = torch.stack([
                        crop_and_rescale_symmetric(s, self.trainer.crop_ratio_h,
                                                   self.trainer.crop_ratio_l, self.trainer.resize)
                        for s in support_signals.unbind(0)
                    ])
                    query_signals = torch.stack([
                        crop_and_rescale_symmetric(s, self.trainer.crop_ratio_h,
                                                   self.trainer.crop_ratio_l, self.trainer.resize)
                        for s in query_signals.unbind(0)
                    ])

                # 特征提取
                support_v, support_s = self.model(support_signals)
                query_v, query_s = self.model(query_signals)

                # 标签映射
                unique_labels = torch.unique(support_labels)
                label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
                remapped_to_original = {i: label.item() for i, label in enumerate(unique_labels)}

                remapped_support_labels = torch.tensor(
                    [label_mapping[l.item()] for l in support_labels], device=self.device
                )
                remapped_query_labels = torch.tensor(
                    [label_mapping[l.item()] for l in query_labels], device=self.device
                )

                # 计算预测
                if self.trainer.use_multi:
                    k_shot = len(support_labels) // n_ways
                    prototypes = support_v
                    precision_matrices = self.trainer.compute_precision_matrices_from_support(support_s)

                    output = self.trainer.loss_fn(
                        query_v, prototypes, precision_matrices,
                        remapped_query_labels, epoch=self.trainer.current_epoch
                    )

                    logits = -output['distances']
                    predictions = torch.argmax(logits, dim=1)

                    # 记录原型数量
                    prototype_mask = output['prototype_mask']
                    for i, label in enumerate(unique_labels):
                        active_count = prototype_mask[i].sum().item()
                        prototype_counts_per_class[label.item()].append(active_count)

                    if return_embeddings:
                        all_prototypes.append({
                            'prototypes': prototypes.cpu().numpy(),
                            'mask': prototype_mask.cpu().numpy(),
                            'labels': unique_labels.cpu().numpy()
                        })
                else:
                    prototypes, precision_matrices = self.trainer.compute_gaussian_prototypes(
                        support_v, support_s, remapped_support_labels, n_ways
                    )
                    distances = self.trainer.loss_fn.distance_metric(query_v, prototypes, precision_matrices)
                    predictions = torch.argmin(distances, dim=1)

                # 收集结果
                for pred, true_label in zip(predictions.cpu().numpy(),
                                           remapped_query_labels.cpu().numpy()):
                    original_pred = remapped_to_original[pred]
                    original_true = remapped_to_original[true_label]

                    all_predictions.append(original_pred)
                    all_labels.append(original_true)

                    per_class_total[original_true] += 1
                    if original_pred == original_true:
                        per_class_correct[original_true] += 1

                if return_embeddings:
                    all_query_embeddings.append(query_v.cpu().numpy())
                    all_support_embeddings.append(support_v.cpu().numpy())
                    all_support_labels.append(support_labels.cpu().numpy())
                    all_query_labels.append(query_labels.cpu().numpy())

        # 计算指标
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = np.mean(all_predictions == all_labels)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')

        # Per-class accuracy
        per_class_acc = {
            cls: per_class_correct[cls] / per_class_total[cls]
            for cls in per_class_total.keys()
        }

        # 混淆矩阵
        unique_classes = sorted(set(all_labels))
        cm = confusion_matrix(all_labels, all_predictions, labels=unique_classes)

        result = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'per_class_acc': per_class_acc,
            'confusion_matrix': cm,
            'class_labels': unique_classes,
            'all_predictions': all_predictions,
            'all_labels': all_labels,
            'prototype_counts': dict(prototype_counts_per_class)
        }

        if return_embeddings:
            result['query_embeddings'] = np.concatenate(all_query_embeddings, axis=0)
            result['support_embeddings'] = np.concatenate(all_support_embeddings, axis=0)
            result['support_labels'] = np.concatenate(all_support_labels, axis=0)
            result['query_labels'] = np.concatenate(all_query_labels, axis=0)
            result['prototypes_info'] = all_prototypes

        return result


    def generate_main_results(self, test_dataset, k_shots=None,
                              n_way=None, n_trials=None, q_query=None):
        "\n        生成Table <configured>的完整数据\n        \n        Returns:\n            dict: {method: {k_shot: {'acc': (mean, std), 'f1': (mean, std)}}}\n        "
        k_shots = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.generate_main_results.k_shots', k_shots)
        n_way = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.generate_main_results.n_way', n_way)
        n_trials = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.generate_main_results.n_trials', n_trials)
        q_query = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.generate_main_results.q_query', q_query)
        results = {}

        for k_shot in k_shots:
            print(f"\n{'='*50}")
            print(f"Evaluating {n_way}-way {k_shot}-shot")
            print(f"{'='*50}")

            accuracies = []
            f1_scores = []

            for trial in tqdm(range(n_trials), desc=f"{n_way}w{k_shot}s"):
                from amgpn.data.episodes import MetaDataset
                meta_test = MetaDataset(
                    test_dataset, n_way, k_shot, q_query=q_query,
                    num_tasks=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.generate_main_results.num_tasks'), seed=random.randint(0, _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.generate_main_results.size_or_budget'))
                )
                loader = DataLoader(meta_test, batch_size=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.generate_main_results.batch_size'))

                eval_result = self.evaluate_with_metrics(loader, n_way)
                accuracies.append(eval_result['accuracy'])
                f1_scores.append(eval_result['f1_macro'])

                del meta_test, loader
                if (trial + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 计算<configured>置信区间
            acc_mean = np.mean(accuracies)
            acc_ci = 1.96 * np.std(accuracies) / np.sqrt(n_trials)
            f1_mean = np.mean(f1_scores)
            f1_ci = 1.96 * np.std(f1_scores) / np.sqrt(n_trials)

            results[f'{k_shot}-shot'] = {
                'accuracy': (acc_mean, acc_ci),
                'f1_macro': (f1_mean, f1_ci)
            }

            print(f"Accuracy: {acc_mean:.4f} ± {acc_ci:.4f}")
            print(f"F1-score: {f1_mean:.4f} ± {f1_ci:.4f}")

        return results


    def compare_initialization_strategies(self, test_dataset, n_way=None, k_shot=None,
                                          n_trials=None, q_query=None):
        '\n        对比不同初始化策略\n        \n        策略:\n        <configured>. Instance-level (AMGPN默认)\n        <configured>. K-means clustering\n        <configured>. DP-means (adaptive k)\n        <configured>. Spectral clustering\n        '
        n_way = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_initialization_strategies.n_way', n_way)
        k_shot = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_initialization_strategies.k_shot', k_shot)
        n_trials = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_initialization_strategies.n_trials', n_trials)
        q_query = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_initialization_strategies.q_query', q_query)
        strategies = {
            'Instance-level (Ours)': self._init_instance_level,
            'K-means (k=8)': lambda v, s, n: self._init_kmeans(v, s, n, k=8),
            'DP-means': self._init_dpmeans,
        }

        results = {}

        for strategy_name, init_fn in strategies.items():
            print(f"\n{'='*50}")
            print(f"Testing initialization: {strategy_name}")
            print(f"{'='*50}")

            accuracies = []
            f1_scores = []
            avg_prototypes = []
            times = []

            for trial in tqdm(range(n_trials), desc=strategy_name):
                from amgpn.data.episodes import MetaDataset
                meta_test = MetaDataset(
                    test_dataset, n_way, k_shot, q_query=q_query,
                    num_tasks=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.compare_initialization_strategies.num_tasks'), seed=random.randint(0, _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.compare_initialization_strategies.size_or_budget__2'))
                )
                loader = DataLoader(meta_test, batch_size=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.compare_initialization_strategies.batch_size'))

                # 使用自定义初始化进行评估
                start_time = time.time()
                eval_result = self._evaluate_with_custom_init(
                    loader, n_way, init_fn
                )
                elapsed = (time.time() - start_time) * _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.compare_initialization_strategies.size_or_budget')  # ms

                accuracies.append(eval_result['accuracy'])
                f1_scores.append(eval_result['f1_macro'])
                avg_prototypes.append(eval_result.get('avg_prototypes', k_shot))
                times.append(elapsed)

                del meta_test, loader

            results[strategy_name] = {
                'accuracy': (np.mean(accuracies), 1.96 * np.std(accuracies) / np.sqrt(n_trials)),
                'f1_macro': (np.mean(f1_scores), 1.96 * np.std(f1_scores) / np.sqrt(n_trials)),
                'avg_Mc': np.mean(avg_prototypes),
                'time_ms': np.mean(times)
            }

            print(f"Accuracy: {np.mean(accuracies):.4f} ± {1.96*np.std(accuracies)/np.sqrt(n_trials):.4f}")
            print(f"Avg Mc: {np.mean(avg_prototypes):.1f}")
            print(f"Time: {np.mean(times):.1f} ms")

        return results

    def _init_instance_level(self, support_v, support_s, n_ways):
        """Instance-level初始化 (AMGPN默认)"""
        k_shot = support_v.shape[0] // n_ways
        prototypes = support_v.clone()
        precision_matrices = self.trainer.compute_precision_matrices_from_support(support_s)
        return prototypes, precision_matrices, k_shot

    def _init_kmeans(self, support_v, support_s, n_ways, k=8):
        """K-means聚类初始化"""
        k_shot = support_v.shape[0] // n_ways
        feature_dim = support_v.shape[1]

        all_prototypes = []
        all_precisions = []

        for c in range(n_ways):
            start_idx = c * k_shot
            end_idx = start_idx + k_shot
            class_features = support_v[start_idx:end_idx].cpu().numpy()
            class_s = support_s[start_idx:end_idx]

            # K-means聚类
            n_clusters = min(k, len(class_features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator._init_kmeans.random_state'), n_init=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator._init_kmeans.n_init'))
            kmeans.fit(class_features)

            centers = torch.tensor(kmeans.cluster_centers_, device=self.device, dtype=torch.float32)

            # 为每个聚类中心计算精度矩阵
            for center_idx in range(n_clusters):
                cluster_mask = kmeans.labels_ == center_idx
                if cluster_mask.sum() > 0:
                    cluster_s = class_s[cluster_mask].mean(dim=0)
                    sigma = cluster_s
                    precision = torch.diag(sigma)
                else:
                    precision = torch.eye(feature_dim, device=self.device)

                all_prototypes.append(centers[center_idx])
                all_precisions.append(precision)

        prototypes = torch.stack(all_prototypes)
        precision_matrices = torch.stack(all_precisions)

        return prototypes, precision_matrices, n_clusters

    def _init_dpmeans(self, support_v, support_s, n_ways, lambda_param=None):
        """DP-means (Dirichlet Process) 自适应聚类初始化"""
        lambda_param = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator._init_dpmeans.lambda_param', lambda_param)
        k_shot = support_v.shape[0] // n_ways
        feature_dim = support_v.shape[1]

        all_prototypes = []
        all_precisions = []
        total_clusters = 0

        for c in range(n_ways):
            start_idx = c * k_shot
            end_idx = start_idx + k_shot
            class_features = support_v[start_idx:end_idx]
            class_s = support_s[start_idx:end_idx]

            # DP-means算法
            centers = [class_features[0].clone()]
            assignments = [0]

            for i in range(1, len(class_features)):
                x = class_features[i]
                min_dist = float('inf')
                closest = 0

                for j, center in enumerate(centers):
                    dist = torch.norm(x - center).item()
                    if dist < min_dist:
                        min_dist = dist
                        closest = j

                if min_dist > lambda_param:
                    centers.append(x.clone())
                    assignments.append(len(centers) - 1)
                else:
                    assignments.append(closest)

            # 更新聚类中心
            assignments = torch.tensor(assignments, device=self.device)
            for j in range(len(centers)):
                mask = (assignments == j)
                if mask.sum() > 0:
                    centers[j] = class_features[mask].mean(dim=0)
                    cluster_s = class_s[mask].mean(dim=0)
                    sigma = cluster_s
                    precision = torch.diag(sigma)

                    all_prototypes.append(centers[j])
                    all_precisions.append(precision)

            total_clusters += len(centers)

        prototypes = torch.stack(all_prototypes)
        precision_matrices = torch.stack(all_precisions)
        avg_clusters = total_clusters / n_ways

        return prototypes, precision_matrices, avg_clusters

    def _evaluate_with_custom_init(self, test_loader, n_ways, init_fn):
        """使用自定义初始化策略进行评估"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_prototype_counts = []

        with torch.no_grad():
            for meta_task in test_loader:
                support_signals, support_labels, query_signals, query_labels = meta_task

                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                # 数据增强
                if self.trainer.crop_ratio_h != 0 or self.trainer.crop_ratio_l != 0:
                    from amgpn.data.preprocessing import crop_and_rescale_symmetric
                    support_signals = torch.stack([
                        crop_and_rescale_symmetric(s, self.trainer.crop_ratio_h,
                                                   self.trainer.crop_ratio_l, self.trainer.resize)
                        for s in support_signals.unbind(0)
                    ])
                    query_signals = torch.stack([
                        crop_and_rescale_symmetric(s, self.trainer.crop_ratio_h,
                                                   self.trainer.crop_ratio_l, self.trainer.resize)
                        for s in query_signals.unbind(0)
                    ])

                support_v, support_s = self.model(support_signals)
                query_v, query_s = self.model(query_signals)

                unique_labels = torch.unique(support_labels)
                label_mapping = {l.item(): i for i, l in enumerate(unique_labels)}
                remapped_to_original = {i: l.item() for i, l in enumerate(unique_labels)}

                remapped_query_labels = torch.tensor(
                    [label_mapping[l.item()] for l in query_labels], device=self.device
                )

                # 使用自定义初始化
                prototypes, precision_matrices, avg_protos = init_fn(support_v, support_s, n_ways)
                all_prototype_counts.append(avg_protos)

                # 应用合并 (如果是AMGPN)
                if self.trainer.use_multi and prototypes.shape[0] == support_v.shape[0]:
                    k_shot = support_v.shape[0] // n_ways
                    prototype_mask = torch.ones(n_ways, k_shot, dtype=torch.bool, device=self.device)

                    prototypes, precision_matrices, prototype_mask, _ = \
                        self.trainer.loss_fn._merge_prototypes_in_task_optimized(
                            prototypes, precision_matrices, prototype_mask, n_ways, k_shot
                        )

                    # 计算距离
                    all_distances = self.trainer.loss_fn.distance_metric(
                        query_v, prototypes, precision_matrices
                    )
                    distances_reshaped = all_distances.view(-1, n_ways, k_shot)

                    mask_expanded = prototype_mask.unsqueeze(0).expand(distances_reshaped.shape[0], -1, -1)
                    distances_reshaped = torch.where(
                        mask_expanded, distances_reshaped,
                        torch.tensor(float('inf'), device=self.device)
                    )

                    min_distances, _ = torch.min(distances_reshaped, dim=2)
                    predictions = torch.argmin(min_distances, dim=1)
                else:
                    distances = self.trainer.loss_fn.distance_metric(
                        query_v, prototypes, precision_matrices
                    )
                    predictions = torch.argmin(distances, dim=1)

                for pred, true_label in zip(predictions.cpu().numpy(),
                                           remapped_query_labels.cpu().numpy()):
                    all_predictions.append(remapped_to_original[pred])
                    all_labels.append(remapped_to_original[true_label])

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        return {
            'accuracy': np.mean(all_predictions == all_labels),
            'f1_macro': f1_score(all_labels, all_predictions, average='macro'),
            'avg_prototypes': np.mean(all_prototype_counts)
        }


    def compare_merging_strategies(self, test_dataset, n_way=None, k_shot=None,
                                   n_trials=None, q_query=None, threshold=None):
        '\n        对比不同合并策略\n        \n        策略:\n        <configured>. No merging\n        <configured>. Pairwise merging (no transitivity)\n        <configured>. Graph connectivity (AMGPN默认)\n        <configured>. Hierarchical agglomerative\n        <configured>. DBSCAN-based\n        '
        n_way = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.n_way', n_way)
        k_shot = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.k_shot', k_shot)
        n_trials = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.n_trials', n_trials)
        q_query = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.q_query', q_query)
        threshold = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.threshold', threshold)
        strategies = {
            'No merging': self._merge_none,
            'Pairwise (no trans.)': lambda p, prec, m, nw, ks: self._merge_pairwise(p, prec, m, nw, ks, threshold),
            'Graph connectivity (Ours)': lambda p, prec, m, nw, ks: self._merge_graph(p, prec, m, nw, ks, threshold),
            'Hierarchical': lambda p, prec, m, nw, ks: self._merge_hierarchical(p, prec, m, nw, ks, threshold),
            'DBSCAN-based': lambda p, prec, m, nw, ks: self._merge_dbscan(p, prec, m, nw, ks, threshold),
        }

        results = {}

        for strategy_name, merge_fn in strategies.items():
            print(f"\n{'='*50}")
            print(f"Testing merging strategy: {strategy_name}")
            print(f"{'='*50}")

            accuracies = []
            f1_scores = []
            avg_prototypes = []
            times = []

            for trial in tqdm(range(n_trials), desc=strategy_name):
                from amgpn.data.episodes import MetaDataset
                meta_test = MetaDataset(
                    test_dataset, n_way, k_shot, q_query=q_query,
                    num_tasks=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.num_tasks'), seed=random.randint(0, _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.size_or_budget__2'))
                )
                loader = DataLoader(meta_test, batch_size=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.batch_size'))

                start_time = time.time()
                eval_result = self._evaluate_with_custom_merge(
                    loader, n_way, merge_fn
                )
                elapsed = (time.time() - start_time) * _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.compare_merging_strategies.size_or_budget')

                accuracies.append(eval_result['accuracy'])
                f1_scores.append(eval_result['f1_macro'])
                avg_prototypes.append(eval_result['avg_prototypes'])
                times.append(elapsed)

                del meta_test, loader

            results[strategy_name] = {
                'accuracy': (np.mean(accuracies), 1.96 * np.std(accuracies) / np.sqrt(n_trials)),
                'f1_macro': (np.mean(f1_scores), 1.96 * np.std(f1_scores) / np.sqrt(n_trials)),
                'avg_Mc': np.mean(avg_prototypes),
                'time_ms': np.mean(times)
            }

            print(f"Accuracy: {np.mean(accuracies):.4f}")
            print(f"Avg Mc: {np.mean(avg_prototypes):.1f}")
            print(f"Time: {np.mean(times):.1f} ms")

        return results

    def _merge_none(self, prototypes, precision_matrices, mask, n_ways, k_shot):
        """不合并"""
        return prototypes, precision_matrices, mask, 0

    def _merge_pairwise(self, prototypes, precision_matrices, mask, n_ways, k_shot, threshold):
        """简单配对合并（无传递性）"""
        feature_dim = prototypes.shape[1]
        prototypes_reshaped = prototypes.view(n_ways, k_shot, feature_dim)
        precision_reshaped = precision_matrices.view(n_ways, k_shot, feature_dim, feature_dim)

        merge_count = 0

        for c in range(n_ways):
            active_indices = torch.where(mask[c])[0].tolist()
            merged = set()

            for i in range(len(active_indices)):
                if active_indices[i] in merged:
                    continue
                for j in range(i + 1, len(active_indices)):
                    if active_indices[j] in merged:
                        continue

                    idx_i, idx_j = active_indices[i], active_indices[j]
                    dist = self._compute_mahalanobis_distance(
                        prototypes_reshaped[c, idx_i],
                        prototypes_reshaped[c, idx_j],
                        precision_reshaped[c, idx_i],
                        precision_reshaped[c, idx_j]
                    )

                    if dist < threshold:
                        # 合并到i
                        prototypes_reshaped[c, idx_i] = (
                            prototypes_reshaped[c, idx_i] + prototypes_reshaped[c, idx_j]
                        ) / 2
                        precision_reshaped[c, idx_i] = (
                            precision_reshaped[c, idx_i] + precision_reshaped[c, idx_j]
                        ) / 2
                        mask[c, idx_j] = False
                        merged.add(idx_j)
                        merge_count += 1

        return (prototypes_reshaped.view(-1, feature_dim),
                precision_reshaped.view(-1, feature_dim, feature_dim),
                mask, merge_count)

    def _merge_graph(self, prototypes, precision_matrices, mask, n_ways, k_shot, threshold):
        """图连通分量合并（AMGPN默认）"""
        return self.trainer.loss_fn._merge_prototypes_in_task_optimized(
            prototypes, precision_matrices, mask, n_ways, k_shot
        )

    def _merge_hierarchical(self, prototypes, precision_matrices, mask, n_ways, k_shot, threshold):
        """层次聚类合并"""
        feature_dim = prototypes.shape[1]
        prototypes_reshaped = prototypes.view(n_ways, k_shot, feature_dim)
        precision_reshaped = precision_matrices.view(n_ways, k_shot, feature_dim, feature_dim)

        merge_count = 0

        for c in range(n_ways):
            active_indices = torch.where(mask[c])[0]
            if len(active_indices) <= 1:
                continue

            class_protos = prototypes_reshaped[c, active_indices].cpu().numpy()

            # 层次聚类
            Z = linkage(class_protos, method='average')
            clusters = fcluster(Z, t=threshold, criterion='distance')

            # 合并同一簇的原型
            unique_clusters = np.unique(clusters)
            for cluster_id in unique_clusters:
                cluster_mask = (clusters == cluster_id)
                cluster_indices = active_indices[cluster_mask]

                if len(cluster_indices) > 1:
                    # 合并到第一个
                    merged_proto = prototypes_reshaped[c, cluster_indices].mean(dim=0)
                    merged_prec = precision_reshaped[c, cluster_indices].mean(dim=0)

                    target_idx = cluster_indices[0].item()
                    prototypes_reshaped[c, target_idx] = merged_proto
                    precision_reshaped[c, target_idx] = merged_prec

                    for idx in cluster_indices[1:]:
                        mask[c, idx] = False
                        merge_count += 1

        return (prototypes_reshaped.view(-1, feature_dim),
                precision_reshaped.view(-1, feature_dim, feature_dim),
                mask, merge_count)

    def _merge_dbscan(self, prototypes, precision_matrices, mask, n_ways, k_shot, threshold):
        """DBSCAN聚类合并"""
        feature_dim = prototypes.shape[1]
        prototypes_reshaped = prototypes.view(n_ways, k_shot, feature_dim)
        precision_reshaped = precision_matrices.view(n_ways, k_shot, feature_dim, feature_dim)

        merge_count = 0

        for c in range(n_ways):
            active_indices = torch.where(mask[c])[0]
            if len(active_indices) <= 1:
                continue

            class_protos = prototypes_reshaped[c, active_indices].cpu().numpy()

            # DBSCAN
            dbscan = DBSCAN(eps=threshold, min_samples=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator._merge_dbscan.min_samples'))
            clusters = dbscan.fit_predict(class_protos)

            unique_clusters = np.unique(clusters[clusters >= 0])
            for cluster_id in unique_clusters:
                cluster_mask = (clusters == cluster_id)
                cluster_indices = active_indices[cluster_mask]

                if len(cluster_indices) > 1:
                    merged_proto = prototypes_reshaped[c, cluster_indices].mean(dim=0)
                    merged_prec = precision_reshaped[c, cluster_indices].mean(dim=0)

                    target_idx = cluster_indices[0].item()
                    prototypes_reshaped[c, target_idx] = merged_proto
                    precision_reshaped[c, target_idx] = merged_prec

                    for idx in cluster_indices[1:]:
                        mask[c, idx] = False
                        merge_count += 1

        return (prototypes_reshaped.view(-1, feature_dim),
                precision_reshaped.view(-1, feature_dim, feature_dim),
                mask, merge_count)

    def _compute_mahalanobis_distance(self, proto_i, proto_j, prec_i, prec_j):
        """计算两个原型之间的马氏距离"""
        avg_prec = (prec_i + prec_j) / 2
        diff = proto_i - proto_j
        dist_sq = torch.dot(diff, torch.mv(avg_prec, diff))
        return torch.sqrt(torch.clamp(dist_sq, min=1e-10)).item()

    def _evaluate_with_custom_merge(self, test_loader, n_ways, merge_fn):
        """使用自定义合并策略进行评估"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_prototype_counts = []

        with torch.no_grad():
            for meta_task in test_loader:
                support_signals, support_labels, query_signals, query_labels = meta_task

                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                if self.trainer.crop_ratio_h != 0 or self.trainer.crop_ratio_l != 0:
                    from amgpn.data.preprocessing import crop_and_rescale_symmetric
                    support_signals = torch.stack([
                        crop_and_rescale_symmetric(s, self.trainer.crop_ratio_h,
                                                   self.trainer.crop_ratio_l, self.trainer.resize)
                        for s in support_signals.unbind(0)
                    ])
                    query_signals = torch.stack([
                        crop_and_rescale_symmetric(s, self.trainer.crop_ratio_h,
                                                   self.trainer.crop_ratio_l, self.trainer.resize)
                        for s in query_signals.unbind(0)
                    ])

                support_v, support_s = self.model(support_signals)
                query_v, query_s = self.model(query_signals)

                k_shot = support_v.shape[0] // n_ways
                unique_labels = torch.unique(support_labels)
                label_mapping = {l.item(): i for i, l in enumerate(unique_labels)}
                remapped_to_original = {i: l.item() for i, l in enumerate(unique_labels)}

                remapped_query_labels = torch.tensor(
                    [label_mapping[l.item()] for l in query_labels], device=self.device
                )

                # Instance-level初始化
                prototypes = support_v.clone()
                precision_matrices = self.trainer.compute_precision_matrices_from_support(support_s)
                prototype_mask = torch.ones(n_ways, k_shot, dtype=torch.bool, device=self.device)

                # 应用自定义合并
                prototypes, precision_matrices, prototype_mask, _ = merge_fn(
                    prototypes, precision_matrices, prototype_mask, n_ways, k_shot
                )

                # 记录激活原型数
                active_count = prototype_mask.sum().item() / n_ways
                all_prototype_counts.append(active_count)

                # 计算距离和预测
                all_distances = self.trainer.loss_fn.distance_metric(
                    query_v, prototypes, precision_matrices
                )
                distances_reshaped = all_distances.view(-1, n_ways, k_shot)

                mask_expanded = prototype_mask.unsqueeze(0).expand(distances_reshaped.shape[0], -1, -1)
                distances_reshaped = torch.where(
                    mask_expanded, distances_reshaped,
                    torch.tensor(float('inf'), device=self.device)
                )

                min_distances, _ = torch.min(distances_reshaped, dim=2)
                predictions = torch.argmin(min_distances, dim=1)

                for pred, true_label in zip(predictions.cpu().numpy(),
                                           remapped_query_labels.cpu().numpy()):
                    all_predictions.append(remapped_to_original[pred])
                    all_labels.append(remapped_to_original[true_label])

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        return {
            'accuracy': np.mean(all_predictions == all_labels),
            'f1_macro': f1_score(all_labels, all_predictions, average='macro'),
            'avg_prototypes': np.mean(all_prototype_counts)
        }


    def analyze_threshold_sensitivity(self, test_dataset, thresholds=None,
                                      n_way=None, k_shot=None, n_trials=None, q_query=None):
        """
        分析合并阈值τ的敏感性
        """
        thresholds = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.analyze_threshold_sensitivity.thresholds', thresholds)
        n_way = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.analyze_threshold_sensitivity.n_way', n_way)
        k_shot = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.analyze_threshold_sensitivity.k_shot', k_shot)
        n_trials = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.analyze_threshold_sensitivity.n_trials', n_trials)
        q_query = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.analyze_threshold_sensitivity.q_query', q_query)
        results = {}
        original_threshold = self.trainer.loss_fn.merge_threshold

        for tau in thresholds:
            print(f"\n{'='*50}")
            print(f"Testing threshold τ = {tau}")
            print(f"{'='*50}")

            # 临时修改阈值
            self.trainer.loss_fn.merge_threshold = tau

            accuracies = []
            f1_scores = []
            avg_prototypes = []

            for trial in tqdm(range(n_trials), desc=f"τ={tau}"):
                from amgpn.data.episodes import MetaDataset
                meta_test = MetaDataset(
                    test_dataset, n_way, k_shot, q_query=q_query,
                    num_tasks=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.analyze_threshold_sensitivity.num_tasks'), seed=random.randint(0, _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.analyze_threshold_sensitivity.size_or_budget'))
                )
                loader = DataLoader(meta_test, batch_size=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.analyze_threshold_sensitivity.batch_size'))

                eval_result = self.evaluate_with_metrics(loader, n_way)
                accuracies.append(eval_result['accuracy'])
                f1_scores.append(eval_result['f1_macro'])

                # 计算平均原型数
                if eval_result['prototype_counts']:
                    avg_mc = np.mean([np.mean(v) for v in eval_result['prototype_counts'].values()])
                    avg_prototypes.append(avg_mc)

                del meta_test, loader

            results[tau] = {
                'accuracy': (np.mean(accuracies), 1.96 * np.std(accuracies) / np.sqrt(n_trials)),
                'f1_macro': (np.mean(f1_scores), 1.96 * np.std(f1_scores) / np.sqrt(n_trials)),
                'avg_Mc': np.mean(avg_prototypes) if avg_prototypes else k_shot
            }

            print(f"Accuracy: {np.mean(accuracies):.4f} ± {1.96*np.std(accuracies)/np.sqrt(n_trials):.4f}")
            print(f"Avg Mc: {results[tau]['avg_Mc']:.1f}")

        # 恢复原阈值
        self.trainer.loss_fn.merge_threshold = original_threshold

        return results

    # ==================== 可视化方法 ====================

    def plot_tsne_embeddings(self, test_dataset, n_way=None, k_shot=None, q_query=None,
                             target_class=None, save_path=None):
        '\n        Fig <configured>: 单类 support set 与 prototype 的 t-SNE 可视化\n        '
        n_way = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.n_way', n_way)
        k_shot = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.k_shot', k_shot)
        q_query = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.q_query', q_query)
        save_path = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.save_path', save_path)
        from amgpn.data.episodes import MetaDataset

        meta_test = MetaDataset(
            test_dataset, n_way, k_shot, q_query=q_query,
            num_tasks=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.num_tasks'), seed=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.seed')
        )
        loader = DataLoader(meta_test, batch_size=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.batch_size'))

        eval_result = self.evaluate_with_metrics(self.trainer.model, loader, n_way, return_embeddings=True)

        support_embeddings = eval_result['support_embeddings']
        support_labels = eval_result['support_labels']

        unique_support_labels = np.unique(support_labels)
        if len(unique_support_labels) == 0:
            raise ValueError('No support samples found for t-SNE visualization')

        if target_class is None:
            target_class = unique_support_labels[0]
        if target_class not in unique_support_labels:
            raise ValueError(f'target_class={target_class} is not present in the current support set')

        class_mask = support_labels == target_class
        class_support_embeddings = support_embeddings[class_mask]
        class_support_labels = support_labels[class_mask]
        class_prototype = class_support_embeddings.mean(axis=0, keepdims=True)

        tsne_input = np.concatenate([class_support_embeddings, class_prototype], axis=0)
        n_samples = tsne_input.shape[0]
        perplexity = min(30, max(2, n_samples - 1))
        tsne = TSNE(n_components=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.n_components'), random_state=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.random_state'), perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(tsne_input)

        support_2d = embeddings_2d[:-1]
        prototype_2d = embeddings_2d[-1]

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.scatter(
            support_2d[:, 0], support_2d[:, 1],
            c=['#1f77b4'], label=f'Support class {target_class}',
            alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.alpha'), s=70, edgecolors='white', linewidths=0.6
        )
        ax.scatter(
            prototype_2d[0], prototype_2d[1],
            c='#d62728', marker='*', s=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.size_or_budget'), label='Prototype',
            edgecolors='black', linewidths=0.8
        )

        for idx, (x_coord, y_coord) in enumerate(support_2d):
            ax.text(x_coord, y_coord, str(idx + 1), fontsize=8, ha='center', va='center')

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f'Single-Class t-SNE: class {target_class}', fontsize=14)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.alpha__2'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_tsne_embeddings.size_or_budget__2'), bbox_inches='tight')
        plt.close()

        print(f"t-SNE visualization saved to {save_path} (class {target_class})")
        return embeddings_2d, class_support_labels

    def plot_confusion_matrix(self, test_dataset, n_way=None, k_shot=None, n_trials=None,
                              q_query=None, save_path=None):
        '\n        Fig <configured>: 混淆矩阵\n        '
        n_way = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.n_way', n_way)
        k_shot = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.k_shot', k_shot)
        n_trials = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.n_trials', n_trials)
        q_query = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.q_query', q_query)
        save_path = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.save_path', save_path)
        from amgpn.data.episodes import MetaDataset

        # 收集所有预测
        all_preds = []
        all_labels = []

        for trial in tqdm(range(n_trials), desc="Collecting predictions"):
            meta_test = MetaDataset(
                test_dataset, n_way, k_shot, q_query=q_query,
                num_tasks=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.num_tasks'), seed=random.randint(0, _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.size_or_budget__2'))
            )
            loader = DataLoader(meta_test, batch_size=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.batch_size'))

            eval_result = self.evaluate_with_metrics(loader, n_way)
            all_preds.extend(eval_result['all_predictions'])
            all_labels.extend(eval_result['all_labels'])

            del meta_test, loader

        # 计算混淆矩阵
        unique_classes = sorted(set(all_labels))
        cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=unique_classes, yticklabels=unique_classes, ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_confusion_matrix.size_or_budget'), bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {save_path}")
        return cm, cm_normalized, unique_classes

    def plot_per_class_accuracy(self, results_amgpn, results_gpn, class_names=None,
                                save_path=None):
        '\n        Fig <configured>: Per-class Accuracy对比柱状图\n        '
        save_path = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_per_class_accuracy.save_path', save_path)
        classes = sorted(results_amgpn.keys())

        if class_names is None:
            class_names = [f'Class {c}' for c in classes]

        amgpn_acc = [results_amgpn[c] for c in classes]
        gpn_acc = [results_gpn[c] for c in classes]

        # 按improvement排序
        improvements = [a - g for a, g in zip(amgpn_acc, gpn_acc)]
        sorted_indices = np.argsort(improvements)[::-1]

        classes = [classes[i] for i in sorted_indices]
        class_names = [class_names[i] for i in sorted_indices]
        amgpn_acc = [amgpn_acc[i] for i in sorted_indices]
        gpn_acc = [gpn_acc[i] for i in sorted_indices]
        improvements = [improvements[i] for i in sorted_indices]

        # 绘图
        x = np.arange(len(classes))
        width = _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_per_class_accuracy.width')

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, gpn_acc, width, label='GPN', color='steelblue', alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_per_class_accuracy.alpha'))
        bars2 = ax.bar(x + width/2, amgpn_acc, width, label='AMGPN (Ours)', color='darkorange', alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_per_class_accuracy.alpha__2'))

        # 添加improvement标注
        for i, (bar, imp) in enumerate(zip(bars2, improvements)):
            height = bar.get_height()
            ax.annotate(f'+{imp*100:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, color='darkgreen')

        ax.set_xlabel('UAV Class', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-class Accuracy Comparison: GPN vs AMGPN', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0.7, 1.05])
        ax.grid(True, alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_per_class_accuracy.alpha__3'), axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_per_class_accuracy.size_or_budget'), bbox_inches='tight')
        plt.close()

        print(f"Per-class accuracy plot saved to {save_path}")

    def plot_threshold_sensitivity(self, results, save_path=None):
        '\n        Fig <configured>: 阈值敏感性双轴曲线图\n        '
        save_path = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_threshold_sensitivity.save_path', save_path)
        thresholds = sorted(results.keys())
        accuracies = [results[t]['accuracy'][0] for t in thresholds]
        acc_cis = [results[t]['accuracy'][1] for t in thresholds]
        avg_mcs = [results[t]['avg_Mc'] for t in thresholds]

        fig, ax1 = plt.subplots(figsize=(8, 6))

        # 左轴：Accuracy
        color1 = 'tab:blue'
        ax1.set_xlabel('Merging Threshold τ', fontsize=12)
        ax1.set_ylabel('Accuracy', color=color1, fontsize=12)
        line1 = ax1.plot(thresholds, accuracies, 'o-', color=color1, linewidth=2,
                         markersize=8, label='Accuracy')
        ax1.fill_between(thresholds,
                         [a - ci for a, ci in zip(accuracies, acc_cis)],
                         [a + ci for a, ci in zip(accuracies, acc_cis)],
                         alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_threshold_sensitivity.alpha'), color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([min(accuracies) - 0.02, max(accuracies) + 0.02])

        # 右轴：Avg Mc
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Average Prototype Count $\\bar{M}_c$', color=color2, fontsize=12)
        line2 = ax2.plot(thresholds, avg_mcs, 's--', color=color2, linewidth=2,
                         markersize=8, label='Avg $M_c$')
        ax2.tick_params(axis='y', labelcolor=color2)

        # 标注最优点
        best_idx = np.argmax(accuracies)
        best_tau = thresholds[best_idx]
        ax1.axvline(x=best_tau, color='green', linestyle=':', alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_threshold_sensitivity.alpha__2'))
        ax1.annotate(f'Optimal τ={best_tau}', xy=(best_tau, accuracies[best_idx]),
                    xytext=(best_tau + 0.3, accuracies[best_idx] + 0.005),
                    fontsize=10, color='green')

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower right')

        ax1.set_title('Threshold Sensitivity Analysis', fontsize=14)
        ax1.grid(True, alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_threshold_sensitivity.alpha__3'))

        plt.tight_layout()
        plt.savefig(save_path, dpi=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_threshold_sensitivity.size_or_budget'), bbox_inches='tight')
        plt.close()

        print(f"Threshold sensitivity plot saved to {save_path}")

    def plot_kshot_curves(self, results, save_path=None):
        "\n        Fig <configured>: K-shot性能曲线\n        \n        Args:\n            results: dict of {method_name: {k_shot: {'accuracy': (mean, ci)}}}\n        "
        save_path = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_kshot_curves.save_path', save_path)
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {'CNN': 'gray', 'ProtoNet': 'green', 'GPN': 'blue', 'AMGPN (Ours)': 'red'}
        markers = {'CNN': 'o', 'ProtoNet': '^', 'GPN': 's', 'AMGPN (Ours)': 'D'}

        for method, method_results in results.items():
            k_shots = sorted([int(k.split('-')[0]) for k in method_results.keys()])
            accuracies = [method_results[f'{k}-shot']['accuracy'][0] for k in k_shots]
            cis = [method_results[f'{k}-shot']['accuracy'][1] for k in k_shots]

            color = colors.get(method, 'black')
            marker = markers.get(method, 'o')

            ax.plot(k_shots, accuracies, f'{marker}-', color=color, linewidth=2,
                   markersize=8, label=method)
            ax.fill_between(k_shots,
                           [a - ci for a, ci in zip(accuracies, cis)],
                           [a + ci for a, ci in zip(accuracies, cis)],
                           alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_kshot_curves.alpha__2'), color=color)

        ax.set_xlabel('Number of Shots (K)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('8-way K-shot Classification Performance', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_kshot_curves.alpha'))
        ax.set_xticks(k_shots)

        plt.tight_layout()
        plt.savefig(save_path, dpi=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_kshot_curves.size_or_budget'), bbox_inches='tight')
        plt.close()

        print(f"K-shot curves saved to {save_path}")

    def plot_prototype_distribution(self, test_dataset, n_way=None, k_shot=None, n_trials=None,
                                    q_query=None, save_path=None):
        '\n        Fig <configured>: 各类别原型数量分布箱线图\n        '
        n_way = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.n_way', n_way)
        k_shot = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.k_shot', k_shot)
        n_trials = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.n_trials', n_trials)
        q_query = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.q_query', q_query)
        save_path = _cfg_resolve('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.save_path', save_path)
        from amgpn.data.episodes import MetaDataset

        # 收集每个类别的原型数量
        class_prototype_counts = defaultdict(list)

        for trial in tqdm(range(n_trials), desc="Collecting prototype counts"):
            meta_test = MetaDataset(
                test_dataset, n_way, k_shot, q_query=q_query,
                num_tasks=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.num_tasks'), seed=random.randint(0, _cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.size_or_budget__2'))
            )
            loader = DataLoader(meta_test, batch_size=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.batch_size'))

            eval_result = self.evaluate_with_metrics(loader, n_way)

            for cls, counts in eval_result['prototype_counts'].items():
                class_prototype_counts[cls].extend(counts)

            del meta_test, loader

        # 准备数据
        classes = sorted(class_prototype_counts.keys())
        data = [class_prototype_counts[c] for c in classes]

        # 按中位数排序
        medians = [np.median(d) for d in data]
        sorted_indices = np.argsort(medians)
        classes = [classes[i] for i in sorted_indices]
        data = [data[i] for i in sorted_indices]

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))

        bp = ax.boxplot(data, patch_artist=True, labels=[f'Class {c}' for c in classes])

        # 设置颜色
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel('UAV Class', fontsize=12)
        ax.set_ylabel('Number of Active Prototypes', fontsize=12)
        ax.set_title(f'Adaptive Prototype Distribution ({k_shot}-shot)', fontsize=14)
        ax.axhline(y=k_shot, color='red', linestyle='--', alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.alpha'), label=f'Initial (K={k_shot})')
        ax.legend()
        ax.grid(True, alpha=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.alpha__2'), axis='y')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=_cfg_require('evaluation_module_false.py.ComprehensiveEvaluator.plot_prototype_distribution.size_or_budget'), bbox_inches='tight')
        plt.close()

        print(f"Prototype distribution plot saved to {save_path}")
        return class_prototype_counts


# ==================== 辅助函数：生成LaTeX表格 ====================

def generate_latex_table1(results):
    '生成Table <configured>的LaTeX代码'
    latex = '\n\\begin{table*}[t]\n\\centering\n\\caption{Classification Accuracy (\\%) on DroneRFb-Spectra Dataset (<configured>-way, <configured> episodes)}\n\\label{tab:main_results}\n\\small\n\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Method} & \\textbf{<configured>-shot} & \\textbf{<configured>-shot} & \\textbf{<configured>-shot} & \\textbf{<configured>-shot} & \\textbf{<configured>-shot} \\\\\n\\midrule\n'

    for method, method_results in results.items():
        row = f"{method}"
        for k in [1, 4, 7, 10, 15]:
            key = f'{k}-shot'
            if key in method_results:
                acc, ci = method_results[key]['accuracy']
                row += f" & {acc*100:.1f}$\\pm${ci*100:.1f}"
            else:
                row += " & -"
        row += r" \\"
        latex += row + "\n"

    latex += '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    return latex


def generate_latex_table2(results):
    '生成Table <configured> (Initialization Strategy) 的LaTeX代码'
    latex = '\n\\begin{table*}[h]\n\\centering\n\\caption{Initialization Strategy Comparison (<configured>-shot, <configured>-way)}\n\\label{tab:ablation_init}\n\\small\n\\begin{tabular}{lcccc}\n\\toprule\n\\textbf{Initialization Method} & \\textbf{Avg. $M_c$} & \\textbf{Time (ms)} & \\textbf{Accuracy (\\%)} & \\textbf{F1 (\\%)} \\\\\n\\midrule\n'

    for method, data in results.items():
        acc_mean, acc_ci = data['accuracy']
        f1_mean, f1_ci = data['f1_macro']
        latex += f"{method} & {data['avg_Mc']:.1f} & {data['time_ms']:.0f} & "
        latex += f"{acc_mean*100:.1f}$\\pm${acc_ci*100:.1f} & {f1_mean*100:.1f}$\\pm${f1_ci*100:.1f}"
        latex += r" \\" + "\n"

    latex += '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    return latex


def generate_latex_table3(results):
    '生成Table <configured> (Merging Strategy) 的LaTeX代码'
    latex = '\n\\begin{table*}[h]\n\\centering\n\\caption{Merging Strategy Comparison (<configured>-shot, <configured>-way)}\n\\label{tab:ablation_merging}\n\\small\n\\begin{tabular}{lcccc}\n\\toprule\n\\textbf{Merging Strategy} & \\textbf{Avg. $M_c$} & \\textbf{Time (ms)} & \\textbf{Accuracy (\\%)} & \\textbf{F1 (\\%)} \\\\\n\\midrule\n'

    for method, data in results.items():
        acc_mean, acc_ci = data['accuracy']
        f1_mean, f1_ci = data['f1_macro']
        latex += f"{method} & {data['avg_Mc']:.1f} & {data['time_ms']:.0f} & "
        latex += f"{acc_mean*100:.1f}$\\pm${acc_ci*100:.1f} & {f1_mean*100:.1f}$\\pm${f1_ci*100:.1f}"
        latex += r" \\" + "\n"

    latex += '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    return latex


# ==================== 主执行脚本 ====================

if __name__ == "__main__":
    print("="*60)
    print("AMGPN Comprehensive Evaluation Module")
    print("="*60)
    print("\n使用方法:")
    print("1. 创建GPNTrainer实例")
    print("2. 加载训练好的模型")
    print("3. 创建ComprehensiveEvaluator实例")
    print("4. 调用各评估方法生成数据")
    print("\n示例代码:")
    print("\nfrom evaluation_module import ComprehensiveEvaluator\n\n# 假设trainer已创建并加载了模型\nevaluator = ComprehensiveEvaluator(trainer, device='cuda')\n\n# Table 1: Main Results\nmain_results = evaluator.generate_main_results(test_dataset)\n\n# Table 2: Initialization Comparison\ninit_results = evaluator.compare_initialization_strategies(test_dataset)\n\n# Table 3: Merging Comparison  \nmerge_results = evaluator.compare_merging_strategies(test_dataset)\n\n# Fig 7: Threshold Sensitivity\nthreshold_results = evaluator.analyze_threshold_sensitivity(test_dataset)\nevaluator.plot_threshold_sensitivity(threshold_results)\n\n# Fig 4: t-SNE Visualization\nevaluator.plot_tsne_embeddings(test_dataset, target_class=0)\n\n# Fig 5: Confusion Matrix\nevaluator.plot_confusion_matrix(test_dataset)\n\n# Fig 8: Prototype Distribution\nevaluator.plot_prototype_distribution(test_dataset)\n")
