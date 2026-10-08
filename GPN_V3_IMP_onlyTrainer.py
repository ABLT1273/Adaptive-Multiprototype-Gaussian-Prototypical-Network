"""
GPNTrainerWithIMP - 全监督版本
改造适配 SupervisedIMP
"""
from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import gc
from typing import Tuple, List, Optional

# 导入全监督IMP组件
from loss_IMP import (
    SupervisedIMPPrototypeGenerator,
    SupervisedIMPLoss
)

#为了后期兼容裁剪，对原始文件进行了修改：插入裁剪相关逻辑
from data_feature_show import *
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import time

class GPNTrainerWithIMP:
    '\n    集成全监督IMP算法的GPNTrainer\n    \n    改造要点：\n    <configured>. 移除sigma_l和sigma_u，统一为sigma_init\n    <configured>. 移除所有is_labeled相关逻辑\n    <configured>. 简化原型生成接口\n    <configured>. 保持与原有训练框架的兼容性\n    '

    def __init__(self,
                 model,
                 device,
                 crop_ratio_h=None,crop_ratio_l=None, resize=None,
                 alpha: float = None,
                 sigma_init: float = None,  # ✓ 改动<configured>：统一方差参数
                 prototype_mode: str = None,  # 'single', 'multi', 'imp'
                 distance_type: str = None,     # 'E': Euclidean, 'M': Mahalanobis
                 lr: float = None,
                 prototypes_per_class: int = None):  # 仅用于multi模式
        """
        Args:
            model: 特征提取模型
            device: 计算设备
            alpha: CRP浓度参数（仅IMP模式）
            sigma_init: 初始cluster方差（仅IMP模式）
            prototype_mode: 原型模式
                - 'imp': 使用全监督IMP（推荐）
                - 'multi': 使用固定多原型
                - 'single': 使用单原型
            distance_type: 距离度量类型
            lr: 学习率
            prototypes_per_class: 每类原型数量（仅multi模式）
        """
        crop_ratio_h = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.crop_ratio_h', crop_ratio_h)
        crop_ratio_l = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.crop_ratio_l', crop_ratio_l)
        resize = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.resize', resize)
        alpha = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.alpha', alpha)
        sigma_init = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.sigma_init', sigma_init)
        prototype_mode = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.prototype_mode', prototype_mode)
        distance_type = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.distance_type', distance_type)
        lr = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.lr', lr)
        prototypes_per_class = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.prototypes_per_class', prototypes_per_class)
        self.model = model
        self.device = device
        self.prototype_mode = prototype_mode
        self.distance_type = distance_type
        self.prototypes_per_class = prototypes_per_class

        self.crop_ratio_h = crop_ratio_h
        self.crop_ratio_l=crop_ratio_l
        self.resize = resize

        # 模型参数和优化器
        self.model_params = list(self.model.parameters())
        self.mlr = lr
        self.optimizer_model = torch.optim.Adam(self.model_params, lr=self.mlr)

        # ✓ 改动<configured>：初始化全监督IMP组件
        if prototype_mode == 'imp':
            print(f"初始化全监督IMP模式：α={alpha}, σ={sigma_init}")

            # 使用全监督IMP生成器
            self.prototype_generator = SupervisedIMPPrototypeGenerator(
                alpha=alpha,
                sigma_init=sigma_init,
                device=device
            )

            # 使用全监督IMP损失
            self.loss_fn = SupervisedIMPLoss(distance_metric=distance_type)

            # IMP的可学习参数（如果有）
            self.learnable_params = []
            if hasattr(self.prototype_generator, 'parameters'):
                self.learnable_params = list(self.prototype_generator.parameters())

            print(f"✓ IMP模式初始化完成")

        elif prototype_mode == 'multi':
            # 使用原有的固定多原型方法
            from loss import MultiPrototypeGPNLoss
            self.loss_fn = MultiPrototypeGPNLoss(
                prototypes_per_class=self.prototypes_per_class,
                use_multi=_cfg_require('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.use_multi')
            )
            self.learnable_params = []
            print(f"使用固定多原型模式：每类{self.prototypes_per_class}个原型")

        else:  # 'single'
            # 使用单原型方法
            from loss import MultiPrototypeGPNLoss
            self.loss_fn = MultiPrototypeGPNLoss(use_multi=_cfg_require('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.__init__.use_multi__2'))
            self.learnable_params = []
            print(f"使用单原型模式")

        # 移动到设备
        self.model.to(device)
        self.loss_fn.to(device)

    def parameters(self):
        """返回所有可学习参数"""
        params = list(self.model.parameters())
        if hasattr(self, 'learnable_params') and self.learnable_params:
            params.extend(self.learnable_params)
        return params

    def save_model(self, save_path=None):
        """
        保存模型和损失函数的所有可学习参数
        """
        # 过滤掉不需要保存的参数
        save_path = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.save_model.save_path', save_path)
        filtered_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if not ("total_ops" in k or "total_params" in k)
        }

        checkpoint = {
            'model_state_dict': filtered_state_dict,
            'prototype_mode': self.prototype_mode,
            'distance_type': self.distance_type,
        }

        # 保存IMP生成器的参数
        if self.prototype_mode == 'imp' and hasattr(self, 'prototype_generator'):
            checkpoint['imp_alpha'] = self.prototype_generator.imp_algo.alpha
            checkpoint['imp_sigma'] = self.prototype_generator.imp_algo.sigma

        # 保存损失函数参数
        if hasattr(self.loss_fn, 'state_dict'):
            checkpoint['loss_fn_state_dict'] = self.loss_fn.state_dict()

        torch.save(checkpoint, save_path)
        print(f"模型已保存到 {save_path}")

    def load_model(self, model_path):
        """
        加载模型和损失函数参数
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)

        # 加载IMP参数
        if self.prototype_mode == 'imp' and 'imp_alpha' in checkpoint:
            self.prototype_generator.imp_algo.alpha = checkpoint['imp_alpha']
            self.prototype_generator.imp_algo.sigma = checkpoint['imp_sigma']
            print(f"IMP参数加载完成：α={checkpoint['imp_alpha']}, σ={checkpoint['imp_sigma']}")

        # 加载损失函数参数
        if 'loss_fn_state_dict' in checkpoint and hasattr(self.loss_fn, 'load_state_dict'):
            self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
            print(f"损失函数参数已加载")

        print(f"模型已从 {model_path} 加载")

    def _single_task_forward_imp(self,
                                support_signals,
                                support_labels,
                                query_signals,
                                query_labels):
        '\n        ✓ 改动<configured>：使用全监督IMP的单任务前向传播\n        \n        改动：\n        - 移除is_labeled参数\n        - 简化原型生成接口\n        '
        # 数据预处理
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

        # 前向传播提取特征
        support_v, support_s = self.model(support_signals)
        query_v, query_s = self.model(query_signals)

        # 标签重映射（将原始标签映射到<configured>, <configured>, <configured>, ...）
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

        # ✓ 改动<configured>：使用全监督IMP生成原型（无需is_labeled）
        all_prototypes, all_precision_matrices, prototypes_per_class = \
            self.prototype_generator.compute_imp_prototypes(
                support_v,                # [n_support, feature_dim]
                remapped_support_labels,  # [n_support] - 全都有标签
                n_ways,
                verbose=False
            )

        # ✓ 改动<configured>：使用全监督IMP损失计算
        loss, probabilities, distances = self.loss_fn(
            query_v,                  # [n_query, feature_dim]
            all_prototypes,           # List[Tensor]
            all_precision_matrices,   # List[Tensor]
            remapped_query_labels,    # [n_query]
            prototypes_per_class      # List[int]
        )

        # 计算准确率
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = (predictions == remapped_query_labels).float().mean()

        return loss, accuracy.item()

    def train_step_batch_imp(self, meta_batch, step_index):
        """
        IMP批处理训练步骤

        Args:
            meta_batch: (support_signals, support_labels, query_signals, query_labels)
            step_index: 当前步骤索引

        Returns:
            avg_loss: 平均损失
            avg_acc: 平均准确率
        """
        self.model.train()

        support_signals, support_labels, query_signals, query_labels = meta_batch
        batch_size = support_signals.size(0)

        total_loss = 0
        total_acc = 0

        # 清零梯度
        self.optimizer_model.zero_grad()

        # 处理批内所有任务
        for i in range(batch_size):
            loss, acc = self._single_task_forward_imp(
                support_signals[i],
                support_labels[i],
                query_signals[i],
                query_labels[i]
            )

            # 累积梯度（归一化）
            (loss / batch_size).backward()

            total_loss += loss.item()
            total_acc += acc

        # 梯度裁剪和参数更新
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=_cfg_require('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.train_step_batch_imp.max_norm'))
        self.optimizer_model.step()

        return total_loss / batch_size, total_acc / batch_size

    def train(self,
              train_loader,
              test_loader,
              epochs: int = None,
              n_ways: int = None,
              save_path: str = None,
              eval_interval: int = None,
              early_stop_patience: int = None):
        """
        元学习训练循环

        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            n_ways: 类别数（用于测试）
            save_path: 模型保存路径
            eval_interval: 评估间隔（每N个epoch评估一次）
            early_stop_patience: 早停耐心值
        """
        epochs = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.train.epochs', epochs)
        n_ways = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.train.n_ways', n_ways)
        save_path = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.train.save_path', save_path)
        eval_interval = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.train.eval_interval', eval_interval)
        early_stop_patience = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.train.early_stop_patience', early_stop_patience)
        self.model.train()

        # 学习率调度器
        scheduler_model = torch.optim.lr_scheduler.StepLR(
            self.optimizer_model,
            step_size=_cfg_require('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.train.step_size'),
            gamma=_cfg_require('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.train.gamma')
        )

        print(f"\n{'='*70}")
        print(f"开始元学习训练".center(70))
        print(f"{'='*70}")
        print(f"训练配置：")
        print(f"  - 总轮数：{epochs}")
        print(f"  - 每轮批次数：{len(train_loader)}")
        print(f"  - 批大小：{train_loader.batch_size}")
        print(f"  - 每轮训练任务数：{len(train_loader) * train_loader.batch_size}")
        print(f"  - 原型模式：{self.prototype_mode}")
        print(f"  - 距离度量：{self.distance_type}")
        print(f"{'='*70}\n")

        best_acc = 0
        early_stop = 0

        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            processed_batches = 0

            # 训练进度条
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, meta_batch in enumerate(pbar):
                    # 批处理训练步骤
                    loss, acc = self.train_step_batch_imp(meta_batch, batch_idx)

                    total_loss += loss
                    total_acc += acc
                    processed_batches += 1

                    # 更新进度条
                    pbar.set_postfix(
                        loss=f'{loss:.4f}',
                        acc=f'{acc:.4f}',
                        lr=f'{self.optimizer_model.param_groups[0]["lr"]:.6f}'
                    )

            # 学习率调度
            scheduler_model.step()

            # 计算平均指标
            avg_loss = total_loss / processed_batches
            avg_acc = total_acc / processed_batches

            print(f'Epoch {epoch + 1} 训练完成: avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.4f}')

            # 定期评估
            if (epoch + 1) % eval_interval == 0:
                print(f"\n开始测试评估...")
                test_acc = self.evaluate(
                    test_loader=test_loader,
                    n_ways=n_ways,
                    show_progress=True,
                    show_error_stats=True,
                    show_prototype_stats=(epoch + 1) % (eval_interval * 2) == 0  # 减少输出
                )
                print(f"测试准确率: {test_acc:.4f}")

                # 保存最佳模型
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.save_model(save_path)
                    print(f"✓ 最佳模型已保存（准确率：{best_acc:.4f}）")
                    early_stop = 0
                else:
                    early_stop += 1
                    print(f"未提升（连续 {early_stop} 次）")

                    # 早停检查
                    if early_stop >= early_stop_patience:
                        print(f"\n早停触发，训练结束")
                        break

        # 清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"\n{'='*70}")
        print(f"训练完成！".center(70))
        print(f"{'='*70}")
        print(f"最佳测试准确率: {best_acc:.4f}")
        print(f"模型已保存到: {save_path}")

    def evaluate(self,
                test_loader,
                n_ways: int,
                show_progress: bool = True,
                show_error_stats: bool = False,  # 改为False以匹配第一个方法
                return_stats: bool = False,      # 新增
                all_result: bool = False,        # 新增
                return_features: bool = False,   # 新增
                return_predictions: bool = False, # 新增
                show_prototype_stats: bool = False):
        '\n        ✓ 改动<configured>：评估模型性能（适配全监督IMP，统一返回格式）\n        \n        Args:\n            test_loader: 数据加载器\n            n_ways: 类别数\n            show_progress: 是否显示进度条\n            show_error_stats: 是否显示错误统计\n            return_stats: 是否返回详细统计信息\n            all_result: 返回单次任务指标\n            return_features: 返回特征向量 (用于t-SNE)\n            return_predictions: 返回详细预测 (用于混淆矩阵)\n            show_prototype_stats: 是否显示原型统计\n            \n        Returns:\n            avg_acc: 平均准确率（当所有return参数为False时）\n            result_pack: 包含多种指标的字典（当任一return参数为True时）\n        '
        self.model.eval()
        total_acc = 0
        task_count = 0

        # 容器
        all_task_metrics = []  # 用于 all_result

        # 新增容器
        collected_features = []
        collected_labels = []
        collected_preds = []
        collected_targets = []

        # 错误统计
        error_details = {}
        class_stats = {}

        # IMP原型统计
        prototype_statss = {}#单独测试用

        prototype_stats = {#集成测试用
            'per_class': {},  # 每个类的原型统计
        }


        iterator = tqdm(test_loader, desc="Evaluating") if show_progress else test_loader
        num=0#用于排除第一轮（warm-up）
        with torch.no_grad():
            for task_id, meta_task in enumerate(iterator):

                # ====== 新版数据抽取：包含 selected_classes ======
                support_signals, support_labels, query_signals, query_labels, selected_classes = meta_task

                # 移动到设备并预处理
                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                # 数据增强：裁剪
                if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
                    processed_support_list = []
                    for signal in support_signals.unbind(0):
                        processed_support_list.append(
                            crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                        )
                    support_signals = torch.stack(processed_support_list, dim=0)

                    processed_query_list = []
                    for signal in query_signals.unbind(0):
                        processed_query_list.append(
                            crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                        )
                    query_signals = torch.stack(processed_query_list, dim=0)

                torch.cuda.synchronize()
                task_start_time1=time.time()

                # 前向传播
                support_v, support_s = self.model(support_signals)
                query_v, query_s = self.model(query_signals)

                torch.cuda.synchronize()
                task_end_time1=time.time()

                # ====== 新版标签映射逻辑 ======
                # <configured>. 从 __getitem__ 获得本任务的原始标签列表
                original_class_ids_list = selected_classes
                if not isinstance(original_class_ids_list, list):
                    original_class_ids_list = [original_class_ids_list]

                remapped_to_original = {
                    i: original_id
                    for i, original_id in enumerate(original_class_ids_list)
                }

                # <configured>. 准备用于全局统计的【原始全局标签】
                query_local_indices_np = query_labels.cpu().numpy()

                original_query_labels = np.array([
                    remapped_to_original[idx]
                    for idx in query_local_indices_np
                ])
                # original_query_labels：用于 t-SNE 和全局指标的真实标签

                # <configured>. 准备用于 Loss 计算的【相对标签】
                remapped_support_labels = support_labels
                remapped_query_labels = query_labels

                torch.cuda.synchronize()
                task_start_time2=time.time()
                # ✓ 改动<configured>：IMP原型生成（无需is_labeled）
                k_shot = len(support_labels) // n_ways

                if self.prototype_mode == 'imp':
                    all_prototypes, all_precision_matrices, prototypes_per_class = \
                        self.prototype_generator.compute_imp_prototypes(
                            support_v,
                            remapped_support_labels,  # ✓ 无需is_labeled参数
                            n_ways,
                            verbose=False
                        )
                    torch.cuda.synchronize()
                    task_end_time2=time.time()

                    # 收集原型统计信息
                    if show_prototype_stats:
                        for i, original_class in enumerate(original_class_ids_list):
                            n_prototypes = prototypes_per_class[i]

                            if original_class not in prototype_statss:
                                prototype_statss[original_class] = {
                                    'total_tasks': 0,
                                    'total_prototypes': 0,
                                    'min_prototypes': float('inf'),
                                    'max_prototypes': 0
                                }

                            prototype_statss[original_class]['total_tasks'] += 1
                            prototype_statss[original_class]['total_prototypes'] += n_prototypes
                            prototype_statss[original_class]['min_prototypes'] = min(
                                prototype_statss[original_class]['min_prototypes'], n_prototypes
                            )
                            prototype_statss[original_class]['max_prototypes'] = max(
                                prototype_statss[original_class]['max_prototypes'], n_prototypes
                            )

                    for i, original_class in enumerate(original_class_ids_list):
                        n_prototypes = prototypes_per_class[i]

                        if original_class not in prototype_stats['per_class']:
                            prototype_stats['per_class'][original_class] = {
                                'total_tasks': 0,
                                'active_sum': 0,  # IMP中active_sum就是实际生成的原型数
                                'initial_prototypes': k_shot  # 初始样本数（理论最大原型数）
                            }

                        prototype_stats['per_class'][original_class]['total_tasks'] += 1
                        prototype_stats['per_class'][original_class]['active_sum'] += n_prototypes

                    torch.cuda.synchronize()
                    task_start_time3=time.time()
                    # 使用IMP损失计算
                    _, probabilities, _ = self.loss_fn(
                        query_v,
                        all_prototypes,
                        all_precision_matrices,
                        remapped_query_labels,
                        prototypes_per_class
                    )

                elif self.prototype_mode == 'multi':
                    # 使用固定多原型
                    _, probabilities = self.loss_fn(
                        support_v, support_s, query_v,
                        remapped_support_labels, remapped_query_labels
                    )

                else:  # 'single'
                    # 使用单原型
                    _, probabilities = self.loss_fn(
                        support_v, support_s, query_v,
                        remapped_support_labels, remapped_query_labels
                    )

                # 计算准确率
                predictions = torch.argmax(probabilities, dim=1)
                accuracy = (predictions == remapped_query_labels).float().mean().item()
                total_acc += accuracy
                task_count += 1

                torch.cuda.synchronize()
                task_end_time3 = time.time()

                if return_features:
                    # 收集 Query set 的特征
                    collected_features.append(query_v.cpu().numpy())
                    # 收集 Query set 对应的原始标签 (用于着色)
                    collected_labels.append(original_query_labels.flatten())

                if return_predictions:
                    # 需要映射回原始类别ID以便全局统计
                    preds_cpu = predictions.cpu().numpy()

                    # 将 <configured>-(N-<configured>) 的相对标签映射回原始数据集的绝对标签
                    real_preds = [remapped_to_original[p] for p in preds_cpu]
                    real_targets = [remapped_to_original[t] for t in query_local_indices_np]

                    collected_preds.extend(real_preds)
                    collected_targets.extend(real_targets)

                if all_result:
                    predictions_cpu = predictions.cpu().numpy()
                    labels_cpu = remapped_query_labels.cpu().numpy()

                    task_acc = accuracy
                    # 使用 'macro' 平均，公平对待每个类别
                    task_f1 = f1_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)
                    task_precision = precision_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)
                    task_recall = recall_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)
                    task_duration_ms = (task_end_time1 - task_start_time1+task_end_time2 - task_start_time2+task_end_time3 - task_start_time3) * _cfg_require('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.evaluate.size_or_budget')
                    if num<1:
                        task_duration_ms=0
                    print('h',task_duration_ms)
                    num+=1

                    all_task_metrics.append({
                        'acc': task_acc,
                        'f1_macro': task_f1,
                        'precision_macro': task_precision,
                        'recall_macro': task_recall,
                        'time_ms': task_duration_ms,
                        'task_id': task_id,
                    })

                # ========== 错误统计 ==========
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
                            if key not in error_details:
                                error_details[key] = 0
                            error_details[key] += 1

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

                    # 处理正确预测
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
        avg_acc = total_acc / task_count if task_count > 0 else 0

        # ========== 打印错误统计 ==========
        if show_error_stats and error_details:
            print("\n" + "=" * 60)
            print("错误识别统计")
            print("=" * 60)

            sorted_errors = sorted(
                error_details.items(),
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

        # ========== 打印原型统计 ==========
        if show_prototype_stats and self.prototype_mode == 'imp' and prototype_statss:
            print(f"\n{'='*70}")
            print("IMP原型统计".center(70))
            print(f"{'='*70}")
            print(f"{'类别':<10} {'平均原型数':<15} {'最小':<10} {'最大':<10}")
            print("-" * 70)

            for class_id in sorted(prototype_statss.keys()):
                stats = prototype_statss[class_id]
                avg_protos = stats['total_prototypes'] / stats['total_tasks']
                print(f"{class_id:<10} {avg_protos:<15.2f} {stats['min_prototypes']:<10} {stats['max_prototypes']:<10}")

            # 全局统计
            total_tasks = sum(s['total_tasks'] for s in prototype_statss.values())
            total_protos = sum(s['total_prototypes'] for s in prototype_statss.values())
            global_avg = total_protos / total_tasks if total_tasks > 0 else 0
            print("-" * 70)
            print(f"全局平均：{global_avg:.2f} 个原型/类")

        # ========== 构建返回结果 ==========
        result_pack = {'avg_acc': avg_acc}

        if return_stats:
            # 可以添加额外的统计信息
            eval_stats = {
                'error_details': error_details,
                'class_stats': class_stats,
                'prototype_stats': prototype_stats if self.prototype_mode == 'imp' else None
            }
            result_pack['stats'] = eval_stats

        if all_result:
            result_pack['all_task_metrics'] = all_task_metrics
            # 新增：计算平均指标
            result_pack['avg_f1_macro'] = np.mean([m['f1_macro'] for m in all_task_metrics])
            result_pack['avg_precision_macro'] = np.mean([m['precision_macro'] for m in all_task_metrics])
            result_pack['avg_recall_macro'] = np.mean([m['recall_macro'] for m in all_task_metrics])
            result_pack['avg_time_ms'] = np.mean([m['time_ms'] for m in all_task_metrics if m['time_ms'] > 0])

        if return_features:
            result_pack['features'] = np.concatenate(collected_features, axis=0)
            result_pack['feature_labels'] = np.concatenate(collected_labels, axis=0)

        if return_predictions:
            result_pack['all_predictions'] = collected_preds
            result_pack['all_targets'] = collected_targets
        if self.prototype_mode!='single':
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
                prototype_summary['merge_ratio'] = 1 - (total_active / total_initial) if total_initial > 0 else 0

            result_pack['prototype_summary'] = prototype_summary
        # ========== 根据参数组合返回 (保持向后兼容) ==========
        if not any([all_result, return_features, return_predictions, return_stats]):
            return avg_acc
        else:
            return result_pack

    def compare_methods(self, test_loader, n_ways: int = None, n_tasks: int = None):
        """
        对比不同原型方法的性能

        Args:
            test_loader: 测试数据加载器
            n_ways: 类别数
            n_tasks: 评估任务数

        Returns:
            results: Dict[str, float] 各方法的准确率
        """
        n_ways = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.compare_methods.n_ways', n_ways)
        n_tasks = _cfg_resolve('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.compare_methods.n_tasks', n_tasks)
        print(f"\n{'='*70}")
        print("原型方法对比".center(70))
        print(f"{'='*70}\n")

        results = {}
        original_mode = self.prototype_mode

        # 测试不同模式
        for mode in ['single', 'multi', 'imp']:
            print(f"测试 {mode.upper()} 模式...")
            self.prototype_mode = mode

            # 重新初始化相应组件
            if mode == 'imp':
                if not hasattr(self, 'prototype_generator'):
                    self.prototype_generator = SupervisedIMPPrototypeGenerator(
                        alpha=_cfg_require('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.compare_methods.alpha'), sigma_init=_cfg_require('GPN_V3_IMP_onlyTrainer.py.GPNTrainerWithIMP.compare_methods.sigma_init'), device=self.device
                    )
                self.loss_fn = SupervisedIMPLoss(distance_metric=self.distance_type)

            # 评估
            acc = self.evaluate(
                test_loader,
                n_ways,
                show_progress=False,
                show_error_stats=False,
                show_prototype_stats=False
            )

            results[mode] = acc
            print(f"  准确率: {acc*100:.2f}%\n")

        # 恢复原始模式
        self.prototype_mode = original_mode

        # 打印对比结果
        print(f"{'='*70}")
        print("对比结果".center(70))
        print(f"{'='*70}")
        print(f"{'方法':<15} {'准确率':<15} {'相对提升':<15}")
        print("-" * 70)

        baseline_acc = results.get('single', 0)
        for mode in ['single', 'multi', 'imp']:
            if mode in results:
                acc = results[mode]
                improvement = (acc - baseline_acc) * 100
                print(f"{mode.upper():<15} {acc*100:<15.2f} {improvement:+.2f}%")

        print(f"{'='*70}\n")

        return results


# ============ 使用示例 ============

def usage_example():
    """
    使用示例
    """
    print("="*70)
    print("GPNTrainerWithIMP 使用示例".center(70))
    print("="*70)

    # 模拟模型和数据加载器
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(_cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.DummyModel.__init__.Conv1d_arg0'), _cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.DummyModel.__init__.Conv1d_arg1'), _cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.DummyModel.__init__.Conv1d_arg2'))
            self.fc = nn.Linear(_cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.DummyModel.__init__.Linear_arg0'), _cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.DummyModel.__init__.Linear_arg1'))

        def forward(self, x):
            x = self.conv(x)
            x = x.mean(dim=2)
            v = self.fc(x)
            s = torch.ones_like(v)
            return v, s

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DummyModel()

    print("\n【初始化训练器】")
    print("-" * 70)

    # 初始化训练器
    trainer = GPNTrainerWithIMP(
        model=model,
        device=device,
        alpha=_cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.alpha'),           # CRP浓度参数
        sigma_init=_cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.sigma_init'),      # 初始cluster方差
        prototype_mode=_cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.prototype_mode'), # 使用IMP模式
        distance_type=_cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.distance_type'),    # 欧氏距离
        lr=_cfg_require('GPN_V3_IMP_onlyTrainer.py.usage_example.lr')
    )

    print("✓ 训练器初始化完成")
    print(f"  - 原型模式: {trainer.prototype_mode}")
    print(f"  - 距离度量: {trainer.distance_type}")
    print(f"  - 学习率: {trainer.mlr}")

    print("\n【训练流程】")
    print("-" * 70)
    print("\n    # <configured>. 准备数据加载器\n    train_loader = create_meta_loader(...)\n    test_loader = create_meta_loader(...)\n    \n    # <configured>. 训练模型\n    trainer.train(\n        train_loader=train_loader,\n        test_loader=test_loader,\n        epochs=<configured>,\n        n_ways=<configured>,\n        save_path='models/best_model.pth'\n    )\n    \n    # <configured>. 评估模型\n    test_acc = trainer.evaluate(\n        test_loader=test_loader,\n        n_ways=<configured>,\n        show_progress=True,\n        show_error_stats=True,\n        show_prototype_stats=True\n    )\n    \n    # <configured>. 对比不同方法\n    results = trainer.compare_methods(\n        test_loader=test_loader,\n        n_ways=<configured>,\n        n_tasks=<configured>\n    )\n    ")

    print("\n【关键改动总结】")
    print("-" * 70)
    print("✓ 移除 sigma_l 和 sigma_u，统一为 sigma_init")
    print("✓ 移除所有 is_labeled 相关参数")
    print("✓ 使用 SupervisedIMPPrototypeGenerator")
    print("✓ 使用 SupervisedIMPLoss")
    print("✓ 简化原型生成接口")
    print("✓ 保持与原训练框架兼容")


if __name__ == "__main__":
    usage_example()
