from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import os
import sys
import torch
import numpy as np
import random
import gc
import json
import time
from datetime import datetime
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from draw_all_model_pic import *


# ==================== 核心评估函数 ====================


def test_multiple_models(
    test_dataset,
    model_paths,
    trainer_configs,
    model_initializers,
    trainer_class=None,
    trainer_classes=None,
    n_way=None,
    k_shots=None,
    n_trials=None,
    q_query=None,
    device=None,
    seed=None,
    save_dir=None,
    collect_visualizations=False  # 新增：是否收集可视化数据
):
    """
    测试多个模型

    Args:
        ... (其他参数同前)
        collect_visualizations: 是否收集用于可视化的数据（特征、预测等）
    """

    # 创建输出目录
    n_way = _cfg_resolve('run_eval_example.py.test_multiple_models.n_way', n_way)
    k_shots = _cfg_resolve('run_eval_example.py.test_multiple_models.k_shots', k_shots)
    n_trials = _cfg_resolve('run_eval_example.py.test_multiple_models.n_trials', n_trials)
    q_query = _cfg_resolve('run_eval_example.py.test_multiple_models.q_query', q_query)
    device = _cfg_resolve('run_eval_example.py.test_multiple_models.device', device)
    seed = _cfg_resolve('run_eval_example.py.test_multiple_models.seed', seed)
    save_dir = _cfg_resolve('run_eval_example.py.test_multiple_models.save_dir', save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(save_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # 重定义Trainer类列表
    if trainer_classes is None:
        if trainer_class is None:
            from GPN_V4_adaMulti_test import GPNTrainer
            trainer_class = GPNTrainer
        trainer_classes = [trainer_class] * len(model_paths)

    # 存储结果
    all_results = {}
    results_rows = []

    # ========== 新增：可视化数据容器 ==========
    visualization_data = {
        'features': [],           # 收集特征（用于t-SNE）
        'feature_labels': [],     # 特征对应的标签
        'predictions': [],        # 收集预测（用于混淆矩阵）
        'targets': [],            # 真实标签
        'class_stats': {},        # 每类准确率统计
        'model_names': [],        # 记录每个数据点对应的模型
        'pred_model_names': [],   # 新增：预测对应的模型名
        'pred_k_shots': [],        # 新增：预测对应的 k_shot
        'prototype_stats': []  # 新增：收集原型统计

    }


    print(f"\n{'='*60}")
    print(f"Multi-Model Evaluation")
    print(f"Models: {len(model_paths)}")
    print(f"K-shots: {k_shots}")
    print(f"Trials per config: {n_trials}")
    print(f"{'='*60}")

    # 遍历每个模型
    for idx, (model_info, config, initializer, TrainerClass) in enumerate(
        zip(model_paths, trainer_configs, model_initializers, trainer_classes)
    ):
        model_name = model_info['name']
        model_path = model_info['path']

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(model_paths)}] Evaluating: {model_name}")
        print(f"Model path: {model_path}")
        print(f"{'='*60}")

        # 初始化模型
        model = initializer()
        model = model.to(device)

        # 初始化Trainer
        trainer = TrainerClass(
            model=model,
            device=device,
            **config
        )

        # 加载权重
        if os.path.exists(model_path):
            trainer.load_model(model_path)
            print(f"✓ Loaded weights from: {model_path}")
        else:
            print(f"✗ WARNING: Model not found: {model_path}")

        model_results = {}

        # 遍历不同的shot配置
        for k_shot in k_shots:
            config_key = f'{n_way}w{k_shot}s'
            print(f"\n  Testing {config_key}...")

            # 设置随机种子
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            # 存储当前配置的所有单次任务指标
            all_trial_metrics = []
            accuracies = []

            # 进度条
            pbar = tqdm(range(n_trials), desc=f"{model_name} {config_key}")

            for trial in pbar:
                # 创建元任务
                from data_loader import MetaDataset
                meta_test = MetaDataset(
                    test_dataset,
                    n_way,
                    k_shot,
                    q_query=q_query,
                    num_tasks=_cfg_require('run_eval_example.py.test_multiple_models.num_tasks'),
                    seed=seed+random.randint(0, _cfg_require('run_eval_example.py.test_multiple_models.size_or_budget'))
                )
                loader = DataLoader(meta_test, batch_size=_cfg_require('run_eval_example.py.test_multiple_models.batch_size'), shuffle=True)

                acc = 0
                result = None

                # ========== 标准评估（支持可视化数据收集）==========
                # 决定是否在最后一个trial收集可视化数据
                is_last_trial = (trial == n_trials - 1)
                should_collect = collect_visualizations and is_last_trial

                result = trainer.evaluate(
                    loader,
                    n_way,
                    show_progress=False,
                    show_error_stats=False,
                    return_stats=False,
                )

                acc=result

                # 在收集数据处修改
                if k_shot==10 and should_collect:
                    if 'features' in result:
                        visualization_data['features'].append(result['features'])
                        visualization_data['feature_labels'].append(result['feature_labels'])
                        visualization_data['model_names'].extend(
                            [model_name] * len(result['feature_labels'])
                        )

                    if 'all_predictions' in result:
                        # ===== 修改：添加模型和k_shot标识 =====
                        n_samples = len(result['all_predictions'])

                        visualization_data['predictions'].extend(result['all_predictions'])
                        visualization_data['targets'].extend(result['all_targets'])
                        visualization_data['pred_model_names'].extend([model_name] * n_samples)
                        visualization_data['pred_k_shots'].extend([k_shot] * n_samples)

                accuracies.append(acc)

                pbar.set_postfix({
                    'mean_acc': f"{np.mean(accuracies):.4f}",
                    'current': f"{acc:.4f}"
                })

                del meta_test, loader

                if (trial + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # ========== 统计所有指标 ==========
            if all_trial_metrics:
                metrics_df = pd.DataFrame(all_trial_metrics)

                # 准确率统计
                mean_acc = metrics_df['acc'].mean()
                std_acc = metrics_df['acc'].std()
                ci95_acc = 1.96 * std_acc / np.sqrt(n_trials)

                # F1-Score 统计
                mean_f1 = metrics_df['f1_macro'].mean()
                std_f1 = metrics_df['f1_macro'].std()
                ci95_f1 = 1.96 * std_f1 / np.sqrt(n_trials)

                # Precision 统计
                mean_precision = metrics_df['precision_macro'].mean()
                std_precision = metrics_df['precision_macro'].std()
                ci95_precision = 1.96 * std_precision / np.sqrt(n_trials)

                # Recall 统计
                mean_recall = metrics_df['recall_macro'].mean()
                std_recall = metrics_df['recall_macro'].std()
                ci95_recall = 1.96 * std_recall / np.sqrt(n_trials)

                # 时间统计
                mean_time_ms = metrics_df['time_ms'].mean()
                std_time_ms = metrics_df['time_ms'].std()

            else:
                # 如果没有详细指标，使用 accuracies 列表统计
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                ci95_acc = 1.96 * std_acc / np.sqrt(n_trials)
                mean_f1, std_f1, ci95_f1 = 0, 0, 0
                mean_precision, std_precision, ci95_precision = 0, 0, 0
                mean_recall, std_recall, ci95_recall = 0, 0, 0
                mean_time_ms, std_time_ms = 0, 0

            # 保存详细结果到字典
            model_results[f'{k_shot}-shot'] = {
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'ci95_acc': ci95_acc,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'ci95_f1': ci95_f1,
                'mean_precision': mean_precision,
                'std_precision': std_precision,
                'ci95_precision': ci95_precision,
                'mean_recall': mean_recall,
                'std_recall': std_recall,
                'ci95_recall': ci95_recall,
                'mean_time_ms': mean_time_ms,
                'std_time_ms': std_time_ms,
            }

            # 更新结果行
            results_rows.append({
                'Model': model_name,
                'K-shot': k_shot,
                'Accuracy': mean_acc,
                'Std_Acc': std_acc,
                'CI95_Acc': ci95_acc,
                'F1_Macro': mean_f1,
                'Std_F1': std_f1,
                'CI95_F1': ci95_f1,
                'Precision_Macro': mean_precision,
                'Recall_Macro': mean_recall,
                'Time_ms': mean_time_ms,
                'Accuracy_str': f"{mean_acc*100:.1f}±{ci95_acc*100:.1f}",
                'F1_str': f"{mean_f1*100:.1f}±{ci95_f1*100:.1f}"
            })

            print(f"  Result (Acc): {mean_acc*100:.2f}% ± {ci95_acc*100:.2f}%")
            print(f"  Result (F1):  {mean_f1*100:.2f}% ± {ci95_f1*100:.2f}%")
            print(f"  Avg Time:     {mean_time_ms:.2f} ms")

        # 保存当前模型的结果
        all_results[model_name] = model_results

        # 清理
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 创建DataFrame
    df = pd.DataFrame(results_rows)

    # ========== 整合可视化数据包 ==========
    data_pack = {
        'model_results': all_results,  # 保留原有的详细结果字典
    }

    # 合并特征数据
    if visualization_data['features']:
        data_pack['features'] = np.concatenate(visualization_data['features'], axis=0)
        data_pack['feature_labels'] = np.concatenate(visualization_data['feature_labels'], axis=0)

    # 合并预测数据
    if visualization_data['predictions']:
        data_pack['predictions'] = np.array(visualization_data['predictions'])
        data_pack['targets'] = np.array(visualization_data['targets'])


    if visualization_data['prototype_stats']:
        prototype_df = pd.DataFrame(visualization_data['prototype_stats'])
        data_pack['prototype_stats_df'] = prototype_df

        print(f"[INFO] Collected prototype stats: {len(prototype_df)} records")
        print(prototype_df[['model', 'k_shot', 'global_avg_active', 'merge_ratio']])
    # 计算每类准确率（如果有预测数据）
    if 'predictions' in data_pack and 'targets' in data_pack:
        from sklearn.metrics import classification_report
        class_report = classification_report(
            data_pack['targets'],
            data_pack['predictions'],
            output_dict=True,
            zero_division=0
        )
        data_pack['class_stats'] = class_report

    # ========== 保存结果（修正参数）==========
    save_results(df, data_pack, output_dir, n_way, k_shots)

    # 打印汇总表
    print("\n" + "="*60)
    print("Summary Table (Accuracy):")
    print("="*60)
    pivot_acc = df.pivot(index='Model', columns='K-shot', values='Accuracy_str')
    print(pivot_acc.to_string())

    print("\n" + "="*60)
    print("Summary Table (F1-Macro):")
    print("="*60)
    pivot_f1 = df.pivot(index='Model', columns='K-shot', values='F1_str')
    print(pivot_f1.to_string())

    return df, data_pack  # ✅ 同时返回 DataFrame 和数据包

#针对单模型收集可视化数据
def collect_visualization_data(
    test_dataset,
    trainer,#自行预加载trainer并导入
    n_way=None,
    k_shot=None,
    n_tasks=None,  # 收集<configured>个任务的数据足以进行可视化
    save_path=None
):
    '\n    收集用于 t-SNE (Fig <configured>) 和 混淆矩阵 (Fig <configured>) 的数据\n    '
    n_way = _cfg_resolve('run_eval_example.py.collect_visualization_data.n_way', n_way)
    k_shot = _cfg_resolve('run_eval_example.py.collect_visualization_data.k_shot', k_shot)
    n_tasks = _cfg_resolve('run_eval_example.py.collect_visualization_data.n_tasks', n_tasks)
    save_path = _cfg_resolve('run_eval_example.py.collect_visualization_data.save_path', save_path)
    print(f"Collecting visualization data for {model_path}...")


    # 构造 DataLoader
    from data_loader import MetaDataset
    meta_test = MetaDataset(
        test_dataset,
        n_way,
        k_shot,
        q_query=q_query,
        num_tasks=_cfg_require('run_eval_example.py.collect_visualization_data.num_tasks'),
        seed=random.randint(0, _cfg_require('run_eval_example.py.collect_visualization_data.size_or_budget'))
    )
    loader = DataLoader(meta_test, batch_size=_cfg_require('run_eval_example.py.collect_visualization_data.batch_size'),shuffle=True)

    # 调用 evaluate 获取特征和预测
    result = trainer.evaluate(
        loader, n_way,
        show_progress=True,
        return_features=True,    # 开启特征收集
        return_predictions=True, # 开启预测收集
        all_result=False
    )

    # 保存数据
    np.savez(
        save_path,
        features=result['features'],          # [N_samples, Feature_dim]
        feature_labels=result['feature_labels'], # [N_samples]
        predictions=result['all_predictions'],   # [Total_samples]
        targets=result['all_targets']            # [Total_samples]
    )
    print(f"Visualization data saved to {save_path}")
