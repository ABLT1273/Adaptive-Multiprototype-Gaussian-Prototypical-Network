'\n多模型串行评估框架\n==================\n支持不同模型类、Trainer类、配置参数的统一评估\n\n功能:\n<configured>. 串行测评多个模型\n<configured>. 为不同模型配置不同参数\n<configured>. 统一的评估接口\n<configured>. 结果汇总保存（JSON + CSV + LaTeX）\n<configured>. 自动生成对比图表\n'
from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format

import os
import sys
import json
import time
import random
import gc
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Callable, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# 设置绘图风格
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


# ==================== 核心评估类 ====================

class MultiModelEvaluator:
    """
    多模型评估器

    支持:
    - 不同的模型类 (GPN_Optimized, GPN_Kmeans, GPN_IMP等)
    - 不同的Trainer类 (GPNTrainer, IMPTrainer等)
    - 不同的配置参数
    - 统一的评估接口
    """

    def __init__(self, device: str = None, seed: int = None):
        device = _cfg_resolve('multi_model_evaluation.py.MultiModelEvaluator.__init__.device', device)
        seed = _cfg_resolve('multi_model_evaluation.py.MultiModelEvaluator.__init__.seed', seed)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.set_seed(seed)

    def set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def evaluate_single_model(
        self,
        trainer,
        test_dataset,
        n_way: int,
        k_shot: int,
        q_query: int,
        n_trials: int,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        评估单个模型

        Args:
            trainer: 已初始化并加载权重的Trainer实例
            test_dataset: 测试数据集
            n_way: N-way分类
            k_shot: K-shot
            q_query: 查询集大小
            n_trials: 评估轮数
            model_name: 模型名称（用于显示）

        Returns:
            评估结果字典
        """
        from amgpn.data.episodes import MetaDataset

        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        all_predictions = []
        all_labels = []
        prototype_counts = []
        inference_times = []

        trainer.model.eval()

        pbar = tqdm(range(n_trials), desc=f"{model_name} {n_way}w{k_shot}s")

        for trial in pbar:
            # 创建单个episode
            meta_test = MetaDataset(
                test_dataset, n_way, k_shot, q_query,
                num_tasks=_cfg_require('multi_model_evaluation.py.MultiModelEvaluator.evaluate_single_model.num_tasks'), seed=self.seed + trial
            )
            loader = DataLoader(meta_test, batch_size=_cfg_require('multi_model_evaluation.py.MultiModelEvaluator.evaluate_single_model.batch_size'))

            # 评估单个episode
            start_time = time.time()
            trial_result = self._evaluate_single_episode(
                trainer, loader, n_way
            )
            inference_time = (time.time() - start_time) * _cfg_require('multi_model_evaluation.py.MultiModelEvaluator.evaluate_single_model.size_or_budget')  # ms

            accuracies.append(trial_result['accuracy'])
            f1_scores.append(trial_result['f1_macro'])
            precisions.append(trial_result['precision'])
            recalls.append(trial_result['recall'])
            all_predictions.extend(trial_result['predictions'])
            all_labels.extend(trial_result['labels'])
            inference_times.append(inference_time)

            if 'avg_prototypes' in trial_result:
                prototype_counts.append(trial_result['avg_prototypes'])

            # 更新进度条
            pbar.set_postfix({
                'acc': f"{np.mean(accuracies):.4f}",
                'f1': f"{np.mean(f1_scores):.4f}"
            })

            # 清理
            del meta_test, loader
            if (trial + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 计算统计量
        n = len(accuracies)
        result = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'ci95': 1.96 * np.std(accuracies) / np.sqrt(n)
            },
            'f1_macro': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'ci95': 1.96 * np.std(f1_scores) / np.sqrt(n)
            },
            'precision': {
                'mean': np.mean(precisions),
                'std': np.std(precisions),
                'ci95': 1.96 * np.std(precisions) / np.sqrt(n)
            },
            'recall': {
                'mean': np.mean(recalls),
                'std': np.std(recalls),
                'ci95': 1.96 * np.std(recalls) / np.sqrt(n)
            },
            'inference_time_ms': {
                'mean': np.mean(inference_times),
                'std': np.std(inference_times)
            },
            'n_trials': n_trials,
            'n_way': n_way,
            'k_shot': k_shot
        }

        if prototype_counts:
            result['avg_prototypes'] = {
                'mean': np.mean(prototype_counts),
                'std': np.std(prototype_counts)
            }

        # 计算混淆矩阵
        unique_classes = sorted(set(all_labels))
        result['confusion_matrix'] = confusion_matrix(
            all_labels, all_predictions, labels=unique_classes
        ).tolist()
        result['class_labels'] = unique_classes

        return result

    def _evaluate_single_episode(
        self,
        trainer,
        loader,
        n_ways: int
    ) -> Dict[str, Any]:
        """评估单个episode"""

        predictions_list = []
        labels_list = []
        prototype_count = None

        with torch.no_grad():
            for meta_task in loader:
                support_signals, support_labels, query_signals, query_labels = meta_task

                # 数据预处理 - 适配不同的数据格式
                if support_signals.dim() == 4:  # [batch, n_samples, H, W]
                    support_signals = support_signals.permute(1, 0, 2, 3)
                    query_signals = query_signals.permute(1, 0, 2, 3)

                support_signals = support_signals.to(self.device).float()
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float()
                query_labels = query_labels.to(self.device).flatten()

                # 添加通道维度（如果需要）
                if support_signals.dim() == 3:
                    support_signals = support_signals.unsqueeze(1)
                    query_signals = query_signals.unsqueeze(1)

                # 数据增强
                if hasattr(trainer, 'crop_ratio_h') and (trainer.crop_ratio_h != 0 or trainer.crop_ratio_l != 0):
                    try:
                        from amgpn.data.preprocessing import crop_and_rescale_symmetric
                        support_signals = torch.stack([
                            crop_and_rescale_symmetric(s, trainer.crop_ratio_h,
                                                       trainer.crop_ratio_l, trainer.resize)
                            for s in support_signals.unbind(0)
                        ])
                        query_signals = torch.stack([
                            crop_and_rescale_symmetric(s, trainer.crop_ratio_h,
                                                       trainer.crop_ratio_l, trainer.resize)
                            for s in query_signals.unbind(0)
                        ])
                    except ImportError:
                        pass

                # 特征提取
                support_v, support_s = trainer.model(support_signals)
                query_v, query_s = trainer.model(query_signals)

                # 标签映射
                unique_labels = torch.unique(support_labels)
                label_mapping = {l.item(): i for i, l in enumerate(unique_labels)}
                remapped_to_original = {i: l.item() for i, l in enumerate(unique_labels)}

                remapped_support_labels = torch.tensor(
                    [label_mapping[l.item()] for l in support_labels], device=self.device
                )
                remapped_query_labels = torch.tensor(
                    [label_mapping[l.item()] for l in query_labels], device=self.device
                )

                # 根据trainer类型选择不同的预测方式
                predictions = self._get_predictions(
                    trainer, support_v, support_s, query_v,
                    remapped_support_labels, remapped_query_labels, n_ways
                )

                # 获取原型数量（如果适用）
                if hasattr(trainer, 'use_multi') and trainer.use_multi:
                    prototype_count = self._get_prototype_count(
                        trainer, support_v, support_s, n_ways
                    )

                # 收集结果
                for pred, true_label in zip(predictions.cpu().numpy(),
                                           remapped_query_labels.cpu().numpy()):
                    predictions_list.append(remapped_to_original[pred])
                    labels_list.append(remapped_to_original[true_label])

        predictions_arr = np.array(predictions_list)
        labels_arr = np.array(labels_list)

        result = {
            'accuracy': np.mean(predictions_arr == labels_arr),
            'f1_macro': f1_score(labels_arr, predictions_arr, average='macro', zero_division=0),
            'precision': precision_score(labels_arr, predictions_arr, average='macro', zero_division=0),
            'recall': recall_score(labels_arr, predictions_arr, average='macro', zero_division=0),
            'predictions': predictions_list,
            'labels': labels_list
        }

        if prototype_count is not None:
            result['avg_prototypes'] = prototype_count

        return result

    def _get_predictions(
        self,
        trainer,
        support_v,
        support_s,
        query_v,
        remapped_support_labels,
        remapped_query_labels,
        n_ways: int
    ) -> torch.Tensor:
        """根据trainer类型获取预测结果"""

        k_shot = support_v.shape[0] // n_ways

        # 检查是否是多原型模式
        use_multi = getattr(trainer, 'use_multi', False)

        if use_multi:
            # 多原型模式
            prototypes = support_v
            precision_matrices = trainer.compute_precision_matrices_from_support(support_s)

            # 调用loss_fn获取输出
            output = trainer.loss_fn(
                query_v, prototypes, precision_matrices,
                remapped_query_labels, epoch=getattr(trainer, 'current_epoch', 0)
            )

            logits = -output['distances']
            predictions = torch.argmax(logits, dim=1)
        else:
            # 单原型模式
            prototypes, precision_matrices = trainer.compute_gaussian_prototypes(
                support_v, support_s, remapped_support_labels, n_ways
            )

            distances = trainer.loss_fn.distance_metric(query_v, prototypes, precision_matrices)
            predictions = torch.argmin(distances, dim=1)

        return predictions

    def _get_prototype_count(
        self,
        trainer,
        support_v,
        support_s,
        n_ways: int
    ) -> float:
        """获取平均原型数量"""

        k_shot = support_v.shape[0] // n_ways
        prototypes = support_v
        precision_matrices = trainer.compute_precision_matrices_from_support(support_s)

        # 创建mask并合并
        prototype_mask = torch.ones(n_ways, k_shot, dtype=torch.bool, device=self.device)

        _, _, prototype_mask, _ = trainer.loss_fn._merge_prototypes_in_task_optimized(
            prototypes, precision_matrices, prototype_mask, n_ways, k_shot
        )

        return prototype_mask.sum().item() / n_ways


def test_multiple_models(
    test_dataset,
    model_configs: List[Dict[str, Any]],
    n_way: int = None,
    k_shots: List[int] = None,
    n_trials: int = None,
    q_query: int = None,
    device: str = None,
    seed: int = None,
    save_dir: str = None
) -> pd.DataFrame:
    """
    测试多个模型

    Args:
        test_dataset: 测试数据集
        model_configs: 模型配置列表，每个配置包含:
            - name: 模型名称
            - model_path: 模型权重路径
            - model_class: 模型类
            - trainer_class: Trainer类
            - trainer_kwargs: Trainer初始化参数
        n_way: N-way分类
        k_shots: K-shot列表
        n_trials: 每个配置的评估轮数
        q_query: 查询集大小
        device: 计算设备
        seed: 随机种子
        save_dir: 结果保存目录

    Returns:
        结果DataFrame
    """

    # 创建保存目录
    n_way = _cfg_resolve('multi_model_evaluation.py.test_multiple_models.n_way', n_way)
    k_shots = _cfg_resolve('multi_model_evaluation.py.test_multiple_models.k_shots', k_shots)
    n_trials = _cfg_resolve('multi_model_evaluation.py.test_multiple_models.n_trials', n_trials)
    q_query = _cfg_resolve('multi_model_evaluation.py.test_multiple_models.q_query', q_query)
    device = _cfg_resolve('multi_model_evaluation.py.test_multiple_models.device', device)
    seed = _cfg_resolve('multi_model_evaluation.py.test_multiple_models.seed', seed)
    save_dir = _cfg_resolve('multi_model_evaluation.py.test_multiple_models.save_dir', save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(save_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    config_save = {
        'n_way': n_way,
        'k_shots': k_shots,
        'n_trials': n_trials,
        'q_query': q_query,
        'seed': seed,
        'models': [{'name': c['name'], 'path': c['model_path']} for c in model_configs]
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=2)

    # 初始化评估器
    evaluator = MultiModelEvaluator(device=device, seed=seed)

    # 存储所有结果
    all_results = {}
    results_rows = []

    # 遍历所有模型配置
    for config in model_configs:
        model_name = config['name']
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        # 初始化模型
        model = config['model_class']()

        # 初始化Trainer
        trainer_kwargs = config.get('trainer_kwargs', {})
        trainer = config['trainer_class'](
            model=model,
            device=device,
            **trainer_kwargs
        )

        # 加载模型权重
        if os.path.exists(config['model_path']):
            trainer.load_model(config['model_path'])
            print(f"Loaded weights from: {config['model_path']}")
        else:
            print(f"WARNING: Model path not found: {config['model_path']}")
            print("Evaluating with untrained model...")

        model_results = {}

        # 遍历不同的K-shot
        for k_shot in k_shots:
            print(f"\n--- {n_way}-way {k_shot}-shot ---")

            result = evaluator.evaluate_single_model(
                trainer=trainer,
                test_dataset=test_dataset,
                n_way=n_way,
                k_shot=k_shot,
                q_query=q_query,
                n_trials=n_trials,
                model_name=model_name
            )

            model_results[f'{k_shot}-shot'] = result

            # 添加到结果行
            row = {
                'Model': model_name,
                'K-shot': k_shot,
                'Accuracy': result['accuracy']['mean'],
                'Accuracy_CI': result['accuracy']['ci95'],
                'F1': result['f1_macro']['mean'],
                'F1_CI': result['f1_macro']['ci95'],
                'Precision': result['precision']['mean'],
                'Recall': result['recall']['mean'],
                'Inference_ms': result['inference_time_ms']['mean']
            }

            if 'avg_prototypes' in result:
                row['Avg_Prototypes'] = result['avg_prototypes']['mean']

            results_rows.append(row)

            print(f"Accuracy: {result['accuracy']['mean']*100:.2f}% ± {result['accuracy']['ci95']*100:.2f}%")
            print(f"F1-score: {result['f1_macro']['mean']*100:.2f}%")

        all_results[model_name] = model_results

        # 清理GPU内存
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 创建DataFrame
    df = pd.DataFrame(results_rows)

    # 保存结果
    save_results(all_results, df, output_dir, n_way, k_shots)

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}")

    return df


def save_results(
    all_results: Dict,
    df: pd.DataFrame,
    output_dir: str,
    n_way: int,
    k_shots: List[int]
):
    """保存所有结果"""

    # <configured>. 保存JSON
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(os.path.join(output_dir, 'results_full.json'), 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    # <configured>. 保存CSV
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)

    # <configured>. 生成LaTeX表格
    latex_table = generate_latex_comparison_table(all_results, n_way, k_shots)
    with open(os.path.join(output_dir, 'table_latex.tex'), 'w') as f:
        f.write(latex_table)

    # <configured>. 生成对比图
    plot_comparison_curves(all_results, k_shots, output_dir)
    plot_comparison_bars(df, output_dir)

    # <configured>. 生成汇总报告
    generate_report(all_results, df, output_dir, n_way, k_shots)


def generate_latex_comparison_table(
    all_results: Dict,
    n_way: int,
    k_shots: List[int]
) -> str:
    """生成LaTeX对比表格"""

    # 表头
    k_shot_headers = " & ".join([f"\\textbf{{{k}-shot}}" for k in k_shots])

    latex = f"\\begin{{table*}}[t]\n\\centering\n\\caption{{Classification Accuracy (\\%) on DroneRFb-Spectra Dataset ({n_way}-way, 50 episodes)}}\n\\label{{tab:comparison_results}}\n\\small\n\\begin{{tabular}}{{l{'c' * len(k_shots)}}}\n\\toprule\n\\textbf{{Method}} & {k_shot_headers} \\\\\n\\midrule\n"

    # 找出每列最佳值
    best_per_kshot = {}
    for k in k_shots:
        key = f'{k}-shot'
        best_acc = 0
        for model_name, model_results in all_results.items():
            if key in model_results:
                acc = model_results[key]['accuracy']['mean']
                if acc > best_acc:
                    best_acc = acc
        best_per_kshot[k] = best_acc

    # 数据行
    for model_name, model_results in all_results.items():
        row = model_name
        for k in k_shots:
            key = f'{k}-shot'
            if key in model_results:
                acc = model_results[key]['accuracy']['mean']
                ci = model_results[key]['accuracy']['ci95']

                # 加粗最佳值
                if abs(acc - best_per_kshot[k]) < 0.001:
                    row += f" & \\textbf{{{acc*100:.1f}$\\pm${ci*100:.1f}}}"
                else:
                    row += f" & {acc*100:.1f}$\\pm${ci*100:.1f}"
            else:
                row += " & -"
        row += " \\\\"
        latex += row + "\n"

    latex += '\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
    return latex


def plot_comparison_curves(
    all_results: Dict,
    k_shots: List[int],
    output_dir: str
):
    """绘制K-shot性能曲线对比图"""

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, (model_name, model_results) in enumerate(all_results.items()):
        k_values = _cfg_require('multi_model_evaluation.py.plot_comparison_curves.k_values')
        accuracies = []
        cis = []

        for k in k_shots:
            key = f'{k}-shot'
            if key in model_results:
                k_values.append(k)
                accuracies.append(model_results[key]['accuracy']['mean'])
                cis.append(model_results[key]['accuracy']['ci95'])

        if k_values:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]

            ax.plot(k_values, accuracies, f'{marker}-', color=color,
                   linewidth=2, markersize=8, label=model_name)
            ax.fill_between(k_values,
                           [a - ci for a, ci in zip(accuracies, cis)],
                           [a + ci for a, ci in zip(accuracies, cis)],
                           alpha=_cfg_require('multi_model_evaluation.py.plot_comparison_curves.alpha__2'), color=color)

    ax.set_xlabel('Number of Shots (K)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('K-shot Classification Performance Comparison', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=_cfg_require('multi_model_evaluation.py.plot_comparison_curves.alpha'))
    ax.set_xticks(k_shots)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_kshot_curves.pdf'), dpi=_cfg_require('multi_model_evaluation.py.plot_comparison_curves.size_or_budget'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_kshot_curves.png'), dpi=_cfg_require('multi_model_evaluation.py.plot_comparison_curves.size_or_budget__2'), bbox_inches='tight')
    plt.close()

    print(f"K-shot curves saved to {output_dir}")


def plot_comparison_bars(df: pd.DataFrame, output_dir: str):
    """绘制分组柱状图对比"""

    # 对于每个K-shot绘制柱状图
    k_shots = df['K-shot'].unique()
    models = df['Model'].unique()

    fig, axes = plt.subplots(1, len(k_shots), figsize=(4 * len(k_shots), 5))
    if len(k_shots) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for ax, k in zip(axes, k_shots):
        subset = df[df['K-shot'] == k]
        x = np.arange(len(subset))

        bars = ax.bar(x, subset['Accuracy'], color=colors[:len(subset)],
                     yerr=subset['Accuracy_CI'], capsize=5)

        ax.set_xlabel('Method', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(f'{k}-shot', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(subset['Model'], rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0.5, 1.05])
        ax.grid(True, alpha=_cfg_require('multi_model_evaluation.py.plot_comparison_bars.alpha'), axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_comparison_bars.pdf'), dpi=_cfg_require('multi_model_evaluation.py.plot_comparison_bars.size_or_budget'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_comparison_bars.png'), dpi=_cfg_require('multi_model_evaluation.py.plot_comparison_bars.size_or_budget__2'), bbox_inches='tight')
    plt.close()

    print(f"Comparison bars saved to {output_dir}")


def generate_report(
    all_results: Dict,
    df: pd.DataFrame,
    output_dir: str,
    n_way: int,
    k_shots: List[int]
):
    """生成文本报告"""

    report = []
    report.append("=" * 70)
    report.append("Multi-Model Evaluation Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    report.append("")

    report.append(f"Configuration: {n_way}-way classification")
    report.append(f"K-shots tested: {k_shots}")
    report.append(f"Models evaluated: {len(all_results)}")
    report.append("")

    # 表格形式显示结果
    report.append("-" * 70)
    report.append("Results Summary (Accuracy %)")
    report.append("-" * 70)

    # 表头
    header = f"{'Model':<25}"
    for k in k_shots:
        header += f" | {k}-shot".ljust(15)
    report.append(header)
    report.append("-" * 70)

    # 数据行
    for model_name, model_results in all_results.items():
        row = f"{model_name:<25}"
        for k in k_shots:
            key = f'{k}-shot'
            if key in model_results:
                acc = model_results[key]['accuracy']['mean'] * 100
                ci = model_results[key]['accuracy']['ci95'] * 100
                row += f" | {acc:.1f}±{ci:.1f}".ljust(15)
            else:
                row += " | -".ljust(15)
        report.append(row)

    report.append("-" * 70)
    report.append("")

    # 最佳模型分析
    report.append("Best Model per K-shot:")
    for k in k_shots:
        key = f'{k}-shot'
        best_model = None
        best_acc = 0
        for model_name, model_results in all_results.items():
            if key in model_results:
                acc = model_results[key]['accuracy']['mean']
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name
        if best_model:
            report.append(f"  {k}-shot: {best_model} ({best_acc*100:.2f}%)")

    report.append("")
    report.append("=" * 70)

    # 保存报告
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(report_text)

    print(report_text)


# ==================== 示例用法 ====================

if __name__ == "__main__":
    """
    示例：如何使用多模型评估框架
    """

    print("\n================================================================================\n多模型评估框架使用示例\n================================================================================\n\n使用方法:\n\n<configured>. 准备数据集:\n```python\nfrom data_loader_clean import UAVDataset\n\ntest_dataset = UAVDataset(\n    data_dir_path='/path/to/DroneRFb-Spectra/Data',\n    seed=<configured>,\n    is_train=False,\n    train_ratio=<configured>/<configured>,\n    max_sample_count=<configured>\n)\n```\n\n<configured>. 配置模型列表:\n```python\nfrom GPN_V4_adaMulti_clean import GPNTrainer, GPN_Optimized\n\nmodel_configs = [\n    {\n        'name': 'AMGPN (τ=<configured>)',\n        'model_path': '/path/to/amgpn_model.pth',\n        'model_class': GPN_Optimized,\n        'trainer_class': GPNTrainer,\n        'trainer_kwargs': {\n            'use_Mdistance': True,\n            'use_multi': True,\n            'crop_ratio_h': <configured>,\n            'crop_ratio_l': <configured>,\n            'resize': True,\n            'merge_threshold': <configured>\n        }\n    },\n    {\n        'name': 'GPN',\n        'model_path': '/path/to/gpn_model.pth',\n        'model_class': GPN_Optimized,\n        'trainer_class': GPNTrainer,\n        'trainer_kwargs': {\n            'use_Mdistance': True,\n            'use_multi': False,\n            'crop_ratio_h': <configured>,\n            'crop_ratio_l': <configured>,\n            'resize': True\n        }\n    },\n    # ... 更多模型配置\n]\n```\n\n<configured>. 运行评估:\n```python\nfrom multi_model_evaluation import test_multiple_models\n\nresults_df = test_multiple_models(\n    test_dataset=test_dataset,\n    model_configs=model_configs,\n    n_way=<configured>,\n    k_shots=[<configured>, <configured>, <configured>, <configured>, <configured>],\n    n_trials=<configured>,\n    q_query=<configured>,\n    device='cuda',\n    seed=<configured>,\n    save_dir='./results'\n)\n```\n\n<configured>. 输出文件:\n- results_full.json     # 完整结果\n- results_summary.csv   # 汇总表格\n- table_latex.tex       # LaTeX表格代码\n- fig_kshot_curves.pdf  # K-shot曲线图\n- fig_comparison_bars.pdf # 柱状对比图\n- report.txt            # 文本报告\n\n================================================================================\n")
