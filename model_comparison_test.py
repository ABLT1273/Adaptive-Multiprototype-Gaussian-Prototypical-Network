from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import os
from datetime import datetime
import random

def test_multiple_models(
    test_dataset,
    model_paths,
    trainer_configs,
    model_initializers,
    n_way=None,
    k_shots=None,
    n_trials=None,
    q_query=None,
    device=None,
    seed=None,
    save_dir=None
):
    "\n    测试多个模型并比较性能和原型激活情况\n    \n    Args:\n        test_dataset: 测试数据集(MetaDataset)\n        model_paths: 模型路径列表 [{'name': 'Model1', 'path': 'path/to/model1.pth'}, ...]\n        trainer_configs: Trainer配置列表，与model_paths对应\n                        [{'use_multi': True, 'cut_ratio': <configured>, ...}, ...]\n        model_initializers: 模型初始化函数列表，返回模型实例\n                           [lambda: Model1(), lambda: Model2(), ...]\n        n_way: n-way分类任务，由外部配置提供\n        k_shots: k-shot列表，默认[<configured>, <configured>, <configured>]\n        n_trials: 每个配置的试验次数\n        q_query: 查询集大小\n        device: 设备\n        seed: 随机种子\n        save_dir: 结果保存目录\n        \n    Returns:\n        results_df: 包含所有结果的DataFrame\n    "

    # 创建保存目录
    n_way = _cfg_resolve('model_comparison_test.py.test_multiple_models.n_way', n_way)
    k_shots = _cfg_resolve('model_comparison_test.py.test_multiple_models.k_shots', k_shots)
    n_trials = _cfg_resolve('model_comparison_test.py.test_multiple_models.n_trials', n_trials)
    q_query = _cfg_resolve('model_comparison_test.py.test_multiple_models.q_query', q_query)
    device = _cfg_resolve('model_comparison_test.py.test_multiple_models.device', device)
    seed = _cfg_resolve('model_comparison_test.py.test_multiple_models.seed', seed)
    save_dir = _cfg_resolve('model_comparison_test.py.test_multiple_models.save_dir', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 结果存储
    all_results = []
    acc_results = defaultdict(lambda: defaultdict(list))  # model_name -> shot -> [acc_mean, acc_std]
    proto_results = defaultdict(lambda: defaultdict(list))  # model_name -> shot -> [proto_mean, proto_std]

    print(f"\n{'='*80}")
    print(f"Starting Multi-Model Evaluation")
    print(f"N-way: {n_way}, K-shots: {k_shots}, Trials: {n_trials}")
    print(f"{'='*80}\n")

    # 遍历所有模型
    for model_idx, (model_info, trainer_config, model_init) in enumerate(
        zip(model_paths, trainer_configs, model_initializers)
    ):
        model_name = model_info['name']
        model_path = model_info['path']

        print(f"\n[Model {model_idx + 1}/{len(model_paths)}] Testing: {model_name}")
        print(f"Model path: {model_path}")
        print("-" * 80)

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

            # 存储当前配置的结果
            accuracies = []
            avg_active_protos = []

            # 创建模型和trainer
            model = model_init()
            model = model.to(device)

            trainer=create_trainer(model,device,model_name,trainer_config)

            # 加载模型权重
            if os.path.exists(model_path):
                trainer.load_model(model_path)

            # 创建Trainer实例（需要根据实际的Trainer类调整）
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
                    num_tasks=_cfg_require('model_comparison_test.py.test_multiple_models.num_tasks'),
                    seed=random.randint(0, _cfg_require('model_comparison_test.py.test_multiple_models.size_or_budget'))
                )
                loader = DataLoader(meta_test, batch_size=_cfg_require('model_comparison_test.py.test_multiple_models.batch_size'),shuffle=True)
                acc=0
                eval_stats={}
                if model_name == 'CNN':
                    acc=trainer.evaluate_single_task(loader)
                elif model_name.find('Kmeans++')!=-1:
                    acc = trainer.evaluate(
                        loader,
                        n_way,
                        show_progress=False,
                        show_error_stats=False,
                    )
                elif model_name.find('IMP')!=-1:
                    acc = trainer.evaluate(
                        loader,
                        n_way,
                        show_progress=False,
                        show_error_stats=False,
                        show_prototype_stats=True,
                    )
                else:
                    # 评估并获取详细统计信息
                    acc,eval_stats = trainer.evaluate(# 经典模型参与时eval_stats删除
                        loader,
                        n_way,
                        show_progress=False,
                        show_error_stats=False,
                        return_stats=True  # 返回详细统计(经典模型参与时false)
                    )

                accuracies.append(acc)

                # 提取原型激活统计(经典模型参与时注释掉)
                if 'prototype_stats' in eval_stats and trainer_config.get('use_multi', False):
                    proto_stats = eval_stats['prototype_stats']['per_class']

                    # 计算平均激活原型数
                    total_active = 0
                    total_classes = 0

                    for class_id, stats in proto_stats.items():
                        if stats['total_tasks'] > 0:
                            avg_active = stats['active_sum'] / stats['total_tasks']
                            total_active += avg_active
                            total_classes += 1

                    if total_classes > 0:
                        trial_avg_active = total_active / total_classes
                        avg_active_protos.append(trial_avg_active)

                # 更新进度条
                mean_acc = np.mean(accuracies)
                pbar.set_postfix({'Acc': f"{mean_acc:.4f}"})

                # 清理内存
                del meta_test, loader
                if (trial + 1) % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # 计算统计量
            acc_mean = np.mean(accuracies)
            acc_std = np.std(accuracies)

            if avg_active_protos:
                proto_mean = np.mean(avg_active_protos)
                proto_std = np.std(avg_active_protos)
            else:
                proto_mean = k_shot  # 如果没有多原型模式，默认为k_shot
                proto_std = 0.0

            # 存储结果
            acc_results[model_name][k_shot] = [acc_mean, acc_std]
            proto_results[model_name][k_shot] = [proto_mean, proto_std]

            # 添加到结果列表
            result_row = {
                'Model': model_name,
                'N-way': n_way,
                'K-shot': k_shot,
                'Config': config_key,
                'Accuracy_Mean': acc_mean,
                'Accuracy_Std': acc_std,
                'Avg_Active_Prototypes': proto_mean,
                'Proto_Std': proto_std
            }
            all_results.append(result_row)

            print(f'    Results: Acc={acc_mean:.4f}±{acc_std:.4f}, Active Protos={proto_mean:.2f}±{proto_std:.2f}')

            # 清理模型和trainer
            del trainer, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        print(f"\nCompleted testing for {model_name}")
        print("-" * 80)

    # 创建DataFrame并保存CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(save_dir, f'model_comparison_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # 打印结果表格
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))

    # 绘制图表
    plot_comparison_charts(acc_results, proto_results, k_shots, save_dir, timestamp)

    return results_df


def plot_comparison_charts(acc_results, proto_results, k_shots, save_dir, timestamp):
    """
    绘制模型比较图表

    Args:
        acc_results: 准确率结果字典
        proto_results: 原型激活数结果字典
        k_shots: k-shot列表
        save_dir: 保存目录
        timestamp: 时间戳
    """
    # 设置图形样式
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 定义颜色和标记
    colors = plt.cm.tab10(np.linspace(0, 1, len(acc_results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    # 图<configured>: 准确率比较
    for idx, (model_name, results) in enumerate(acc_results.items()):
        shots = sorted(results.keys())
        means = [results[s][0] for s in shots]
        stds = [results[s][1] for s in shots]

        ax1.errorbar(shots, means, yerr=stds,
                    label=model_name,
                    color=colors[idx],
                    marker=markers[idx % len(markers)],
                    markersize=8,
                    linewidth=2,
                    capsize=5,
                    capthick=2)

    ax1.set_xlabel('K-shot', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('8-way Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(k_shots)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=_cfg_require('model_comparison_test.py.plot_comparison_charts.alpha'))
    ax1.set_ylim([0.7, 1.0])

    # 添加网格线
    ax1.yaxis.grid(True, linestyle='--', alpha=_cfg_require('model_comparison_test.py.plot_comparison_charts.alpha__2'))
    ax1.xaxis.grid(True, linestyle='--', alpha=_cfg_require('model_comparison_test.py.plot_comparison_charts.alpha__3'))

    # 图<configured>: 平均激活原型数比较
    for idx, (model_name, results) in enumerate(proto_results.items()):
        shots = sorted(results.keys())
        means = [results[s][0] for s in shots]
        stds = [results[s][1] for s in shots]

        ax2.errorbar(shots, means, yerr=stds,
                    label=model_name,
                    color=colors[idx],
                    marker=markers[idx % len(markers)],
                    markersize=8,
                    linewidth=2,
                    capsize=5,
                    capthick=2)

    ax2.set_xlabel('K-shot', fontsize=12)
    ax2.set_ylabel('Average Active Prototypes', fontsize=12)
    ax2.set_title('Average Active Prototypes Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(k_shots)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=_cfg_require('model_comparison_test.py.plot_comparison_charts.alpha__4'))

    # 添加参考线（显示理想的原型数量）
    for k in k_shots:
        ax2.axhline(y=k, color='gray', linestyle=':', alpha=_cfg_require('model_comparison_test.py.plot_comparison_charts.alpha__7'))
        ax2.text(k_shots[-1] + 0.1, k, f'k={k}', fontsize=9, color='gray')

    # 添加网格线
    ax2.yaxis.grid(True, linestyle='--', alpha=_cfg_require('model_comparison_test.py.plot_comparison_charts.alpha__5'))
    ax2.xaxis.grid(True, linestyle='--', alpha=_cfg_require('model_comparison_test.py.plot_comparison_charts.alpha__6'))

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plot_path = os.path.join(save_dir, f'model_comparison_plots_{timestamp}.png')
    plt.savefig(plot_path, dpi=_cfg_require('model_comparison_test.py.plot_comparison_charts.size_or_budget'), bbox_inches='tight')
    plt.show()
    print(f"\nPlots saved to: {plot_path}")


# 示例使用函数
def example_usage():
    """
    示例：如何使用test_multiple_models函数
    """
    # 假设已经有了数据集
    from your_dataset_module import load_test_dataset
    test_dataset = load_test_dataset()

    # 定义模型路径
    model_paths = _cfg_require('model_comparison_test.py.example_usage.model_paths')

    # 定义trainer配置（根据实际需要调整）
    trainer_configs = _cfg_require('model_comparison_test.py.example_usage.trainer_configs')

    # 定义模型初始化函数
    from your_model_module import BaselineModel, MultiProtoModel, EnhancedModel
    model_initializers = [
        lambda: BaselineModel(),
        lambda: MultiProtoModel(),
        lambda: EnhancedModel()
    ]

    # 运行测试
    results_df = test_multiple_models(
        test_dataset=test_dataset,
        model_paths=model_paths,
        trainer_configs=trainer_configs,
        model_initializers=model_initializers,
        n_way=_cfg_require('model_comparison_test.py.example_usage.n_way'),
        k_shots=_cfg_require('model_comparison_test.py.example_usage.k_shots'),
        n_trials=_cfg_require('model_comparison_test.py.example_usage.n_trials'),
        q_query=_cfg_require('model_comparison_test.py.example_usage.q_query'),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=_cfg_require('model_comparison_test.py.example_usage.seed'),
        save_dir=_cfg_require('model_comparison_test.py.example_usage.save_dir')
    )

    return results_df

def plot_comparison_from_csv(csv_path, save_dir=None, timestamp=None):
    """
    通过读取 CSV 文件，重新构建数据结构并绘制模型比较图表。

    Args:
        csv_path: 由 test_multiple_models 生成的 CSV 文件路径。
        save_dir: 保存图表的目录。如果未提供，将使用 CSV 文件所在的目录。
        timestamp: 用于命名保存图表文件的时间戳。如果未提供，将尝试从文件名推断。
    """
    if not os.path.exists(csv_path):
        print(f"错误: CSV 文件未找到于 {csv_path}")
        return

    # <configured>. 读取 CSV 文件到 DataFrame
    results_df = pd.read_csv(csv_path)

    acc_results = defaultdict(lambda: defaultdict(list))
    proto_results = defaultdict(lambda: defaultdict(list))

    # 提取唯一的 K-shot 值并排序
    k_shots = sorted(results_df['K-shot'].unique())

    # 遍历 DataFrame 填充字典
    for _, row in results_df.iterrows():
        model_name = row['Model']
        k_shot = row['K-shot']

        # 填充准确率结果: {Model: {K-shot: [Mean, Std]}}
        acc_results[model_name][k_shot] = [row['Accuracy_Mean'], row['Accuracy_Std']]

        # 填充原型结果: {Model: {K-shot: [Mean, Std]}}
        proto_results[model_name][k_shot] = [row['Avg_Active_Prototypes'], row['Proto_Std']]

    # <configured>. 处理 save_dir 和 timestamp
    if save_dir is None:
        save_dir = os.path.dirname(csv_path)
    if timestamp is None:
        # 尝试从文件名推断时间戳（假设格式为 model_comparison_YYYYMMDD_HHMMSS.csv）
        filename = os.path.basename(csv_path)
        try:
            timestamp = filename.split('_')[-1].replace('.csv', '')
        except:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # <configured>. 调用原有的绘图函数
    print(f"\n正在从 CSV 文件 {os.path.basename(csv_path)} 绘制图表...")
    plot_comparison_charts(acc_results, proto_results, k_shots, save_dir, timestamp)

def create_trainer(model,device,model_name,trainer_config):
    if model_name=='CNN':
        from CNN_baseline import CNNBaselineTrainer
        trainer=CNNBaselineTrainer(
            model=model,
            device=device,
            **trainer_config  # 传入配置参数
        )
        return trainer
    elif model_name.find('Kmeans++')!=-1:
        from GPN_V3 import GPNTrainer
        trainer=GPNTrainer(
            model=model,
            device=device,
            **trainer_config  # 传入配置参数
        )
        return trainer
    elif model_name=='IMP':
        from GPN_V3_IMP_onlyTrainer import GPNTrainerWithIMP
        trainer=GPNTrainerWithIMP(
            model=model,
            device=device,
            **trainer_config  # 传入配置参数
            )
        return trainer
    else:
        from GPN_V4_adaMulti import GPNTrainer
        trainer = GPNTrainer(
            model=model,
            device=device,
            **trainer_config  # 传入配置参数
        )
        return trainer

if __name__ == "__main__":
    # 运行示例
    results = example_usage()
    print("\nExperiment completed successfully!")
