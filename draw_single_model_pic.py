from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体（根据系统选择）
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用于显示中文
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_evaluation_results(results, save_path=None):
    """
    绘制评估结果折线图

    Args:
        results: full_evaluation返回的字典
                 格式: {'3w1s': (mean, std), '3w5s': (mean, std), ...}
        save_path: 保存图片的路径
    """
    # 解析results
    save_path = _cfg_resolve('draw_single_model_pic.py.plot_evaluation_results.save_path', save_path)
    n_ways = _cfg_require('draw_single_model_pic.py.plot_evaluation_results.n_ways')
    k_shots = _cfg_require('draw_single_model_pic.py.plot_evaluation_results.k_shots')

    # 准备数据结构
    data = {k: [] for k in k_shots}
    stds = {k: [] for k in k_shots}

    # 填充数据
    for n_way in n_ways:
        for k_shot in k_shots:
            key = f'{n_way}w{k_shot}s'
            if key in results:
                mean_acc, std_acc = results[key]
                data[k_shot].append(mean_acc)
                stds[k_shot].append(std_acc)
            else:
                data[k_shot].append(None)
                stds[k_shot].append(None)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))

    # 定义颜色和标记
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    markers = ['o', 's', '^']

    # 绘制每个k-shot的曲线
    for idx, k_shot in enumerate(k_shots):
        means = data[k_shot]
        errors = stds[k_shot]

        # 绘制主线
        line = ax.plot(n_ways, means,
                      marker=markers[idx],
                      color=colors[idx],
                      linewidth=2.5,
                      markersize=10,
                      label=f'{k_shot}-shot',
                      alpha=_cfg_require('draw_single_model_pic.py.plot_evaluation_results.alpha__2'))

        # 添加误差带
        ax.fill_between(n_ways,
                       [m - e for m, e in zip(means, errors)],
                       [m + e for m, e in zip(means, errors)],
                       color=colors[idx],
                       alpha=_cfg_require('draw_single_model_pic.py.plot_evaluation_results.alpha__3'))

        # 在每个点上标注具体数值
        for x, y, err in zip(n_ways, means, errors):
            ax.annotate(f'{y:.3f}',
                       xy=(x, y),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=8,
                       color=colors[idx],
                       weight='bold')

    # 设置图表属性
    ax.set_xlabel('Number of Ways (N-way)', fontsize=14, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, weight='bold')
    ax.set_title('Few-Shot UAV Recognition Performance\n(GPN Model)',
                fontsize=16, weight='bold', pad=20)

    # 设置x轴
    ax.set_xticks(n_ways)
    ax.set_xticklabels([f'{n}-way' for n in n_ways])

    # 设置y轴范围和格式
    ax.set_ylim([0.5, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

    # 添加网格
    ax.grid(True, linestyle='--', alpha=_cfg_require('draw_single_model_pic.py.plot_evaluation_results.alpha'), which='both')
    ax.set_axisbelow(True)

    # 图例
    ax.legend(loc='lower left',
             fontsize=12,
             frameon=True,
             shadow=True,
             title='K-shot Settings',
             title_fontsize=12)

    # 优化布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=_cfg_require('draw_single_model_pic.py.plot_evaluation_results.size_or_budget'), bbox_inches='tight')
    print(f"图表已保存到: {save_path}")

    # 显示图表
    plt.show()

    return fig, ax


def plot_detailed_comparison(results, save_path=None):
    """
    绘制更详细的对比图（包含子图）
    """
    save_path = _cfg_resolve('draw_single_model_pic.py.plot_detailed_comparison.save_path', save_path)
    n_ways = _cfg_require('draw_single_model_pic.py.plot_detailed_comparison.n_ways')
    k_shots = _cfg_require('draw_single_model_pic.py.plot_detailed_comparison.k_shots')

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Few-Shot Performance Analysis by K-shot',
                fontsize=16, weight='bold', y=1.02)

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_ways)))

    for idx, k_shot in enumerate(k_shots):
        ax = axes[idx]

        # 收集数据
        means = []
        stds = []
        for n_way in n_ways:
            key = f'{n_way}w{k_shot}s'
            if key in results:
                mean_acc, std_acc = results[key]
                means.append(mean_acc)
                stds.append(std_acc)

        # 绘制柱状图
        bars = ax.bar(range(len(n_ways)), means,
                     color=colors,
                     alpha=_cfg_require('draw_single_model_pic.py.plot_detailed_comparison.alpha'),
                     edgecolor='black',
                     linewidth=1.5)

        # 添加误差线
        ax.errorbar(range(len(n_ways)), means, yerr=stds,
                   fmt='none',
                   ecolor='black',
                   capsize=5,
                   capthick=2,
                   alpha=_cfg_require('draw_single_model_pic.py.plot_detailed_comparison.alpha__2'))

        # 在柱子上标注数值
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}\n±{std:.3f}',
                   ha='center', va='bottom',
                   fontsize=9, weight='bold')

        # 设置子图属性
        ax.set_title(f'{k_shot}-shot', fontsize=14, weight='bold', pad=10)
        ax.set_xlabel('N-way', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xticks(range(len(n_ways)))
        ax.set_xticklabels([f'{n}' for n in n_ways])
        ax.set_ylim([0.5, 1.05])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(True, axis='y', linestyle='--', alpha=_cfg_require('draw_single_model_pic.py.plot_detailed_comparison.alpha__3'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=_cfg_require('draw_single_model_pic.py.plot_detailed_comparison.size_or_budget'), bbox_inches='tight')
    print(f"详细对比图已保存到: {save_path}")
    plt.show()

    return fig, axes


def plot_heatmap(results, save_path=None):
    """
    绘制性能热力图
    """
    save_path = _cfg_resolve('draw_single_model_pic.py.plot_heatmap.save_path', save_path)
    n_ways = _cfg_require('draw_single_model_pic.py.plot_heatmap.n_ways')
    k_shots = _cfg_require('draw_single_model_pic.py.plot_heatmap.k_shots')

    # 准备数据矩阵
    data_matrix = np.zeros((len(k_shots), len(n_ways)))

    for i, k_shot in enumerate(k_shots):
        for j, n_way in enumerate(n_ways):
            key = f'{n_way}w{k_shot}s'
            if key in results:
                data_matrix[i, j] = results[key][0]  # 只用mean

    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)

    # 设置刻度
    ax.set_xticks(np.arange(len(n_ways)))
    ax.set_yticks(np.arange(len(k_shots)))
    ax.set_xticklabels([f'{n}-way' for n in n_ways])
    ax.set_yticklabels([f'{k}-shot' for k in k_shots])

    # 在每个格子上标注数值
    for i in range(len(k_shots)):
        for j in range(len(n_ways)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                         ha="center", va="center",
                         color="black" if data_matrix[i, j] > 0.75 else "white",
                         fontsize=12, weight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=_cfg_require('draw_single_model_pic.py.plot_heatmap.size_or_budget'), labelpad=20, fontsize=12, weight='bold')

    # 设置标题和标签
    ax.set_title('Few-Shot Recognition Accuracy Heatmap',
                fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Task Complexity (N-way)', fontsize=12, weight='bold')
    ax.set_ylabel('Support Set Size (K-shot)', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=_cfg_require('draw_single_model_pic.py.plot_heatmap.size_or_budget__2'), bbox_inches='tight')
    print(f"热力图已保存到: {save_path}")
    plt.show()

    return fig, ax


def plot_all_visualizations(results, output_dir=None):
    """
    生成所有可视化图表
    """
    output_dir = _cfg_resolve('draw_single_model_pic.py.plot_all_visualizations.output_dir', output_dir)
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("生成可视化图表...")

    # <configured>. 主折线图
    plot_evaluation_results(
        results,
        save_path=os.path.join(output_dir, 'line_plot.png')
    )

    # <configured>. 详细对比图
    plot_detailed_comparison(
        results,
        save_path=os.path.join(output_dir, 'bar_plot.png')
    )

    # <configured>. 热力图
    plot_heatmap(
        results,
        save_path=os.path.join(output_dir, 'heatmap.png')
    )

    print(f"\n所有图表已保存到: {output_dir}")


# # 使用示例
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def plot_multi_model_comparison(
    all_results,
    n_way=None,
    save_path=None,
    title=None
):
    "\n    绘制多模型在<configured>-way不同k-shot下的对比折线图\n    \n    Args:\n        all_results: full_evaluation_multi_models的返回值\n                    格式: {'Model1': {'8w1s': (mean, std), ...}, ...}\n        n_way: 固定的n-way值（由外部配置提供）\n        save_path: 保存路径\n        title: 图表标题（可选）\n    "
    # 提取k-shots（从第一个模型的结果中推断）
    n_way = _cfg_resolve('draw_single_model_pic.py.plot_multi_model_comparison.n_way', n_way)
    save_path = _cfg_resolve('draw_single_model_pic.py.plot_multi_model_comparison.save_path', save_path)
    first_model = list(all_results.keys())[0]
    k_shots = _cfg_require('draw_single_model_pic.py.plot_multi_model_comparison.k_shots')
    for key in all_results[first_model].keys():
        if key.startswith(f'{n_way}w'):
            k_shot = int(key.split('w')[1].rstrip('s'))
            k_shots.append(k_shot)
    k_shots = sorted(k_shots)

    # 准备数据
    model_names = list(all_results.keys())
    data = {model: [] for model in model_names}
    errors = {model: [] for model in model_names}

    for k_shot in k_shots:
        config_key = f'{n_way}w{k_shot}s'
        for model_name in model_names:
            if config_key in all_results[model_name]:
                mean, std = all_results[model_name][config_key]
                data[model_name].append(mean)
                errors[model_name].append(std)
            else:
                data[model_name].append(None)
                errors[model_name].append(None)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))

    # 定义颜色和标记（支持更多模型）
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':']

    # 绘制每个模型的曲线
    for idx, model_name in enumerate(model_names):
        means = data[model_name]
        stds = errors[model_name]

        # 绘制主线
        line = ax.plot(
            k_shots, means,
            marker=markers[idx % len(markers)],
            color=colors[idx],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2.5,
            markersize=10,
            label=model_name,
            alpha=_cfg_require('draw_single_model_pic.py.plot_multi_model_comparison.alpha__2')
        )

        # 添加误差带
        ax.fill_between(
            k_shots,
            [m - e if m is not None else None for m, e in zip(means, stds)],
            [m + e if m is not None else None for m, e in zip(means, stds)],
            color=colors[idx],
            alpha=_cfg_require('draw_single_model_pic.py.plot_multi_model_comparison.alpha__3')
        )

        # 在每个点上标注具体数值
        for x, y, err in zip(k_shots, means, stds):
            if y is not None:
                ax.annotate(
                    f'{y:.3f}',
                    xy=(x, y),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    color=colors[idx],
                    weight='bold'
                )

    # 设置图表属性
    ax.set_xlabel('K-shot', fontsize=14, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, weight='bold')

    if title is None:
        title = f'{n_way}-way Few-Shot Recognition\nModel Comparison'
    ax.set_title(title, fontsize=16, weight='bold', pad=20)

    # 设置x轴
    ax.set_xticks(k_shots)
    ax.set_xticklabels([f'{k}-shot' for k in k_shots])

    # 设置y轴
    ax.set_ylim([0.5, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # 添加网格
    ax.grid(True, linestyle='--', alpha=_cfg_require('draw_single_model_pic.py.plot_multi_model_comparison.alpha'), which='both')
    ax.set_axisbelow(True)

    # 图例
    ax.legend(
        loc='lower right',
        fontsize=11,
        frameon=True,
        shadow=True,
        title='Models',
        title_fontsize=12
    )

    # 优化布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=_cfg_require('draw_single_model_pic.py.plot_multi_model_comparison.size_or_budget'), bbox_inches='tight')
    print(f"对比图已保存到: {save_path}")

    plt.show()

    return fig, ax


def plot_multi_model_bar_comparison(
    all_results,
    n_way=None,
    save_path=None
):
    """
    绘制多模型的柱状对比图

    Args:
        all_results: full_evaluation_multi_models的返回值
        n_way: 固定的n-way值
        save_path: 保存路径
    """
    # 提取k-shots
    n_way = _cfg_resolve('draw_single_model_pic.py.plot_multi_model_bar_comparison.n_way', n_way)
    save_path = _cfg_resolve('draw_single_model_pic.py.plot_multi_model_bar_comparison.save_path', save_path)
    first_model = list(all_results.keys())[0]
    k_shots = _cfg_require('draw_single_model_pic.py.plot_multi_model_bar_comparison.k_shots')
    for key in all_results[first_model].keys():
        if key.startswith(f'{n_way}w'):
            k_shot = int(key.split('w')[1].rstrip('s'))
            k_shots.append(k_shot)
    k_shots = sorted(k_shots)

    model_names = list(all_results.keys())
    n_models = len(model_names)

    # 准备数据
    data = {k: [] for k in k_shots}
    errors = {k: [] for k in k_shots}

    for k_shot in k_shots:
        config_key = f'{n_way}w{k_shot}s'
        for model_name in model_names:
            mean, std = all_results[model_name][config_key]
            data[k_shot].append(mean)
            errors[k_shot].append(std)

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))

    # 设置柱子的位置和宽度
    bar_width = 0.8 / n_models
    x = np.arange(len(k_shots))

    # 定义颜色
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # 绘制每个模型的柱子
    for i, model_name in enumerate(model_names):
        means = [data[k][i] for k in k_shots]
        stds = [errors[k][i] for k in k_shots]

        offset = (i - n_models/2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            label=model_name,
            color=colors[i],
            alpha=_cfg_require('draw_single_model_pic.py.plot_multi_model_bar_comparison.alpha__2'),
            edgecolor='black',
            linewidth=1.2,
            capsize=5
        )

        # 在柱子上标注数值
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{mean:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                weight='bold'
            )

    # 设置图表属性
    ax.set_xlabel('K-shot', fontsize=14, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, weight='bold')
    ax.set_title(f'{n_way}-way Few-Shot Recognition\nModel Performance Comparison',
                fontsize=16, weight='bold', pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}-shot' for k in k_shots])
    ax.set_ylim([0.5, 1.05])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # 添加网格
    ax.grid(True, axis='y', linestyle='--', alpha=_cfg_require('draw_single_model_pic.py.plot_multi_model_bar_comparison.alpha'))
    ax.set_axisbelow(True)

    # 图例
    ax.legend(
        loc='upper left',
        fontsize=11,
        frameon=True,
        shadow=True,
        ncol=min(3, n_models)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=_cfg_require('draw_single_model_pic.py.plot_multi_model_bar_comparison.size_or_budget'), bbox_inches='tight')
    print(f"柱状对比图已保存到: {save_path}")
    plt.show()

    return fig, ax


def plot_improvement_analysis(
    all_results,
    baseline_model,
    n_way=None,
    save_path=None
):
    """
    绘制相对于baseline的改进分析图

    Args:
        all_results: full_evaluation_multi_models的返回值
        baseline_model: baseline模型的名称（字符串）
        n_way: 固定的n-way值
        save_path: 保存路径
    """
    # 提取k-shots
    n_way = _cfg_resolve('draw_single_model_pic.py.plot_improvement_analysis.n_way', n_way)
    save_path = _cfg_resolve('draw_single_model_pic.py.plot_improvement_analysis.save_path', save_path)
    k_shots = _cfg_require('draw_single_model_pic.py.plot_improvement_analysis.k_shots')
    for key in all_results[baseline_model].keys():
        if key.startswith(f'{n_way}w'):
            k_shot = int(key.split('w')[1].rstrip('s'))
            k_shots.append(k_shot)
    k_shots = sorted(k_shots)

    model_names = [m for m in all_results.keys() if m != baseline_model]

    # 计算改进百分比
    improvements = {model: [] for model in model_names}

    for k_shot in k_shots:
        config_key = f'{n_way}w{k_shot}s'
        baseline_acc = all_results[baseline_model][config_key][0]

        for model_name in model_names:
            model_acc = all_results[model_name][config_key][0]
            improvement = (model_acc - baseline_acc) / baseline_acc * 100
            improvements[model_name].append(improvement)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    markers = ['o', 's', '^', 'D', 'v']

    # 绘制改进曲线
    for idx, model_name in enumerate(model_names):
        ax.plot(
            k_shots,
            improvements[model_name],
            marker=markers[idx % len(markers)],
            color=colors[idx],
            linewidth=2.5,
            markersize=10,
            label=f'{model_name} vs {baseline_model}',
            alpha=_cfg_require('draw_single_model_pic.py.plot_improvement_analysis.alpha__3')
        )

        # 标注数值
        for x, y in zip(k_shots, improvements[model_name]):
            ax.annotate(
                f'{y:+.2f}%',
                xy=(x, y),
                xytext=(0, 10 if y > 0 else -15),
                textcoords='offset points',
                ha='center',
                fontsize=9,
                color=colors[idx],
                weight='bold'
            )

    # 添加零线
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=_cfg_require('draw_single_model_pic.py.plot_improvement_analysis.alpha'))

    # 设置图表属性
    ax.set_xlabel('K-shot', fontsize=14, weight='bold')
    ax.set_ylabel('Relative Improvement (%)', fontsize=14, weight='bold')
    ax.set_title(f'{n_way}-way Relative Improvement Analysis\n(vs {baseline_model})',
                fontsize=16, weight='bold', pad=20)

    ax.set_xticks(k_shots)
    ax.set_xticklabels([f'{k}-shot' for k in k_shots])

    # 添加网格
    ax.grid(True, linestyle='--', alpha=_cfg_require('draw_single_model_pic.py.plot_improvement_analysis.alpha__2'))
    ax.set_axisbelow(True)

    # 图例
    ax.legend(
        loc='best',
        fontsize=11,
        frameon=True,
        shadow=True
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=_cfg_require('draw_single_model_pic.py.plot_improvement_analysis.size_or_budget'), bbox_inches='tight')
    print(f"改进分析图已保存到: {save_path}")
    plt.show()

    return fig, ax


def plot_all_multi_model_visualizations(
    all_results,
    baseline_model=None,
    n_way=None,
    output_dir=None
):
    """
    生成所有多模型对比可视化

    Args:
        all_results: full_evaluation_multi_models的返回值
        baseline_model: baseline模型名称（用于改进分析）
        n_way: 固定的n-way值
        output_dir: 输出目录
    """
    n_way = _cfg_resolve('draw_single_model_pic.py.plot_all_multi_model_visualizations.n_way', n_way)
    output_dir = _cfg_resolve('draw_single_model_pic.py.plot_all_multi_model_visualizations.output_dir', output_dir)
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("生成多模型对比可视化...")
    print("="*70)

    # <configured>. 折线对比图
    print("\n[1/4] 生成折线对比图...")
    plot_multi_model_comparison(
        all_results,
        n_way=n_way,
        save_path=os.path.join(output_dir, f'line_comparison_{n_way}way.png')
    )

    # <configured>. 柱状对比图
    print("\n[2/4] 生成柱状对比图...")
    plot_multi_model_bar_comparison(
        all_results,
        n_way=n_way,
        save_path=os.path.join(output_dir, f'bar_comparison_{n_way}way.png')
    )

    # <configured>. 改进分析图（如果指定了baseline）
    if baseline_model and baseline_model in all_results:
        print("\n[3/4] 生成改进分析图...")
        plot_improvement_analysis(
            all_results,
            baseline_model=baseline_model,
            n_way=n_way,
            save_path=os.path.join(output_dir, f'improvement_{n_way}way.png')
        )
    else:
        print("\n[3/4] 跳过改进分析图（未指定baseline模型）")

    # <configured>. 打印结果表格
    print("\n[4/4] 打印结果表格...")
    print_results_table(all_results)

    print(f"\n所有可视化已保存到: {output_dir}")


# ==================== 使用示例 ====================
