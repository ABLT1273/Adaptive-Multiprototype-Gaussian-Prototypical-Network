from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def save_results(df, data_pack, output_dir, n_way, k_shots):
    """
    保存所有结果的主控函数

    Args:
        df (pd.DataFrame): 包含多次实验平均指标的表格 (来自 test_multiple_models)
        data_pack (dict): 包含详细数据的字典 (可能包含 features, predictions, sensitivity_df 等)
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # <configured>. 设置学术绘图风格
    _set_academic_style()

    # <configured>. 保存基础数据
    # CSV
    df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    # JSON (序列化 DataFrame 为字典以便阅读)
    json_results = df.to_dict(orient='records')
    with open(os.path.join(output_dir, 'summary_metrics.json'), 'w') as f:
        json.dump(json_results, f, indent=2, default=float)

    latex_code = _generate_latex_table_from_df(df, n_way, k_shots)
    with open(os.path.join(output_dir, 'main_results_table.tex'), 'w') as f:
        f.write(latex_code)

    # 依赖: df
    _plot_k_shot_curves(df, output_dir,)
    # 依赖: data_pack['features'] 和 data_pack['feature_labels']
    if 'features' in data_pack and 'feature_labels' in data_pack:
        _plot_tsne(
            data_pack['features'],
            data_pack['feature_labels'],
            output_dir
        )

    # 依赖: data_pack['predictions'] 和 data_pack['targets']
    if 'predictions' in data_pack and 'targets' in data_pack:
        if 'pred_model_names' in data_pack and 'pred_k_shots' in data_pack:
            # ===== 方案A：分模型绘制多个混淆矩阵 =====
            _plot_confusion_matrices_per_model(
                data_pack['targets'],
                data_pack['predictions'],
                data_pack['pred_model_names'],
                data_pack['pred_k_shots'],
                output_dir,
                class_names=data_pack.get('class_names', None)
                )
        else:
            # ===== 方案B：整体混淆矩阵（向后兼容）=====
            _plot_confusion_matrix(
            data_pack['targets'],
            data_pack['predictions'],
            output_dir,
            class_names=data_pack.get('class_names', None)
            )

    if 'prototype_stats_df' in data_pack:
        k_shots_theory=_cfg_require('draw_all_model_pic.py.save_results.k_shots_theory')
        _plot_prototype_curves(data_pack['prototype_stats_df'], output_dir,
                                        extra_models=['GMM(comp=<configured>)'],
                                        extra_k_shots=[k_shots_theory,],
                                        extra_active_protos=[[1,4,5,5],]
                              )
        print(f"All results and artifacts saved to: {output_dir}")

def _generate_latex_table_from_df(df, n_way, k_shots):
    """基于DataFrame生成LaTeX表格，自动加粗最优结果"""
    # 转换数据格式为 Pivot Table
    pivot = df.pivot(index='Model', columns='K-shot', values=['Accuracy', 'CI95_Acc'])

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(fr"\caption{{Classification Accuracy (\%) on Dataset ({n_way}-way, 95\% CI).}}")
    latex.append(r"\label{tab:main_results}")
    latex.append(r"\small")

    # 构建表头
    col_desc = "l" + "c" * len(k_shots)
    header_shots = " & ".join([fr"\textbf{{{k}-shot}}" for k in k_shots])
    latex.append(fr"\begin{{tabular}}{{{col_desc}}}")
    latex.append(r"\toprule")
    latex.append(fr"\textbf{{Method}} & {header_shots} \\")
    latex.append(r"\midrule")

    # 遍历每一行 (模型)
    models = df['Model'].unique()

    # 获取每一列 (K-shot) 的最大值用于加粗
    best_accs = {}
    for k in k_shots:
        try:
            # 注意：此处假设 index 是 Model
            k_col = pivot['Accuracy'][k]
            best_accs[k] = k_col.max()
        except KeyError:
            best_accs[k] = -1

    for model in models:
        row_str = f"{model}"
        for k in k_shots:
            try:
                acc = pivot.loc[model, ('Accuracy', k)]
                ci = pivot.loc[model, ('CI95_Acc', k)]

                # 格式化: <configured> ± <configured>
                cell_content = f"{acc*100:.1f} $\\pm$ {ci*100:.1f}"

                if acc >= best_accs[k] - 0.0001:
                    row_str += f" & \\textbf{{{cell_content}}}"
                else:
                    row_str += f" & {cell_content}"
            except KeyError:
                row_str += " & -"
        row_str += r" \\"
        latex.append(row_str)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)

def _set_academic_style():
    """配置符合学术出版标准的matplotlib样式"""
    plt.style.use('seaborn-v0_8-paper') # 使用适合论文的预设

    # 字体配置 (推荐 Times New Roman 或 Arial)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False, # 如果系统安装了LaTeX可设为True，否则False
        'figure.dpi': _cfg_require('draw_all_model_pic.py._set_academic_style.size_or_budget'),
        'savefig.dpi': _cfg_require('draw_all_model_pic.py._set_academic_style.size_or_budget__2'),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

def _plot_k_shot_curves(df, output_dir):
    """绘制带误差带的 K-shot 性能曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 获取所有模型和K值
    models = df['Model'].unique()
    # 颜色映射
# ===== 使用 colorblind 调色板并手动映射颜色 =====
    colorblind_palette = sns.color_palette("colorblind", 10)  # 获取完整调色板

    # 定义模型到颜色的映射（基于 colorblind 调色板的索引）
    color_map = {
        'GMM(comp=5)': colorblind_palette[0],
        'IMP': colorblind_palette[3],
        'AMGPN': colorblind_palette[1],
        'Kmeans++(K=7)': colorblind_palette[2]
    }

    # 定义模型到标记点的映射
    marker_map = {
        'GMM(comp=5)': 'o',        # 圆圈
        'IMP': '^',        # 三角形
        'AMGPN': 's',      # 正方形
        'Kmeans++(K=7)': 'D'  # 菱形
    }

    # 备用标记点列表（用于额外模型）

    # 辅助函数：根据模型名匹配颜色
    def get_color(model_name):
        for key in color_map:
            if key.lower() in model_name.lower():
                return color_map[key]
        return colorblind_palette[7]  # 默认灰色

    # 辅助函数：根据模型名匹配标记点
    def get_marker(model_name):
        for key in marker_map:
            if key.lower() in model_name.lower():
                return marker_map[key]
        return 'o'  # 默认圆圈

    for idx, model_name in enumerate(models):
        subset = df[df['Model'] == model_name].sort_values('K-shot')

        k_vals = subset['K-shot'].values
        accs = subset['Accuracy'].values * 100
        cis = subset['CI95_Acc'].values * 100

        # 绘制主线
        ax.plot(k_vals, accs,
                marker=get_marker(model_name),
                markersize=8,
                linewidth=2,
                color=get_color(model_name),
                label=model_name)

        ax.fill_between(k_vals, accs - cis, accs + cis,
                        color=get_color(model_name), alpha=_cfg_require('draw_all_model_pic.py._plot_k_shot_curves.alpha'))

    ax.set_xlabel('Number of Shots (K)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Few-shot Classification Performance', fontsize=15, pad=15)

    # 设置X轴刻度为整数
    ax.set_xticks(sorted(df['K-shot'].unique()))

    # 优化图例
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_performance_curve.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_performance_curve.png'), bbox_inches='tight')
    plt.close()

# #针对阈值敏感性
def _plot_tsne(features, labels, output_dir):
    """绘制 t-SNE 可视化图"""
    print(f"  [t-SNE] 输入特征形状: {features.shape}")

    if isinstance(labels, list):
        labels = np.array(labels)

    if labels.ndim > 1:
        labels = labels.flatten()

    # 额外检查：如果元素是数组，提取标量
    if labels.dtype == object or (len(labels) > 0 and isinstance(labels[0], np.ndarray)):
        labels = np.array([int(x) if isinstance(x, (np.ndarray, list)) else x for x in labels])

    print(f"  [t-SNE] 标签形状: {labels.shape}, dtype: {labels.dtype}")

    # 限制样本数量以加快速度
    if len(features) > _cfg_require('draw_all_model_pic.py._plot_tsne.size_or_budget'):
        indices = np.random.choice(len(features), _cfg_require('draw_all_model_pic.py._plot_tsne.size_or_budget__2'), replace=False)
        features = features[indices]
        labels = labels[indices]

    # t-SNE 降维
    tsne = TSNE(n_components=_cfg_require('draw_all_model_pic.py._plot_tsne.n_components'), perplexity=30, max_iter=_cfg_require('draw_all_model_pic.py._plot_tsne.max_iter'), random_state=_cfg_require('draw_all_model_pic.py._plot_tsne.random_state'))
    embeddings = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 获取唯一标签
    unique_labels = np.unique(labels)
    palette = sns.color_palette("bright", len(unique_labels))

    sns.scatterplot(
        x=embeddings[:, 0], y=embeddings[:, 1],
        hue=labels,
        palette=palette,
        legend='full',
        s=60,
        alpha=_cfg_require('draw_all_model_pic.py._plot_tsne.alpha'),
        edgecolor='w',
        linewidth=0.5,
        ax=ax
    )

    # 移除坐标轴刻度 (t-SNE 坐标绝对值无意义)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('t-SNE Visualization of Feature Embeddings', fontsize=15)

    # 调整图例到图外
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Classes')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_tsne.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig4_tsne.png'), bbox_inches='tight')
    plt.close()

def _plot_confusion_matrices_per_model(y_true, y_pred, model_names, k_shots,
                                    output_dir, class_names=None):
    """为每个模型的每个k_shot配置绘制独立的混淆矩阵"""
    # 验证数据
    y_true = _validate_and_fix_array(y_true, "y_true")
    y_pred = _validate_and_fix_array(y_pred, "y_pred")

    # 转为 DataFrame 便于分组
    import pandas as pd
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'model': model_names,
        'k_shot': k_shots
    })

    # 按模型和k_shot分组
    grouped = df.groupby(['model', 'k_shot'])

    print(f"\n[Confusion Matrix] Found {len(grouped)} model-kshot combinations:")
    for (model, k), group in grouped:
        print(f"  - {model} @ {k}-shot: {len(group)} samples")

    # 为每个组绘制混淆矩阵
    for (model, k), group in grouped:
        _plot_single_confusion_matrix(
            group['y_true'].values,
            group['y_pred'].values,
            output_dir,
            class_names=class_names,
            title=f'{model} ({k}-shot)',
            filename=f'fig5_confusion_{model.replace(" ", "_")}_{k}shot'
        )

    # 额外：绘制全局混淆矩阵（所有模型合并）
def _plot_single_confusion_matrix(y_true, y_pred, output_dir,
    class_names=None, title=None, filename=None):
    """绘制单个混淆矩阵（内部函数）"""
    # 验证数据
    y_true = _validate_and_fix_array(y_true, "y_true")
    y_pred = _validate_and_fix_array(y_pred, "y_pred")

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    if class_names is None:
        class_names = sorted(list(set(y_true)))

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'label': 'Accuracy / Recall'},
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')

    if title:
        ax.set_title(f'Confusion Matrix: {title}', fontsize=15, pad=15)
    else:
        ax.set_title('Confusion Matrix (Normalized)', fontsize=15, pad=15)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # 保存文件
    if filename is None:
        filename = _cfg_require('draw_all_model_pic.py._plot_single_confusion_matrix.filename')

    plt.savefig(os.path.join(output_dir, f'{filename}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {filename}.pdf/png")
# 原有的 _plot_confusion_matrix 改为调用 _plot_single_confusion_matrix
def _plot_confusion_matrix(y_true, y_pred, output_dir, class_names=None):
    """绘制归一化混淆矩阵（向后兼容接口）"""
    _plot_single_confusion_matrix(
    y_true, y_pred, output_dir,
    class_names=class_names,
    title='Normalized',
    filename=_cfg_require('draw_all_model_pic.py._plot_confusion_matrix.filename')
    )
def _validate_and_fix_array(arr, name="array"):
    """验证并修复数组格式"""
    if isinstance(arr, list):
        arr = np.array(arr)
    # 确保是数值类型
    if arr.dtype == object:
        arr = np.array([int(x) if isinstance(x, (np.ndarray, list, np.integer)) else x
                       for x in arr.flatten()])

    # 确保是 1D
    if arr.ndim > 1:
        arr = arr.flatten()

    # 转为 int6<configured>
    arr = arr.astype(np.int64)

    return arr

### <configured>. **可选：同时支持分k_shot对比图**
def _plot_confusion_matrix_comparison(y_true, y_pred, model_names, k_shots,
                                     output_dir, class_names=None):
    """在一张图中对比不同k_shot的混淆矩阵（同一模型）"""

    import pandas as pd
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'model': model_names,
        'k_shot': k_shots
    })

    # 为每个模型单独绘制
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        k_shot_list = sorted(model_df['k_shot'].unique())

        n_cols = len(k_shot_list)
        fig, axes = plt.subplots(1, n_cols, figsize=(8*n_cols, 8))

        if n_cols == 1:
            axes = [axes]

        for idx, k in enumerate(k_shot_list):
            k_df = model_df[model_df['k_shot'] == k]

            cm = confusion_matrix(
                k_df['y_true'].values,
                k_df['y_pred'].values,
                normalize='true'
            )

            if class_names is None:
                class_names = sorted(list(set(k_df['y_true'].values)))

            sns.heatmap(
                cm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True,
                cbar_kws={'label': 'Recall'},
                ax=axes[idx]
            )

            axes[idx].set_title(f'{k}-shot', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted' if idx == n_cols//2 else '')
            axes[idx].set_ylabel('True' if idx == 0 else '')

        plt.suptitle(f'Confusion Matrices: {model}', fontsize=16, y=1.02)
        plt.tight_layout()

        safe_name = model.replace(' ', '_').replace(_cfg_require('draw_all_model_pic.py._plot_confusion_matrix_comparison.path'), '_')
        plt.savefig(os.path.join(output_dir, f'fig5_cm_comparison_{safe_name}.pdf'),
                   bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'fig5_cm_comparison_{safe_name}.png'),
                   bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved comparison for {model}")

def _plot_sensitivity(sensitivity_df, output_dir):
    """
    绘制参数敏感性分析
    sensitivity_df 需包含: 'parameter' (值), 'accuracy' (指标)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    param_name = sensitivity_df.columns[0] # 假设第一列是参数
    x = sensitivity_df[param_name]
    y = sensitivity_df['accuracy'] * 100

    ax.plot(x, y, marker='o', linestyle='-', linewidth=2, color='#e74c3c')

    # 标注最大值
    max_idx = y.idxmax()
    max_x = x[max_idx]
    max_y = y[max_idx]

    ax.annotate(f'Peak: {max_y:.2f}%',
                xy=(max_x, max_y),
                xytext=(max_x, max_y + 1),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_xlabel(f'Parameter: {param_name}', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Sensitivity Analysis on {param_name}', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_sensitivity.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig7_sensitivity.png'), bbox_inches='tight')
    plt.close()


# 阈值敏感度用绘图函数
def _plot_prototype_curves(prototype_df, output_dir,
                          extra_models=None,
                          extra_k_shots=None,
                          extra_active_protos=None):
    """绘制不同模型不同k-shot下的原型数量曲线"""

    fig1, ax1 = plt.subplots(figsize=(8, 6))  # 第一个图
    fig2, ax2 = plt.subplots(figsize=(8, 6))  # 第二个图

    models = prototype_df['model'].unique()
    n_original_models = len(models)

    # ===== 使用 colorblind 调色板并手动映射颜色 =====
    colorblind_palette = sns.color_palette("colorblind", 10)  # 获取完整调色板

    # 定义模型到颜色的映射（基于 colorblind 调色板的索引）
    color_map = {
        'GMM(comp=5)': colorblind_palette[0],
        'IMP': colorblind_palette[3],
        'AMGPN': colorblind_palette[1],
        'Kmeans++(K=7)': colorblind_palette[2]
    }

    # 定义模型到标记点的映射
    marker_map = {
        'GMM(comp=5)': 'o',        # 圆圈
        'IMP': '^',        # 三角形
        'AMGPN': 's',      # 正方形
        'Kmeans++(K=7)': 'D'  # 菱形
    }

    # 备用标记点列表（用于额外模型）

    # 辅助函数：根据模型名匹配颜色
    def get_color(model_name):
        for key in color_map:
            if key.lower() in model_name.lower():
                return color_map[key]
        return colorblind_palette[7]  # 默认灰色

    # 辅助函数：根据模型名匹配标记点
    def get_marker(model_name):
        for key in marker_map:
            if key.lower() in model_name.lower():
                return marker_map[key]
        return 'o'  # 默认圆圈

    # 绘制原始数据（实线）
    for idx, model_name in enumerate(models):
        subset = prototype_df[prototype_df['model'] == model_name].sort_values('k_shot')

        k_vals = subset['k_shot'].values
        active_protos = subset['global_avg_active'].values

        ax1.plot(k_vals, active_protos,
                marker=get_marker(model_name),  # 使用 get_marker 函数
                markersize=8,
                linewidth=2,
                color=get_color(model_name),  # 使用 colorblind 调色板
                label=model_name,
                linestyle='-')

    # 绘制额外数据（虚线）
    if extra_models is not None and extra_k_shots is not None and extra_active_protos is not None:
        for i, (extra_model, extra_k, extra_active) in enumerate(
            zip(extra_models, extra_k_shots, extra_active_protos)
        ):
            idx = n_original_models + i

            ax1.plot(extra_k, extra_active,
                    marker=get_marker(extra_model),  # 使用 markers 列表
                    markersize=8,
                    linewidth=2,
                    color=get_color(extra_model),  # 使用 colorblind 调色板
                    label=extra_model,
                    linestyle='--',  # 虚线
                    alpha=_cfg_require('draw_all_model_pic.py._plot_prototype_curves.alpha__3'))

    ax1.set_xlabel('Number of Shots (K)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Active Prototypes', fontsize=14, fontweight='bold')
    ax1.set_title('Prototype Retention vs K-shot', fontsize=15, pad=15)
    ax1.set_xticks(sorted(prototype_df['k_shot'].unique()))
    ax1.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    ax1.grid(True, alpha=_cfg_require('draw_all_model_pic.py._plot_prototype_curves.alpha'))

    # 绘制原始数据（实线）
    for idx, model_name in enumerate(models):
        subset = prototype_df[prototype_df['model'] == model_name].sort_values('k_shot')

        k_vals = subset['k_shot'].values
        merge_ratios = subset['merge_ratio'].values * 100

        ax2.plot(k_vals, merge_ratios,
                marker=get_marker(model_name),  # 改用 get_marker 函数
                markersize=8,
                linewidth=2,
                color=get_color(model_name),  # 使用 colorblind 调色板
                label=model_name,
                linestyle='-')

#     # 绘制额外数据（虚线）
    if extra_models is not None and extra_k_shots is not None and extra_active_protos is not None:
        for i, (extra_model, extra_k, extra_active) in enumerate(
            zip(extra_models, extra_k_shots, extra_active_protos)
        ):
            idx = n_original_models + i

            extra_k_array = np.array(extra_k)
            extra_active_array = np.array(extra_active)
            merge_ratios = (1 - extra_active_array / extra_k_array) * 100

            ax2.plot(extra_k, merge_ratios,
                    marker=get_marker(model_name),  # 使用 markers 列表
                    markersize=8,
                    linewidth=2,
                    color=get_color(extra_model),  # 使用 colorblind 调色板
                    label=extra_model,
                    linestyle='--',
                    alpha=_cfg_require('draw_all_model_pic.py._plot_prototype_curves.alpha__4'))

    ax2.set_xlabel('Number of Shots (K)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Merge Ratio (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Prototype Merge Ratio vs K-shot', fontsize=15, pad=15)
    ax2.set_xticks(sorted(prototype_df['k_shot'].unique()))
    ax2.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    ax2.grid(True, alpha=_cfg_require('draw_all_model_pic.py._plot_prototype_curves.alpha__2'))

    # 第一个图保存
    plt.figure(fig1.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prototype_retention.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'prototype_retention.png'), bbox_inches='tight')
    plt.close(fig1)

    # 第二个图保存
    plt.figure(fig2.number)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prototype_merge_ratio.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'prototype_merge_ratio.png'), bbox_inches='tight')
    plt.close(fig2)

    print("  ✓ Saved: fig8_prototype_curves.pdf/png and prototype_merge_ratio.pdf/png")

    _plot_per_class_prototype_heatmap(prototype_df, output_dir)
def _plot_per_class_prototype_heatmap(prototype_df, output_dir):
    """绘制每个类的平均原型数热图（可选）"""

    # 提取每类数据
    all_class_data = []
    for _, row in prototype_df.iterrows():
        per_class = row['per_class_avg_active']
        for class_id, avg_active in per_class.items():
            all_class_data.append({
                'model': row['model'],
                'k_shot': row['k_shot'],
                'class': class_id,
                'avg_active': avg_active
            })

    if not all_class_data:
        return
    class_df = pd.DataFrame(all_class_data)
    class_df = class_df.groupby(['model', 'k_shot', 'class'], as_index=False)['avg_active'].mean()


    # 为每个模型绘制热图
    models = class_df['model'].unique()

    for model_name in models:
        model_data = class_df[class_df['model'] == model_name]

        # 创建透视表
        pivot = model_data.pivot(index='class', columns='k_shot', values='avg_active')

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                   cbar_kws={'label': 'Avg Active Prototypes'}, ax=ax)

        ax.set_xlabel('K-shot', fontsize=13, fontweight='bold')
        ax.set_ylabel('Class ID', fontsize=13, fontweight='bold')
        ax.set_title(f'Per-Class Prototype Count: {model_name}', fontsize=15, pad=15)

        plt.tight_layout()
        safe_name = model_name.replace(' ', '_').replace(_cfg_require('draw_all_model_pic.py._plot_per_class_prototype_heatmap.path'), '_')
        plt.savefig(os.path.join(output_dir, f'fig8_per_class_{safe_name}.pdf'),
                   bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'fig8_per_class_{safe_name}.png'),
                   bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: fig8_per_class_{safe_name}.pdf/png")
