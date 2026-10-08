from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch

class PrototypeActivationAnalyzer:
    """
    用于分析原型激活模式的辅助类
    """

    def __init__(self, results_df):
        """
        初始化分析器

        Args:
            results_df: 包含测试结果的DataFrame
        """
        self.results_df = results_df
        self.model_names = results_df['Model'].unique()
        self.k_shots = sorted(results_df['K-shot'].unique())

    def analyze_prototype_efficiency(self):
        """
        分析原型使用效率

        Returns:
            efficiency_df: 包含效率指标的DataFrame
        """
        efficiency_data = []

        for _, row in self.results_df.iterrows():
            model = row['Model']
            k_shot = row['K-shot']
            avg_active = row['Avg_Active_Prototypes']

            # 计算原型利用率
            utilization = avg_active / k_shot if k_shot > 0 else 0

            # 计算冗余度（<configured> - 利用率）
            redundancy = 1 - utilization

            efficiency_data.append({
                'Model': model,
                'K-shot': k_shot,
                'Utilization': utilization,
                'Redundancy': redundancy,
                'Active_Ratio': avg_active / k_shot
            })

        efficiency_df = pd.DataFrame(efficiency_data)
        return efficiency_df

    def plot_prototype_heatmap(self, save_path=None):
        """
        绘制原型激活热力图

        Args:
            save_path: 保存路径
        """
        # 创建数据透视表
        pivot_data = self.results_df.pivot(
            index='Model',
            columns='K-shot',
            values='Avg_Active_Prototypes'
        )

        # 创建热力图
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Average Active Prototypes'},
            linewidths=0.5,
            linecolor='gray'
        )

        plt.title('Prototype Activation Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('K-shot', fontsize=12)
        plt.ylabel('Model', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=_cfg_require('prototype_analysis.py.PrototypeActivationAnalyzer.plot_prototype_heatmap.size_or_budget'), bbox_inches='tight')
        plt.show()

    def plot_efficiency_comparison(self, save_path=None):
        """
        绘制效率比较图

        Args:
            save_path: 保存路径
        """
        efficiency_df = self.analyze_prototype_efficiency()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 图<configured>: 原型利用率
        for model in self.model_names:
            model_data = efficiency_df[efficiency_df['Model'] == model]
            ax1.plot(
                model_data['K-shot'],
                model_data['Utilization'],
                marker='o',
                label=model,
                linewidth=2,
                markersize=8
            )

        ax1.set_xlabel('K-shot', fontsize=12)
        ax1.set_ylabel('Prototype Utilization Rate', fontsize=12)
        ax1.set_title('Prototype Utilization Efficiency', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=_cfg_require('prototype_analysis.py.PrototypeActivationAnalyzer.plot_efficiency_comparison.alpha'))
        ax1.set_ylim([0, 1.1])
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=_cfg_require('prototype_analysis.py.PrototypeActivationAnalyzer.plot_efficiency_comparison.alpha__2'))

        # 图<configured>: 准确率 vs 原型激活数的关系
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.model_names)))

        for idx, model in enumerate(self.model_names):
            model_data = self.results_df[self.results_df['Model'] == model]
            ax2.scatter(
                model_data['Avg_Active_Prototypes'],
                model_data['Accuracy_Mean'],
                s=100,
                alpha=_cfg_require('prototype_analysis.py.PrototypeActivationAnalyzer.plot_efficiency_comparison.alpha__4'),
                label=model,
                color=colors[idx]
            )

            # 添加趋势线
            z = np.polyfit(
                model_data['Avg_Active_Prototypes'],
                model_data['Accuracy_Mean'],
                1
            )
            p = np.poly1d(z)
            x_trend = np.linspace(
                model_data['Avg_Active_Prototypes'].min(),
                model_data['Avg_Active_Prototypes'].max(),
                100
            )
            ax2.plot(x_trend, p(x_trend), '--', alpha=_cfg_require('prototype_analysis.py.PrototypeActivationAnalyzer.plot_efficiency_comparison.alpha__5'), color=colors[idx])

        ax2.set_xlabel('Average Active Prototypes', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy vs Prototype Activation', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=_cfg_require('prototype_analysis.py.PrototypeActivationAnalyzer.plot_efficiency_comparison.alpha__3'))
        ax2.set_ylim([0.6, 1.0])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=_cfg_require('prototype_analysis.py.PrototypeActivationAnalyzer.plot_efficiency_comparison.size_or_budget'), bbox_inches='tight')
        plt.show()

    def generate_statistical_report(self):
        """
        生成统计报告

        Returns:
            report: 统计报告字符串
        """
        report = []
        report.append("="*80)
        report.append("PROTOTYPE ACTIVATION STATISTICAL REPORT")
        report.append("="*80)

        # <configured>. 整体统计
        report.append("\n1. Overall Statistics:")
        report.append("-"*40)

        for model in self.model_names:
            model_data = self.results_df[self.results_df['Model'] == model]

            avg_acc = model_data['Accuracy_Mean'].mean()
            avg_proto = model_data['Avg_Active_Prototypes'].mean()

            report.append(f"\n{model}:")
            report.append(f"  Average Accuracy: {avg_acc:.4f}")
            report.append(f"  Average Active Prototypes: {avg_proto:.2f}")

            # 计算相关性
            if len(model_data) > 1:
                corr, p_value = stats.pearsonr(
                    model_data['Avg_Active_Prototypes'],
                    model_data['Accuracy_Mean']
                )
                report.append(f"  Correlation (Proto vs Acc): {corr:.4f} (p={p_value:.4f})")

        # <configured>. K-shot分析
        report.append("\n2. K-shot Analysis:")
        report.append("-"*40)

        for k in self.k_shots:
            k_data = self.results_df[self.results_df['K-shot'] == k]
            report.append(f"\nK={k}:")

            best_model = k_data.loc[k_data['Accuracy_Mean'].idxmax(), 'Model']
            best_acc = k_data['Accuracy_Mean'].max()

            most_efficient = k_data.loc[
                (k_data['Avg_Active_Prototypes'] / k).idxmin(), 'Model'
            ]

            report.append(f"  Best Accuracy: {best_model} ({best_acc:.4f})")
            report.append(f"  Most Efficient: {most_efficient}")

        # <configured>. 原型激活模式
        report.append("\n3. Prototype Activation Patterns:")
        report.append("-"*40)

        efficiency_df = self.analyze_prototype_efficiency()

        for model in self.model_names:
            model_eff = efficiency_df[efficiency_df['Model'] == model]
            avg_util = model_eff['Utilization'].mean()

            report.append(f"\n{model}:")
            report.append(f"  Average Utilization: {avg_util:.2%}")

            # 判断激活模式
            if avg_util < 0.5:
                pattern = "High Redundancy (>50% prototypes inactive)"
            elif avg_util < 0.8:
                pattern = "Moderate Efficiency"
            else:
                pattern = "High Efficiency (Most prototypes active)"

            report.append(f"  Pattern: {pattern}")

        report.append("\n" + "="*80)

        return "\n".join(report)


def compute_prototype_statistics(trainer, support_v, support_labels, n_ways, k_shot):
    """
    计算原型相关的统计信息

    Args:
        trainer: Trainer实例
        support_v: 支持集特征
        support_labels: 支持集标签
        n_ways: 类别数
        k_shot: 每类样本数

    Returns:
        stats: 统计信息字典
    """
    stats = {
        'inter_class_distance': [],
        'intra_class_distance': [],
        'prototype_diversity': []
    }

    with torch.no_grad():
        # 计算每个类的原型
        unique_labels = torch.unique(support_labels)
        class_prototypes = []

        for label in unique_labels:
            mask = support_labels == label
            class_features = support_v[mask]

            # 如果是多原型模式，保留所有样本
            if hasattr(trainer, 'use_multi') and trainer.use_multi:
                class_prototypes.append(class_features)
            else:
                # 单原型模式，计算均值
                proto = class_features.mean(dim=0, keepdim=True)
                class_prototypes.append(proto)

        # 计算类间距离
        for i in range(len(class_prototypes)):
            for j in range(i+1, len(class_prototypes)):
                if class_prototypes[i].shape[0] == 1:
                    # 单原型
                    dist = torch.norm(
                        class_prototypes[i] - class_prototypes[j]
                    ).item()
                else:
                    # 多原型，计算中心点距离
                    center_i = class_prototypes[i].mean(dim=0)
                    center_j = class_prototypes[j].mean(dim=0)
                    dist = torch.norm(center_i - center_j).item()

                stats['inter_class_distance'].append(dist)

        # 计算类内距离（仅对多原型模式）
        if hasattr(trainer, 'use_multi') and trainer.use_multi:
            for protos in class_prototypes:
                if protos.shape[0] > 1:
                    # 计算类内原型间的平均距离
                    dists = []
                    for i in range(protos.shape[0]):
                        for j in range(i+1, protos.shape[0]):
                            d = torch.norm(protos[i] - protos[j]).item()
                            dists.append(d)

                    if dists:
                        stats['intra_class_distance'].append(np.mean(dists))

        # 计算原型多样性（特征的标准差）
        all_protos = torch.cat(
            [p.mean(dim=0, keepdim=True) if p.shape[0] > 1 else p
             for p in class_prototypes],
            dim=0
        )
        diversity = torch.std(all_protos, dim=0).mean().item()
        stats['prototype_diversity'] = diversity

    return stats


# 示例：如何使用分析器
def analyze_results(csv_path):
    """
    分析已保存的结果

    Args:
        csv_path: CSV文件路径
    """
    # 读取结果
    results_df = pd.read_csv(csv_path)

    # 创建分析器
    analyzer = PrototypeActivationAnalyzer(results_df)

    # 生成各种分析
    print("\nGenerating Analysis...")

    # <configured>. 统计报告
    report = analyzer.generate_statistical_report()
    print(report)

    # <configured>. 效率分析
    efficiency_df = analyzer.analyze_prototype_efficiency()
    print("\nPrototype Efficiency Analysis:")
    print(efficiency_df.to_string(index=False))

    # <configured>. 可视化
    analyzer.plot_prototype_heatmap(save_path=_cfg_require('prototype_analysis.py.analyze_results.save_path'))
    analyzer.plot_efficiency_comparison(save_path=_cfg_require('prototype_analysis.py.analyze_results.save_path__2'))

    return analyzer


if __name__ == "__main__":
    # 示例用法
    print("Prototype Activation Analysis Module Loaded")
    print("Use analyze_results('your_results.csv') to analyze saved results")
