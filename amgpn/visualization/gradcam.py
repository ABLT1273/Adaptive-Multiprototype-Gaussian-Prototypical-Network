from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scipy import ndimage
from amgpn.data.preprocessing import crop_and_rescale_symmetric
import pandas as pd
import os

class FrequencyCropAttentionAnalyzer:
    """
    频率裁剪注意力分析器

    核心目标: 证明频率裁剪提升性能的原因
    - 统计每个测试类查询集的注意力分布
    - 对比频率能量与模型关注度
    - 分析裁剪是否使模型更聚焦有效频段
    """
    def __init__(self, model, device, target_layer_name=None):
        target_layer_name = _cfg_resolve('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.__init__.target_layer_name', target_layer_name)
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None

        # 注册hook
        self._register_hooks(target_layer_name)

        # 统计容器 {class_id: {...}}
        self.class_statistics = defaultdict(lambda: {
            'spectrograms': [],           # 原始频谱图
            'gradcam_maps': [],           # Grad-CAM热力图
            'freq_energy_profiles': [],   # 频率能量轮廓
            'freq_attention_profiles': [], # 频率注意力轮廓
            'valid_freq_ranges': [],      # 有效频率范围
            'predictions': [],            # 预测结果
            'confidences': []             # 预测置信度
        })

    def _register_hooks(self, target_layer_name):
        """注册hook提取激活和梯度"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer = dict([*self.model.named_modules()])[target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(self, input_tensor, target_class_idx):
        '\n        生成Grad-CAM热力图\n        \n        Args:\n            input_tensor: [<configured>, C, H, W]\n            target_class_idx: 目标类别索引\n        \n        Returns:\n            gradcam_map: [H, W] 归一化到<configured>-<configured>\n        '
        self.model.zero_grad()

        # 前向传播
        v, s = self.model(input_tensor)

        # 反向传播到目标类
        if target_class_idx < v.shape[1]:
            target = v[:, target_class_idx]
        else:
            target = v.mean()
        target.backward()

        # 计算Grad-CAM
        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
        gradcam = F.relu(gradcam)

        # 上采样到原始尺寸
        gradcam = F.interpolate(
            gradcam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        gradcam = gradcam.squeeze().cpu().numpy()
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)

        return gradcam

    def compute_valid_frequency_range(self, spectrogram, energy_threshold_percentile=None):
        """
        计算有效频率范围（能量显著的频段）

        Args:
            spectrogram: [H, W] 频谱图
            energy_threshold_percentile: 能量阈值百分位

        Returns:
            (start_freq, end_freq): 有效频率范围
        """
        # 沿时间轴平均，得到频率能量轮廓
        energy_threshold_percentile = _cfg_resolve('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.compute_valid_frequency_range.energy_threshold_percentile', energy_threshold_percentile)
        freq_energy = np.mean(spectrogram, axis=1)

        # 计算阈值
        threshold = np.percentile(freq_energy, energy_threshold_percentile)

        # 找到超过阈值的频率范围
        valid_freqs = np.where(freq_energy > threshold)[0]

        if len(valid_freqs) == 0:
            return (0, len(freq_energy))

        return (valid_freqs[0], valid_freqs[-1])

    def collect_sample_statistics(self, sample, true_class, pred_class, confidence):
        '\n        收集单个样本的统计信息\n        \n        Args:\n            sample: [<configured>, C, H, W] 输入样本\n            true_class: 真实类别\n            pred_class: 预测类别\n            confidence: 预测置信度\n        '
        # 生成Grad-CAM
        gradcam_map = self.generate_gradcam(sample, pred_class)

        # 获取频谱图
        spectrogram = sample[0, 0].cpu().numpy()

        # 频率维度轮廓
        freq_energy = np.mean(spectrogram, axis=1)  # [H]
        freq_attention = np.mean(gradcam_map, axis=1)  # [H]

        # 计算有效频率范围
        valid_range = self.compute_valid_frequency_range(spectrogram)

        # 存储到对应类别
        stats = self.class_statistics[true_class]
        stats['spectrograms'].append(spectrogram)
        stats['gradcam_maps'].append(gradcam_map)
        stats['freq_energy_profiles'].append(freq_energy)
        stats['freq_attention_profiles'].append(freq_attention)
        stats['valid_freq_ranges'].append(valid_range)
        stats['predictions'].append(pred_class)
        stats['confidences'].append(confidence)

    def compute_class_metrics(self):
        """
        计算每个类别的统计指标

        Returns:
            metrics: {class_id: {...}}
        """
        metrics = {}

        for class_id, stats in self.class_statistics.items():
            n_samples = len(stats['freq_energy_profiles'])

            if n_samples == 0:
                continue

            # <configured>. 平均频率能量轮廓
            freq_energy_mean = np.mean(stats['freq_energy_profiles'], axis=0)
            freq_energy_std = np.std(stats['freq_energy_profiles'], axis=0)

            # <configured>. 平均频率注意力轮廓
            freq_attention_mean = np.mean(stats['freq_attention_profiles'], axis=0)
            freq_attention_std = np.std(stats['freq_attention_profiles'], axis=0)

            # <configured>. 频率对齐度（皮尔逊相关系数）
            freq_alignment = np.corrcoef(freq_energy_mean, freq_attention_mean)[0, 1]

            # <configured>. 有效频率范围统计
            valid_ranges = stats['valid_freq_ranges']
            start_freqs = [r[0] for r in valid_ranges]
            end_freqs = [r[1] for r in valid_ranges]
            avg_valid_range = (np.mean(start_freqs), np.mean(end_freqs))

            # <configured>. 注意力集中度（在有效频段内的注意力占比）
            attention_concentration = []
            for i in range(n_samples):
                attention = stats['freq_attention_profiles'][i]
                start, end = valid_ranges[i]
                total_attention = np.sum(attention)
                valid_attention = np.sum(attention[start:end+1])
                concentration = valid_attention / (total_attention + 1e-8)
                attention_concentration.append(concentration)

            avg_concentration = np.mean(attention_concentration)
            std_concentration = np.std(attention_concentration)

            # <configured>. 准确率
            predictions = np.array(stats['predictions'])
            accuracy = np.mean(predictions == class_id)

            # <configured>. 平均置信度
            avg_confidence = np.mean(stats['confidences'])

            metrics[class_id] = {
                'n_samples': n_samples,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'freq_alignment': freq_alignment,
                'freq_energy_mean': freq_energy_mean,
                'freq_energy_std': freq_energy_std,
                'freq_attention_mean': freq_attention_mean,
                'freq_attention_std': freq_attention_std,
                'avg_valid_range': avg_valid_range,
                'attention_concentration_mean': avg_concentration,
                'attention_concentration_std': std_concentration,
                'attention_concentration_all': attention_concentration
            }

        return metrics

    def visualize_class_statistics(self, class_id, save_path):
        """
        可视化单个类别的统计分析
        """
        stats = self.class_statistics[class_id]
        n_samples = len(stats['freq_energy_profiles'])

        if n_samples == 0:
            print(f"类别 {class_id} 无样本")
            return

        # 计算统计量
        freq_energy_mean = np.mean(stats['freq_energy_profiles'], axis=0)
        freq_energy_std = np.std(stats['freq_energy_profiles'], axis=0)
        freq_attention_mean = np.mean(stats['freq_attention_profiles'], axis=0)
        freq_attention_std = np.std(stats['freq_attention_profiles'], axis=0)

        # 有效频率范围
        valid_ranges = stats['valid_freq_ranges']
        avg_start = int(np.mean([r[0] for r in valid_ranges]))
        avg_end = int(np.mean([r[1] for r in valid_ranges]))

        # 相关系数
        freq_corr = np.corrcoef(freq_energy_mean, freq_attention_mean)[0, 1]

        # 注意力集中度
        concentrations = []
        for i in range(n_samples):
            attention = stats['freq_attention_profiles'][i]
            start, end = valid_ranges[i]
            total = np.sum(attention)
            valid = np.sum(attention[start:end+1])
            concentrations.append(valid / (total + 1e-8))
        avg_concentration = np.mean(concentrations)

        # 准确率
        predictions = np.array(stats['predictions'])
        accuracy = np.mean(predictions == class_id)

        # 创建图表
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])

        freq_bins = np.arange(len(freq_energy_mean))

        # 归一化
        freq_energy_norm = (freq_energy_mean - freq_energy_mean.min()) / \
                          (freq_energy_mean.max() - freq_energy_mean.min() + 1e-8)
        freq_attention_norm = (freq_attention_mean - freq_attention_mean.min()) / \
                             (freq_attention_mean.max() - freq_attention_mean.min() + 1e-8)

        # 绘制能量轮廓
        ax1.plot(freq_bins, freq_energy_norm, 'b-', linewidth=2.5,
                label='Frequency Energy', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha'))
        ax1.fill_between(freq_bins,
                         freq_energy_norm - freq_energy_std / (freq_energy_mean.max() + 1e-8),
                         freq_energy_norm + freq_energy_std / (freq_energy_mean.max() + 1e-8),
                         color='b', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__2'))

        # 绘制注意力轮廓
        ax1.plot(freq_bins, freq_attention_norm, 'r-', linewidth=2.5,
                label='Model Attention', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__3'))
        ax1.fill_between(freq_bins,
                         freq_attention_norm - freq_attention_std / (freq_attention_mean.max() + 1e-8),
                         freq_attention_norm + freq_attention_std / (freq_attention_mean.max() + 1e-8),
                         color='r', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__4'))

        # 标记有效频率范围
        ax1.axvspan(avg_start, avg_end, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__5'), color='green',
                   label=f'Valid Frequency Range [{avg_start}, {avg_end}]')
        ax1.axvline(avg_start, color='green', linestyle='--', linewidth=1.5, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__6'))
        ax1.axvline(avg_end, color='green', linestyle='--', linewidth=1.5, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__7'))

        ax1.set_xlabel('Frequency Bin', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Normalized Value', fontsize=13, fontweight='bold')
        ax1.set_title(
            f'Class {class_id}: Frequency Energy vs Model Attention\nAlignment r={freq_corr:.3f} | Concentration={avg_concentration:.3f} | Accuracy={accuracy * 100:.1f}% | N={n_samples}',
            fontsize=15, fontweight='bold'
        )
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__8'))

        ax2 = fig.add_subplot(gs[1, 0])

        # 计算有效频段内外的注意力分布
        valid_attention_all = []
        invalid_attention_all = []

        for i in range(n_samples):
            attention = stats['freq_attention_profiles'][i]
            start, end = valid_ranges[i]

            valid_attention_all.extend(attention[start:end+1])
            invalid_attention_all.extend(
                np.concatenate([attention[:start], attention[end+1:]])
            )

        ax2.hist([valid_attention_all, invalid_attention_all],
                bins=30, label=['Valid Freq', 'Invalid Freq'],
                color=['green', 'red'], alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__9'), edgecolor='black')
        ax2.set_xlabel('Attention Value', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Attention Distribution: Valid vs Invalid Freq', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__10'), axis='y')

        ax3 = fig.add_subplot(gs[1, 1])

        ax3.hist(concentrations, bins=20, color='skyblue', edgecolor='black', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__11'))
        ax3.axvline(avg_concentration, color='r', linestyle='--',
                   linewidth=2, label=f'Mean={avg_concentration:.3f}')
        ax3.axvline(0.5, color='orange', linestyle=':', linewidth=2,
                   label='Baseline (50%)')

        ax3.set_xlabel('Attention Concentration (Valid Freq)', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Attention Concentration Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__12'), axis='y')

        ax4 = fig.add_subplot(gs[1, 2])

        energy_matrix = np.array(stats['freq_energy_profiles']).T  # [H, N]
        im1 = ax4.imshow(energy_matrix, aspect='auto', cmap='viridis', origin='lower')
        ax4.axhline(avg_start, color='green', linestyle='--', linewidth=1.5, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__13'))
        ax4.axhline(avg_end, color='green', linestyle='--', linewidth=1.5, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__14'))
        ax4.set_xlabel('Sample Index', fontsize=11)
        ax4.set_ylabel('Frequency Bin', fontsize=11)
        ax4.set_title('Energy Heatmap (All Samples)', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[2, 0])

        attention_matrix = np.array(stats['freq_attention_profiles']).T  # [H, N]
        im2 = ax5.imshow(attention_matrix, aspect='auto', cmap='hot', origin='lower')
        ax5.axhline(avg_start, color='cyan', linestyle='--', linewidth=1.5, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__15'))
        ax5.axhline(avg_end, color='cyan', linestyle='--', linewidth=1.5, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__16'))
        ax5.set_xlabel('Sample Index', fontsize=11)
        ax5.set_ylabel('Frequency Bin', fontsize=11)
        ax5.set_title('Attention Heatmap (All Samples)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

        ax6 = fig.add_subplot(gs[2, 1])

        sample_correlations = []
        for i in range(n_samples):
            energy = stats['freq_energy_profiles'][i]
            attention = stats['freq_attention_profiles'][i]
            corr = np.corrcoef(energy, attention)[0, 1]
            sample_correlations.append(corr)

        ax6.hist(sample_correlations, bins=20, color='lightcoral', edgecolor='black', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__17'))
        ax6.axvline(np.mean(sample_correlations), color='r', linestyle='--',
                   linewidth=2, label=f'Mean={np.mean(sample_correlations):.3f}')

        ax6.set_xlabel('Energy-Attention Correlation', fontsize=11)
        ax6.set_ylabel('Count', fontsize=11)
        ax6.set_title('Sample-level Alignment Distribution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__18'), axis='y')

        ax7 = fig.add_subplot(gs[2, 2])

        start_freqs = [r[0] for r in valid_ranges]
        end_freqs = [r[1] for r in valid_ranges]

        ax7.scatter(start_freqs, end_freqs, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__19'), s=50, c='purple')
        ax7.axhline(avg_end, color='r', linestyle='--', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__20'), label=f'Mean End={avg_end}')
        ax7.axvline(avg_start, color='g', linestyle='--', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__21'), label=f'Mean Start={avg_start}')

        ax7.set_xlabel('Start Frequency', fontsize=11)
        ax7.set_ylabel('End Frequency', fontsize=11)
        ax7.set_title('Valid Frequency Range Distribution', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__22'))

        # 选择: 最高置信度、中等置信度、最低置信度
        confidences = np.array(stats['confidences'])
        sorted_indices = np.argsort(confidences)

        representative_indices = [
            sorted_indices[-1],  # 最高置信度
            sorted_indices[len(sorted_indices)//2],  # 中等
            sorted_indices[0]    # 最低
        ]

        titles = ['Highest Confidence', 'Medium Confidence', 'Lowest Confidence']

        for idx, (sample_idx, title) in enumerate(zip(representative_indices, titles)):
            ax = fig.add_subplot(gs[3, idx])

            spectrogram = stats['spectrograms'][sample_idx]
            gradcam = stats['gradcam_maps'][sample_idx]

            # 叠加显示
            ax.imshow(spectrogram, aspect='auto', cmap='gray', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__23'), origin='lower')
            im = ax.imshow(gradcam, aspect='auto', cmap='jet', alpha=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.alpha__24'), origin='lower')

            start, end = valid_ranges[sample_idx]
            ax.axhline(start, color='green', linestyle='--', linewidth=2)
            ax.axhline(end, color='green', linestyle='--', linewidth=2)

            pred = stats['predictions'][sample_idx]
            conf = stats['confidences'][sample_idx]

            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{title}\nPred={pred}, Conf={conf:.3f}', fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.savefig(save_path, dpi=_cfg_require('draw_Grad_CAM.py.FrequencyCropAttentionAnalyzer.visualize_class_statistics.size_or_budget'), bbox_inches='tight')
        plt.close()

        print(f"类别 {class_id} 统计图已保存至: {save_path}")


def test_frequency_crop_statistics(trainer, test_loader, device, crop_ratio_h,crop_ratio_l, resize,save_dir=None):
    """
    对测试集生成频率裁剪注意力统计分析

    Args:
        trainer: GPNTrainer实例（包含模型）
        test_loader: 测试数据加载器（配置为单个元任务）
        device: 设备
        save_dir: 结果保存路径
    """
    save_dir = _cfg_resolve('draw_Grad_CAM.py.test_frequency_crop_statistics.save_dir', save_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # 初始化分析器
    analyzer = FrequencyCropAttentionAnalyzer(
        model=trainer.model,
        device=device,
        target_layer_name=_cfg_require('draw_Grad_CAM.py.test_frequency_crop_statistics.target_layer_name')
    )

    print("="*80)
    print("频率裁剪注意力统计分析")
    print("="*80)

    # 获取测试任务
    for task_id, meta_task in enumerate(test_loader):
        if task_id > 0:  # 只处理第一个任务
            break

        with torch.no_grad():
            support_signals, support_labels, query_signals, query_labels = meta_task
            print(f"\n任务配置:")
            print(f"  支持集: {support_signals.shape}")
            print(f"  查询集: {query_signals.shape}")
            # 移动到设备
            support_signals = support_signals[task_id].to(device).float().unsqueeze(1)
            support_labels = support_labels[task_id].to(device)
            query_signals = query_signals[task_id].to(device).float().unsqueeze(1)
            query_labels = query_labels[task_id].to(device)

            # 数据增强
            if crop_ratio_h != 0.0 or crop_ratio_l != 0.0:
                processed_support_list = []
                for signal in support_signals.unbind(0):
                    processed_support_list.append(crop_and_rescale_symmetric(signal, crop_ratio_h,crop_ratio_l, resize))
                support_signals = torch.stack(processed_support_list, dim=0)

                processed_query_list = []
                for signal in query_signals.unbind(0):
                    processed_query_list.append(crop_and_rescale_symmetric(signal, crop_ratio_h,crop_ratio_l, resize))
                query_signals = torch.stack(processed_query_list, dim=0)

            # 前向传播
            support_v, support_s = trainer.model(support_signals)
            query_v, query_s = trainer.model(query_signals)

            # 标签重映射
            unique_labels = torch.unique(support_labels)
            n_ways = len(unique_labels)
            label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

            remapped_support_labels = torch.tensor(
                [label_mapping[label.item()] for label in support_labels],
                device=device
            )
            remapped_query_labels = torch.tensor(
                [label_mapping[label.item()] for label in query_labels],
                device=device
            )

            # ========== 关键修复：根据use_multi选择不同的原型准备方式 ==========
            # 多原型模式：使用所有support样本作为原型
            prototypes = support_v  # [n_ways * k_shot, feature_dim]
            precision_matrices = trainer.compute_precision_matrices_from_support(support_s)

            # 调用损失函数（任务内合并）
            output = trainer.loss_fn(
                query_v,
                prototypes,
                precision_matrices,
                remapped_query_labels,
                epoch=trainer.current_epoch
            )

            loss = output['loss']
            probabilities = output['probabilities']

            # 计算准确率
            predictions = torch.argmax(probabilities, dim=1)
            accuracy = (predictions == remapped_query_labels).float().mean()
            confidences = torch.max(probabilities, dim=1)[0]
            print(f"\n任务准确率: {accuracy*100:.2f}%")

        print(f"\n收集查询集样本统计...")

        for i in tqdm(range(len(query_signals)), desc="处理查询样本"):
            query_sample = query_signals[i:i+1]
            true_class = query_labels[i].item()
            pred_class = predictions[i].item()
            confidence = confidences[i].item()

            analyzer.collect_sample_statistics(
                sample=query_sample,
                true_class=true_class,
                pred_class=pred_class,
                confidence=confidence
            )

        # 计算类别级指标
        print(f"\n计算类别级统计指标...")
        metrics = analyzer.compute_class_metrics()

        # 打印汇总
        print("\n" + "="*80)
        print("类别级统计汇总")
        print("="*80)

        output_metrics_to_table_and_csv_optimized(metrics, file_name=save_dir/"class_metrics_report_optimized.csv")

        # 生成每个类别的可视化
        print(f"\n生成可视化...")
        for class_id in sorted(metrics.keys()):
            analyzer.visualize_class_statistics(
                class_id,
                save_dir / f'class_{class_id}_statistics.png'
            )

        print("\n" + "="*80)
        print("分析完成!")
        print(f"结果已保存至: {save_dir}")
        print("="*80)


def output_metrics_to_table_and_csv_optimized(metrics_data, file_name=None):
    """
    使用列表推导式优化后的函数，将指标转换为DataFrame，打印表格，并保存为CSV文件。
    """

    # --- <configured>. 使用列表推导式替换原有的for循环 ---
    file_name = _cfg_resolve('draw_Grad_CAM.py.output_metrics_to_table_and_csv_optimized.file_name', file_name)
    data_list = [
        {
            "类别ID": class_id,
            "样本数": m['n_samples'],
            "准确率 (%)": f"{m['accuracy']*100:.2f}",
            "平均置信度": f"{m['avg_confidence']:.4f}",
            "频率对齐度": f"{m['freq_alignment']:.4f}",
            "有效频率范围": f"[{int(m['avg_valid_range'][0])}, {int(m['avg_valid_range'][1])}]",
            "注意力集中度": f"{m['attention_concentration_mean']:.4f} ± {m['attention_concentration_std']:.4f}",
            "准确率_raw": m['accuracy'],
            "平均置信度_raw": m['avg_confidence'],
        }
        # 核心迭代逻辑：按排序后的键迭代，m 是对应的值
        for class_id, m in sorted(metrics_data.items())
    ]

    # <configured>. 创建 DataFrame
    df = pd.DataFrame(data_list)

    # 定义打印时要显示的列（移除原始数值列）
    display_columns = [
        "类别ID",
        "样本数",
        "准确率 (%)",
        "平均置信度",
        "频率对齐度",
        "有效频率范围",
        "注意力集中度"
    ]
    df_display = df[display_columns]

    # --- 打印表格和保存CSV（与原代码相同） ---
    print("## 📊 类别识别性能报告")
    print(df_display.to_markdown(index=False))

    try:
        df.to_csv(file_name, index=False, encoding='utf-8')
        print(f"\n✅ 报告已成功保存到文件：{os.path.abspath(file_name)}")
    except Exception as e:
        print(f"\n❌ 保存文件失败：{e}")

# ==================== 使用示例 ====================

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from amgpn.data.wavelet_episodes import UAVDataset, MetaDataset
    from amgpn.legacy.v4.fixed_multiprototype import GPN_Optimized, GPNTrainer

    # 参数配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET_PATH = _cfg_require('draw_Grad_CAM.py.module.DATASET_PATH')
    MODEL_PATH = _cfg_require('draw_Grad_CAM.py.module.MODEL_PATH')
    SAVE_DIR = _cfg_require('draw_Grad_CAM.py.module.SAVE_DIR')

    N_WAY = _cfg_require('draw_Grad_CAM.py.module.N_WAY')
    K_SHOT = _cfg_require('draw_Grad_CAM.py.module.K_SHOT')
    Q_QUERY = _cfg_require('draw_Grad_CAM.py.module.Q_QUERY')  # 每个类<configured>个查询样本
    SEED = _cfg_require('draw_Grad_CAM.py.module.SEED')

    # 加载数据
    print("加载测试数据...")
    test_dataset = UAVDataset(
        data_dir_path=DATASET_PATH,
        seed=SEED,
        is_train=False,
        train_ratio=_cfg_require('draw_Grad_CAM.py.module.train_ratio'),
        max_sample_count=_cfg_require('draw_Grad_CAM.py.module.max_sample_count'))

    # 基线（无裁剪）
    test_frequency_crop_statistics(
        trainer=trainer,
        test_loader=test_loader,
        device=device,
        crop_ratio_h=_cfg_require('draw_Grad_CAM.py.module.crop_ratio_h'),
        crop_ratio_l=_cfg_require('draw_Grad_CAM.py.module.crop_ratio_l'),
        save_dir=_cfg_require('draw_Grad_CAM.py.module.save_dir')
    )

    # 裁剪<configured>并resize
    test_frequency_crop_statistics(
        trainer=trainer,
        test_loader=test_loader,
        device=device,
        crop_ratio_h=_cfg_require('draw_Grad_CAM.py.module.crop_ratio_h__2'),
        crop_ratio_l=_cfg_require('draw_Grad_CAM.py.module.crop_ratio_l__2'),
        resize=_cfg_require('draw_Grad_CAM.py.module.resize'),
        save_dir=_cfg_require('draw_Grad_CAM.py.module.save_dir__2')
    )

    # 裁剪<configured>不resize
    test_frequency_crop_statistics(
        trainer=trainer,
        test_loader=test_loader,
        device=device,
        crop_ratio_h=_cfg_require('draw_Grad_CAM.py.module.crop_ratio_h__3'),
        crop_ratio_l=_cfg_require('draw_Grad_CAM.py.module.crop_ratio_l__3'),        resize=_cfg_require('draw_Grad_CAM.py.module.resize__2'),
        save_dir=_cfg_require('draw_Grad_CAM.py.module.save_dir__3')
    )
