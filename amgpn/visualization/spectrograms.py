from amgpn.data.preprocessing import crop_and_rescale_symmetric
from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
import cv2
from pathlib import Path
import random

class GradCAMAnalyzer:
    """
    Grad-CAM分析器 - 用于判断模型关注频谱特征还是纹理特征
    """
    def __init__(self, model, target_layer_name=None):
        """
        Args:
            model: 训练好的GPN模型
            target_layer_name: 目标层名称 (默认'block4'，最后一个特征层)
        """
        target_layer_name = _cfg_resolve('data_feature_show.py.GradCAMAnalyzer.__init__.target_layer_name', target_layer_name)
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name

        # 用于存储梯度和激活
        self.gradients = None
        self.activations = None

        # 注册hook
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向传播的hook"""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 获取目标层
        target_layer = dict([*self.model.named_modules()])[self.target_layer_name]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_gradcam(self, input_tensor, class_idx):
        '\n        生成Grad-CAM热力图\n        \n        Args:\n            input_tensor: [<configured>, C, H, W] 输入张量\n            class_idx: 目标类别索引\n            \n        Returns:\n            gradcam_map: [H, W] Grad-CAM热力图 (归一化到<configured>-<configured>)\n        '
        self.model.zero_grad()

        # 前向传播
        v, s = self.model(input_tensor)

        # 反向传播到目标类
        # 在元学习中，我们关注v特征对原型距离的影响
        # 这里简化为对v[class_idx]的梯度
        target = v[:, class_idx] if class_idx < v.shape[1] else v.mean()
        target.backward()

        # 计算Grad-CAM
        # gradients: [<configured>, C, H', W']
        # activations: [<configured>, C, H', W']
        gradients = self.gradients
        activations = self.activations

        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [<configured>, C, <configured>, <configured>]

        # 加权求和
        gradcam = torch.sum(weights * activations, dim=1, keepdim=True)  # [<configured>, <configured>, H', W']
        gradcam = F.relu(gradcam)  # 只保留正值

        # 上采样到原始尺寸
        gradcam = F.interpolate(
            gradcam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 归一化到<configured>-<configured>
        gradcam = gradcam.squeeze().cpu().numpy()
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)

        return gradcam


class FeatureAnalyzer:
    """
    特征类型分析器 - 判断模型学到的是频谱特征还是纹理特征
    """
    @staticmethod
    def compute_energy_mask(spectrogram, threshold_percentile=None):
        '\n        计算能量块mask\n        \n        Args:\n            spectrogram: [H, W] 频谱图\n            threshold_percentile: 能量阈值百分位\n            \n        Returns:\n            energy_mask: [H, W] 二值mask (<configured>=能量块区域)\n        '
        threshold_percentile = _cfg_resolve('data_feature_show.py.FeatureAnalyzer.compute_energy_mask.threshold_percentile', threshold_percentile)
        threshold = np.percentile(spectrogram, threshold_percentile)
        energy_mask = (spectrogram > threshold).astype(np.uint8)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        energy_mask = cv2.morphologyEx(energy_mask, cv2.MORPH_CLOSE, kernel)

        return energy_mask

    @staticmethod
    def edge_vs_center_activation(gradcam_map, energy_mask):
        '\n        计算边缘激活 vs 中心激活比例\n        \n        Returns:\n            ratio: 边缘/中心激活比 \n                   > <configured>: 主要关注边缘 (纹理特征)\n                   < <configured>: 主要关注中心 (频谱特征)\n        '
        # 生成边缘mask
        edge_mask = binary_dilation(energy_mask) & ~binary_erosion(energy_mask)

        # 生成中心mask
        center_mask = binary_erosion(energy_mask, iterations=2)

        # 计算激活能量分布
        edge_activation = np.sum(gradcam_map * edge_mask)
        center_activation = np.sum(gradcam_map * center_mask)

        ratio = edge_activation / (center_activation + 1e-8)

        return ratio, edge_mask, center_mask

    @staticmethod
    def analyze_activation_centers(gradcam_map, energy_mask):
        """
        分析激活中心与能量块中心的对齐程度

        Returns:
            center_alignment: 平均距离 (值越小说明越对齐中心)
        """
        # 标记连通域
        labeled_mask, num_blocks = ndimage.label(energy_mask)

        if num_blocks == 0:
            return float('inf')

        # 计算每个能量块的激活质心和能量质心
        distances = []
        for block_id in range(1, num_blocks + 1):
            block_mask = (labeled_mask == block_id)

            # 激活质心
            activation_center = ndimage.center_of_mass(
                gradcam_map * block_mask
            )

            # 能量质心
            energy_center = ndimage.center_of_mass(block_mask)

            # 计算距离
            if activation_center and energy_center:
                dist = np.linalg.norm(
                    np.array(activation_center) - np.array(energy_center)
                )
                distances.append(dist)

        return np.mean(distances) if distances else float('inf')

    @staticmethod
    def compute_frequency_attention_score(gradcam_map):
        """
        计算频率维度的注意力得分

        Returns:
            freq_score: 频率注意力分数 (越高说明越关注频率维度)
        """
        # 沿时间维度平均，得到频率轮廓
        freq_profile = np.mean(gradcam_map, axis=1)

        # 计算频率轮廓的变异系数 (CV)
        # 如果模型关注频谱结构，频率轮廓应该有明显的峰值
        freq_std = np.std(freq_profile)
        freq_mean = np.mean(freq_profile)
        freq_cv = freq_std / (freq_mean + 1e-8)

        return freq_cv


def test_gradcam_analysis(
    model,
    test_loader,
    device,
    save_dir=None,
    n_way=None,
    k_shot=None,
    q_query=None,
    num_samples=None,
    use_multi_prototype=None,
    prototypes_per_class=None,
    crop_ratio_h=None,
    crop_ratio_l=None
):
    '\n    执行Grad-CAM特征分析测试\n    \n    Args:\n        model: 训练好的模型\n        test_loader: 测试数据加载器 (应该配置为<configured>-way <configured>-shot <configured>-query)\n        device: 设备\n        save_dir: 结果保存路径\n        n_way: 类别数\n        k_shot: 支持集样本数\n        q_query: 查询集样本数\n        num_samples: 分析的查询样本数量\n        use_multi_prototype: 是否使用多原型 (应与训练时一致)\n        prototypes_per_class: 每个类的原型数量 (应与训练时一致)\n        crop_ratio_h:样本高频段裁剪比例\n        crop_ratio_l:样本低频段裁剪比例\n\n    '
    save_dir = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.save_dir', save_dir)
    n_way = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.n_way', n_way)
    k_shot = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.k_shot', k_shot)
    q_query = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.q_query', q_query)
    num_samples = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.num_samples', num_samples)
    use_multi_prototype = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.use_multi_prototype', use_multi_prototype)
    prototypes_per_class = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.prototypes_per_class', prototypes_per_class)
    crop_ratio_h = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.crop_ratio_h', crop_ratio_h)
    crop_ratio_l = _cfg_resolve('data_feature_show.py.test_gradcam_analysis.crop_ratio_l', crop_ratio_l)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # 初始化分析器
    gradcam_analyzer = GradCAMAnalyzer(model, target_layer_name=_cfg_require('data_feature_show.py.test_gradcam_analysis.target_layer_name'))
    feature_analyzer = FeatureAnalyzer()

    # 统计结果
    all_metrics = {
        'edge_center_ratios': [],
        'center_alignments': [],
        'freq_scores': []
    }

    print("="*80)
    print("开始Grad-CAM特征分析测试")
    print(f"任务配置: {n_way}-way {k_shot}-shot {q_query}-query")
    print(f"分析样本数: {num_samples}")
    print("="*80)

    for task_id, meta_task in enumerate(test_loader):
        if task_id > 0:  # 只测试第一个任务
            break

        support_signals, support_labels, query_signals, query_labels = meta_task

        # 移动到设备
        support_signals = support_signals.to(device).float().permute(1, 0, 2, 3)
        support_labels = support_labels.to(device).flatten()
        query_signals = query_signals.to(device).float().permute(1, 0, 2, 3)
        query_labels = query_labels.to(device).flatten()

        processed_support_list = []
        for signal in support_signals.unbind(0):
            # 对每个信号应用遮盖函数，并添加到列表中
            processed_support_list.append(crop_and_rescale_symmetric(signal, crop_ratio_h,crop_ratio_l))

        # 将处理后的信号重新堆叠成 [B, <configured>, <configured>, <configured>] 的 Tensor
        support_signals = torch.stack(processed_support_list, dim=0)

        # <configured>. 处理 query_signals
        processed_query_list = [] # 将列表名改为 processed_query_list 更清晰
        for signal in query_signals.unbind(0):
            # 对每个信号应用遮盖函数
            processed_query_list.append(crop_and_rescale_symmetric(signal,crop_ratio_h,crop_ratio_l))

        # 将处理后的信号重新堆叠成 [B, <configured>, <configured>, <configured>] 的 Tensor
        query_signals = torch.stack(processed_query_list, dim=0)
        print(f"\n任务 {task_id + 1}:")
        print(f"支持集: {support_signals.shape}")
        print(f"查询集: {query_signals.shape}")

        with torch.no_grad():
            # 前向传播获取预测
            support_v, support_s = model(support_signals)
            query_v, query_s = model(query_signals)

            # 标签重映射
            unique_labels = torch.unique(support_labels)
            label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
            remapped_support_labels = torch.tensor(
                [label_mapping[label.item()] for label in support_labels],
                device=device
            )
            remapped_query_labels = torch.tensor(
                [label_mapping[label.item()] for label in query_labels],
                device=device
            )

            if use_multi_prototype:
                # === 多原型模式 ===
                from amgpn.legacy.v4.fixed_multiprototype import GPNTrainer

                # 创建临时trainer (用于调用compute_multi_gaussian_prototypes方法)
                temp_trainer = GPNTrainer(
                    model=model,
                    device=device,
                    use_multi=_cfg_require('data_feature_show.py.test_gradcam_analysis.use_multi'),
                    prototypes_per_class=prototypes_per_class
                )

                # 计算多高斯原型
                prototypes, precision_matrices, prototype_assignments = \
                    temp_trainer.compute_multi_gaussian_prototypes(
                        support_v,
                        support_s,
                        remapped_support_labels,
                        n_ways=n_way,
                        k_shot=k_shot
                    )

                # 计算距离 (使用loss_fn的distance_metric)
                all_distances = temp_trainer.loss_fn.distance_metric(
                    query_v,
                    prototypes,
                    precision_matrices
                )

                # 重塑距离矩阵: [N_query, n_ways * prototypes_per_class] -> [N_query, n_ways, prototypes_per_class]
                distances_reshaped = all_distances.view(
                    len(query_v),
                    n_way,
                    prototypes_per_class
                )

                # 每个类别取最小距离 (最近原型)
                min_distances, closest_proto_idx = torch.min(distances_reshaped, dim=2)

                # 预测类别
                predictions = torch.argmin(min_distances, dim=1)

                print(f"使用多原型模式")

            else:
                # === 单原型模式 ===
                from amgpn.legacy.v4.fixed_multiprototype import GPNTrainer

                temp_trainer = GPNTrainer(
                    model=model,
                    device=device,
                    use_multi=_cfg_require('data_feature_show.py.test_gradcam_analysis.use_multi__2')
                )

                # 计算高斯原型
                prototypes, precision_matrices = \
                    temp_trainer.compute_gaussian_prototypes(
                        support_v,
                        support_s,
                        remapped_support_labels,
                        n_ways=n_way
                    )

                # 计算距离
                distances = temp_trainer.loss_fn.distance_metric(
                    query_v,
                    prototypes,
                    precision_matrices
                )

                # 预测类别
                predictions = torch.argmin(distances, dim=1)

                print(f"使用单原型模式")

            # 计算准确率
            accuracy = (predictions == remapped_query_labels).float().mean().item()
            print(f"任务准确率: {accuracy*100:.2f}%")


        # 随机选择num_samples个查询样本进行分析
        random_indices = random.sample(range(len(query_signals)), num_samples)

        for sample_idx in random_indices:
            query_sample = query_signals[sample_idx:sample_idx+1]

            true_label = query_labels[sample_idx].item()
            pred_label = predictions[sample_idx].item()

            print(f"\n--- 样本 {sample_idx + 1}/{num_samples} ---")
            print(f"真实类别: {true_label}, 预测类别: {pred_label}")

            # 生成Grad-CAM (这里不使用no_grad)
            gradcam_map = gradcam_analyzer.generate_gradcam(
                query_sample,
                class_idx=pred_label
            )

            # 获取原始频谱图
            spectrogram = query_sample[0, 0].cpu().numpy()

            # 计算能量mask
            energy_mask = feature_analyzer.compute_energy_mask(spectrogram)

            # 分析<configured>: 边缘 vs 中心激活
            edge_center_ratio, edge_mask, center_mask = \
                feature_analyzer.edge_vs_center_activation(gradcam_map, energy_mask)

            # 分析<configured>: 激活中心对齐
            center_alignment = feature_analyzer.analyze_activation_centers(
                gradcam_map, energy_mask
            )

            # 分析<configured>: 频率注意力得分
            freq_score = feature_analyzer.compute_frequency_attention_score(
                gradcam_map
            )

            # 记录统计
            all_metrics['edge_center_ratios'].append(edge_center_ratio)
            all_metrics['center_alignments'].append(center_alignment)
            all_metrics['freq_scores'].append(freq_score)

            # 打印分析结果
            print(f"边缘/中心激活比: {edge_center_ratio:.4f}", end=" ")
            if edge_center_ratio > 2.0:
                print("-> 关注边缘纹理 ⚠️")
            elif edge_center_ratio < 0.5:
                print("-> 关注能量中心 ✓")
            else:
                print("-> 混合模式")

            print(f"激活-质心距离: {center_alignment:.4f}", end=" ")
            if center_alignment < 5.0:
                print("-> 对齐能量块中心 ✓")
            else:
                print("-> 偏离能量块中心 ⚠️")

            print(f"频率注意力得分: {freq_score:.4f}")

            # 可视化
            visualize_gradcam_analysis(
                spectrogram=spectrogram,
                gradcam_map=gradcam_map,
                energy_mask=energy_mask,
                edge_mask=edge_mask,
                center_mask=center_mask,
                true_label=true_label,
                pred_label=pred_label,
                metrics={
                    'edge_center_ratio': edge_center_ratio,
                    'center_alignment': center_alignment,
                    'freq_score': freq_score
                },
                save_path=save_dir / f'sample_{sample_idx+1}_analysis.png'
            )

    # 打印总体统计
    print("\n" + "="*80)
    print("总体特征分析结果")
    print("="*80)

    avg_edge_center = np.mean(all_metrics['edge_center_ratios'])
    avg_alignment = np.mean(all_metrics['center_alignments'])
    avg_freq_score = np.mean(all_metrics['freq_scores'])

    print(f"\n平均边缘/中心激活比: {avg_edge_center:.4f}")
    print(f"平均激活-质心距离: {avg_alignment:.4f}")
    print(f"平均频率注意力得分: {avg_freq_score:.4f}")

    print("\n【结论】")
    if avg_edge_center > 2.0 and avg_alignment > 5.0:
        print("⚠️  模型主要依赖 **纹理特征** (能量块边缘细节)")
        print("   - 建议: 使用边缘模糊测试验证")
    elif avg_edge_center < 0.5 and avg_alignment < 5.0:
        print("✓  模型主要依赖 **频谱特征** (能量块位置关系)")
        print("   - 建议: 使用频率置换测试验证")
    else:
        print("ℹ️  模型使用 **混合特征** (纹理 + 频谱)")
        print("   - 建议: 进行对抗性扰动测试以进一步确认")

    print("="*80)


def visualize_gradcam_analysis(
    spectrogram,
    gradcam_map,
    energy_mask,
    edge_mask,
    center_mask,
    true_label,
    pred_label,
    metrics,
    save_path
):
    """
    可视化Grad-CAM分析结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # <configured>. 原始频谱图
    im1 = axes[0, 0].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title(f'Original Spectrogram\nTrue: {true_label}, Pred: {pred_label}')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Frequency')
    plt.colorbar(im1, ax=axes[0, 0])

    # <configured>. Grad-CAM叠加
    axes[0, 1].imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=_cfg_require('data_feature_show.py.visualize_gradcam_analysis.alpha'))
    im2 = axes[0, 1].imshow(gradcam_map, aspect='auto', origin='lower', cmap='jet', alpha=_cfg_require('data_feature_show.py.visualize_gradcam_analysis.alpha__2'))
    axes[0, 1].set_title('Grad-CAM Overlay')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Frequency')
    plt.colorbar(im2, ax=axes[0, 1])

    # <configured>. 边缘 vs 中心分析
    axes[0, 2].imshow(spectrogram, aspect='auto', origin='lower', cmap='gray', alpha=_cfg_require('data_feature_show.py.visualize_gradcam_analysis.alpha__3'))
    axes[0, 2].contour(edge_mask, colors='red', linewidths=2, levels=[0.5])
    axes[0, 2].contour(center_mask, colors='blue', linewidths=2, levels=[0.5])
    axes[0, 2].contour(gradcam_map > 0.5, colors='yellow', linewidths=1, levels=[0.5])
    axes[0, 2].set_title(f'Edge vs Center\nRatio: {metrics["edge_center_ratio"]:.2f}')
    axes[0, 2].legend(['Edge (Red)', 'Center (Blue)', 'High Activation (Yellow)'],
                     loc='upper right', fontsize=8)

    # <configured>. 频率维度投影
    freq_profile_activation = np.mean(gradcam_map, axis=1)
    freq_profile_energy = np.mean(spectrogram, axis=1)

    ax4_twin = axes[1, 0].twinx()
    axes[1, 0].plot(freq_profile_activation, 'r-', linewidth=2, label='Activation')
    ax4_twin.plot(freq_profile_energy, 'b--', alpha=_cfg_require('data_feature_show.py.visualize_gradcam_analysis.alpha__4'), label='Energy')
    axes[1, 0].set_xlabel('Frequency Bin')
    axes[1, 0].set_ylabel('Activation', color='r')
    ax4_twin.set_ylabel('Energy', color='b')
    axes[1, 0].set_title(f'Frequency Profile\nFreq Score: {metrics["freq_score"]:.3f}')
    axes[1, 0].legend(loc='upper left')
    ax4_twin.legend(loc='upper right')

    # <configured>. 时间维度投影
    time_profile_activation = np.mean(gradcam_map, axis=0)
    time_profile_energy = np.mean(spectrogram, axis=0)

    ax5_twin = axes[1, 1].twinx()
    axes[1, 1].plot(time_profile_activation, 'r-', linewidth=2, label='Activation')
    ax5_twin.plot(time_profile_energy, 'b--', alpha=_cfg_require('data_feature_show.py.visualize_gradcam_analysis.alpha__5'), label='Energy')
    axes[1, 1].set_xlabel('Time Bin')
    axes[1, 1].set_ylabel('Activation', color='r')
    ax5_twin.set_ylabel('Energy', color='b')
    axes[1, 1].set_title('Time Profile')
    axes[1, 1].legend(loc='upper left')
    ax5_twin.legend(loc='upper right')

    # <configured>. 能量-激活散点图
    energy_flat = spectrogram.flatten()
    activation_flat = gradcam_map.flatten()

    axes[1, 2].scatter(energy_flat, activation_flat, alpha=_cfg_require('data_feature_show.py.visualize_gradcam_analysis.alpha__6'), s=1, c='blue')
    axes[1, 2].set_xlabel('Energy')
    axes[1, 2].set_ylabel('Activation')
    axes[1, 2].set_title(f'Energy-Activation Correlation\nAlignment: {metrics["center_alignment"]:.2f}')

    # 计算相关系数
    correlation = np.corrcoef(energy_flat, activation_flat)[0, 1]
    axes[1, 2].text(0.05, 0.95, f'r = {correlation:.3f}',
                   transform=axes[1, 2].transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=_cfg_require('data_feature_show.py.visualize_gradcam_analysis.alpha__7')))

    plt.tight_layout()
    plt.savefig(save_path, dpi=_cfg_require('data_feature_show.py.visualize_gradcam_analysis.size_or_budget'), bbox_inches='tight')
    plt.close()

    print(f"可视化结果已保存至: {save_path}")

def mask_low_frequency(spectrogram, mask_ratio=None):
    """将底部低频区域置零 (适用于单个 [C, H, W] 信号)"""
    # 假设输入形状是 [C, H, W] 或 [<configured>, H, W]
    mask_ratio = _cfg_resolve('data_feature_show.py.mask_low_frequency.mask_ratio', mask_ratio)
    C, H, W = spectrogram.shape # 修正：正确解包 C, H, W
    mask_height = int(H * mask_ratio)

    masked = spectrogram.clone()

    return masked




def test_mask_values():
    """测试模型对不同遮挡值的反应"""

    masks = {
        'zero': 0.0,           # 完全置零
        'negative': -0.5,      # 负值（异常）
        'mean': spec.mean(),   # 平均值
        'noise': np.random.randn(*spec.shape) * 0.01  # 随机噪声
    }

    results = {}
    for name, mask_value in masks.items():
        masked_spec = spec.clone()
        masked_spec[:50, :] = mask_value  # 遮挡底部<configured>

        acc = evaluate(masked_spec)
        gradcam = generate_gradcam(masked_spec)
        mask_activation = gradcam[:50, :].mean()

        results[name] = {
            'accuracy': acc,
            'mask_activation': mask_activation
        }

    return results
