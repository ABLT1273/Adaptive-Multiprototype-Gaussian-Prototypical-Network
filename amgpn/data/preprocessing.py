"""Frequency-axis cropping used by all prototype trainers."""
from amgpn.config import require as _cfg_require, resolve as _cfg_resolve
import torch
from torch.nn import functional as F

def crop_and_rescale_symmetric(spectrogram, crop_ratio_h=None, crop_ratio_l=None, resize=None):
    '\n    自上而下按比例移除频谱图的上侧高频和下侧低频区域，\n    然后将截取后的中间部分缩放回原始高度 H。\n    Args:\n        spectrogram (torch.Tensor): 输入频谱图，形状为 [C, H, W] 或 [<configured>, H, W]。\n        crop_ratio_h (float): 上侧高频区域要移除的占总高度 H 的比例\n                              (例如 <configured> 表示移除顶部 <configured>)。\n        crop_ratio_l (float): 下侧低频区域要移除的占总高度 H 的比例\n                              (例如 <configured> 表示移除底部 <configured>)。\n        resize (bool): 是否将裁剪后的频谱图缩放回原始高度。\n    Returns:\n        torch.Tensor: 处理后的频谱图，形状与输入相同 [C, H, W] (当resize=True时)。\n    '
    # 假设输入形状是 [C, H, W]
    crop_ratio_h = _cfg_resolve('data_feature_show.py.crop_and_rescale_symmetric.crop_ratio_h', crop_ratio_h)
    crop_ratio_l = _cfg_resolve('data_feature_show.py.crop_and_rescale_symmetric.crop_ratio_l', crop_ratio_l)
    resize = _cfg_resolve('data_feature_show.py.crop_and_rescale_symmetric.resize', resize)
    if spectrogram.ndim != 3:
        raise ValueError(f"输入 Tensor 维度必须是 3 (C, H, W)，但收到了 {spectrogram.ndim} 维。")
    C, H, W = spectrogram.shape

    # <configured>. 确定要移除的上下边界高度
    # 移除的低频部分高度 (从底部开始数)
    low_freq_remove_height = int(H * crop_ratio_l)
    # 移除的高频部分高度 (从顶部开始数)
    high_freq_remove_height = int(H * crop_ratio_h)

    # 确定裁剪的起始和结束索引
    # 裁剪起始点（低频侧）：从 low_freq_remove_height 开始
    crop_start_index = low_freq_remove_height
    # 裁剪结束点（高频侧）：到 H - high_freq_remove_height 结束
    crop_end_index = H - high_freq_remove_height

    # 检查是否裁剪过度 (确保至少保留一行)
    if crop_end_index <= crop_start_index:
        print(f"警告: crop_ratio_h={crop_ratio_h}, crop_ratio_l={crop_ratio_l} 过大（总裁剪比例为 {crop_ratio_h + crop_ratio_l}），将返回原始信号。")
        return spectrogram.clone()
    # <configured>. 截取中间有效频率部分
    # H 维度是索引 <configured>
    # 截取范围 [crop_start_index, crop_end_index)
    cropped_middle_freq = spectrogram[:, crop_start_index:crop_end_index, :]

    # <configured>. 缩放回原始高度 H
    if resize:
        # F.interpolate 需要批次维度 [N, C, H, W]，所以先添加一个批次维度
        # 'align_corners=False' 推荐用于非像素对齐
        rescaled_spectrogram = F.interpolate(
            cropped_middle_freq.unsqueeze(0),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0) # 移除批次维度
        return rescaled_spectrogram
    else:
        return cropped_middle_freq
