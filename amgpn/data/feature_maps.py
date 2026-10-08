from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn.functional as F
import numpy as np

class NoiseInsensitiveFeatureExtractor:
    """
    优化版本：使用卷积加速
    """

    def __init__(self, lambda_L=None, lambda_H=None, delta=None,
                 gamma=None, eta_percentile=None, gaussian_window_size=None):
        lambda_L = _cfg_resolve('feature_extracter.py.NoiseInsensitiveFeatureExtractor.__init__.lambda_L', lambda_L)
        lambda_H = _cfg_resolve('feature_extracter.py.NoiseInsensitiveFeatureExtractor.__init__.lambda_H', lambda_H)
        delta = _cfg_resolve('feature_extracter.py.NoiseInsensitiveFeatureExtractor.__init__.delta', delta)
        gamma = _cfg_resolve('feature_extracter.py.NoiseInsensitiveFeatureExtractor.__init__.gamma', gamma)
        eta_percentile = _cfg_resolve('feature_extracter.py.NoiseInsensitiveFeatureExtractor.__init__.eta_percentile', eta_percentile)
        gaussian_window_size = _cfg_resolve('feature_extracter.py.NoiseInsensitiveFeatureExtractor.__init__.gaussian_window_size', gaussian_window_size)
        self.lambda_L = lambda_L
        self.lambda_H = lambda_H
        self.delta = delta
        self.gamma = gamma
        self.eta_percentile = eta_percentile
        self.gaussian_window_size = gaussian_window_size

        # Sobel算子
        self.sobel_time = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32)

        self.sobel_freq = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], dtype=torch.float32)

    def compute_gradients(self, S):
        """计算梯度（已优化）"""
        device = S.device

        sobel_t = self.sobel_time.to(device).unsqueeze(0).unsqueeze(0)
        sobel_f = self.sobel_freq.to(device).unsqueeze(0).unsqueeze(0)

        G_T = F.conv2d(S, sobel_t, padding=_cfg_require('feature_extracter.py.NoiseInsensitiveFeatureExtractor.compute_gradients.padding'))
        G_F = F.conv2d(S, sobel_f, padding=_cfg_require('feature_extracter.py.NoiseInsensitiveFeatureExtractor.compute_gradients.padding__2'))
        G = torch.sqrt(G_T**2 + G_F**2 + 1e-8)

        return G_T, G_F, G

    def non_maximum_suppression_fast(self, G):
        '\n        ⚡ 快速非极大值抑制 - 使用MaxPool实现\n        \n        速度提升: ~<configured>0x\n        '
        # 使用MaxPool找到局部最大值
        kernel_size = 2 * self.delta + 1

        # 对每个位置，找邻域内的最大值
        G_max = F.max_pool2d(
            G,
            kernel_size=kernel_size,
            stride=_cfg_require('feature_extracter.py.NoiseInsensitiveFeatureExtractor.non_maximum_suppression_fast.stride'),
            padding=self.delta
        )

        # 保留局部最大值点（且大于下界阈值）
        is_local_max = (G == G_max) & (G > self.lambda_L)

        G_hat = torch.where(is_local_max, G, torch.zeros_like(G))

        return G_hat

    def extract_edges_fast(self, S):
        """
        ⚡ 快速边缘提取
        """
        # 计算梯度
        _, _, G = self.compute_gradients(S)

        # 快速非极大值抑制
        G_hat = self.non_maximum_suppression_fast(G)

        # 向量化的双阈值分类
        S_edge = torch.zeros_like(G_hat)
        S_edge[G_hat >= self.lambda_H] = 1.0
        S_edge[(G_hat >= self.lambda_L) & (G_hat < self.lambda_H)] = 0.5

        return S_edge

    def create_gaussian_window(self, size):
        """创建高斯窗口（缓存）"""
        if not hasattr(self, '_gaussian_cache'):
            self._gaussian_cache = {}

        if size not in self._gaussian_cache:
            sigma = size / 6.0
            coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            window = g.unsqueeze(0) * g.unsqueeze(1)
            self._gaussian_cache[size] = window / window.sum()

        return self._gaussian_cache[size]

    def compute_corner_response(self, G_T, G_F):
        """计算角点响应（已优化）"""
        device = G_T.device

        W = self.create_gaussian_window(
            self.gaussian_window_size
        ).to(device).unsqueeze(0).unsqueeze(0)

        padding = self.gaussian_window_size // 2

        a = F.conv2d(G_T**2, W, padding=padding)
        b = F.conv2d(G_T * G_F, W, padding=padding)
        c = F.conv2d(G_F**2, W, padding=padding)

        R = (a * c - b**2) - self.gamma * (a + c)**2

        return R

    def extract_corners_fast(self, S):
        """
        ⚡ 快速角点提取
        """
        G_T, G_F, _ = self.compute_gradients(S)
        R = self.compute_corner_response(G_T, G_F)

        # 向量化的阈值计算
        R_max = R.flatten(1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        eta = self.eta_percentile * R_max

        S_corner = (R >= eta).float()

        return S_corner

    def forward(self, S):
        '\n        ⚡ 快速前向传播\n        \n        Args:\n            S: [B, <configured>, H, W] 原始时频图\n            \n        Returns:\n            S_in: [B, <configured>, H, W] 增强后的输入\n        '
        # 快速边缘提取
        S_edge = self.extract_edges_fast(S)

        # 快速角点提取
        S_corner = self.extract_corners_fast(S)

        # 拼接
        S_in = torch.cat([S, S_edge, S_corner], dim=1)

        return S_in
