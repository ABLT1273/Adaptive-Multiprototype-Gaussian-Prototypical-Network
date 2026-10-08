"""Shared backbone for AMGPN, merging ablations, FEAT and UNEM.

Architecture settings retain their original per-variant configuration namespace.
"""
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from amgpn.configuration import backbone_scope, require_backbone as _cfg_require, resolve_backbone as _cfg_resolve

class SEModule(nn.Module):
    """原始SE模块 - 保持兼容性"""
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResNeXtBlock(nn.Module):
    """
    ResNeXt Bottleneck Block

    支持的注意力机制:标准SE模块

    """
    def __init__(self, in_channels, out_channels, stride=None,
                 cardinality=None, reduction=None,
                 ):
        stride = _cfg_resolve('ResNeXtBlock.__init__.stride', stride)
        cardinality = _cfg_resolve('ResNeXtBlock.__init__.cardinality', cardinality)
        reduction = _cfg_resolve('ResNeXtBlock.__init__.reduction', reduction)
        super().__init__()

        mid_channels = out_channels

        # 1x1 扩展卷积
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=_cfg_require('ResNeXtBlock.__init__.kernel_size'), bias=False,)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 分组卷积
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=_cfg_require('ResNeXtBlock.__init__.kernel_size__2'), stride=stride,
                              padding=_cfg_require('ResNeXtBlock.__init__.padding'), groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 压缩卷积
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                              kernel_size=_cfg_require('ResNeXtBlock.__init__.kernel_size__3'), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.SiLU(inplace=True)

        # Shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=_cfg_require('ResNeXtBlock.__init__.kernel_size__4'), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # 注意力机制SE
        self.attention = SEModule(out_channels, reduction=reduction)

    def forward(self, x):
        identity = self.shortcut(x)

        # Bottleneck路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 注意力模块(在Add之前应用)
        if self.attention is not None:
            out = self.attention(out)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

class GPN_Optimized(nn.Module):
    '\n    优化的GPN模型 - 完全兼容现有训练器\n    \n    架构特点：\n    - 全局：统一的注意力机制SE\n    - 输出：v (embedding), s (precision)\n    \n    参数说明：\n        reduction: SE模块的reduction ratio（由外部配置提供）\n    '
    def __init__(self,
                 reduction=None):
        reduction = _cfg_resolve('GPN_Optimized.__init__.reduction', reduction)
        super().__init__()

        self.conv1 = nn.Conv2d(_cfg_require('GPN_Optimized.__init__.Conv2d_arg0'), _cfg_require('GPN_Optimized.__init__.Conv2d_arg1'), kernel_size=_cfg_require('GPN_Optimized.__init__.kernel_size'), stride=_cfg_require('GPN_Optimized.__init__.stride'),
                                padding=_cfg_require('GPN_Optimized.__init__.padding'), bias=False)

        self.bn1 = nn.BatchNorm2d(_cfg_require('GPN_Optimized.__init__.BatchNorm2d_arg0'))
        self.relu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=_cfg_require('GPN_Optimized.__init__.kernel_size__2'), stride=_cfg_require('GPN_Optimized.__init__.stride__2'), padding=_cfg_require('GPN_Optimized.__init__.padding__2'))
        # 输出: [B, <configured>, <configured>, <configured>]


        self.block1 = ResNeXtBlock(
            _cfg_require('GPN_Optimized.__init__.ResNeXtBlock_arg0'), _cfg_require('GPN_Optimized.__init__.ResNeXtBlock_arg1'), stride=_cfg_require('GPN_Optimized.__init__.stride__3'), cardinality=_cfg_require('GPN_Optimized.__init__.cardinality'),
            reduction=_cfg_require('GPN_Optimized.__init__.reduction__2'),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_Optimized.__init__.kernel_size__3'), stride=_cfg_require('GPN_Optimized.__init__.stride__4'))
        # 输出: [B, <configured>, <configured>, <configured>]


        self.block2 = ResNeXtBlock(
            _cfg_require('GPN_Optimized.__init__.ResNeXtBlock_arg0__2'), _cfg_require('GPN_Optimized.__init__.ResNeXtBlock_arg1__2'), stride=_cfg_require('GPN_Optimized.__init__.stride__5'), cardinality=_cfg_require('GPN_Optimized.__init__.cardinality__2'),
            reduction=_cfg_require('GPN_Optimized.__init__.reduction__3'),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_Optimized.__init__.kernel_size__4'), stride=_cfg_require('GPN_Optimized.__init__.stride__6'))
        # 输出: [B, <configured>, <configured>, <configured>]


        self.block3 = ResNeXtBlock(
            _cfg_require('GPN_Optimized.__init__.ResNeXtBlock_arg0__3'), _cfg_require('GPN_Optimized.__init__.ResNeXtBlock_arg1__3'), stride=_cfg_require('GPN_Optimized.__init__.stride__7'), cardinality=_cfg_require('GPN_Optimized.__init__.cardinality__3'),
            reduction=_cfg_require('GPN_Optimized.__init__.reduction__4'),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_Optimized.__init__.kernel_size__5'), stride=_cfg_require('GPN_Optimized.__init__.stride__8'))
        # 输出: [B, <configured>, <configured>, <configured>]

        self.block4 = ResNeXtBlock(
            _cfg_require('GPN_Optimized.__init__.ResNeXtBlock_arg0__4'), _cfg_require('GPN_Optimized.__init__.ResNeXtBlock_arg1__4'), stride=_cfg_require('GPN_Optimized.__init__.stride__9'), cardinality=_cfg_require('GPN_Optimized.__init__.cardinality__4'),
            reduction=_cfg_require('GPN_Optimized.__init__.reduction__5'),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('GPN_Optimized.__init__.kernel_size__6'), stride=_cfg_require('GPN_Optimized.__init__.stride__10'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # === Global Pooling ===
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

    def forward(self, x, return_intermediate=False):
        '\n        前向传播 - 对齐原模型接口\n        \n        Args:\n            x: [B, <configured>, <configured>, <configured>]\n            return_intermediate: 是否返回中间特征\n            \n        Returns:\n            v: [B, <configured>] embedding特征\n            s: [B, <configured>] precision特征\n            features: (可选) 中间特征字典\n        '
        features = {} if return_intermediate else None

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, <configured>, <configured>, <configured>]

        # Stage <configured>-<configured>
        x = self.block1(x)
        x = self.pool1(x)  # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block1'] = x

        x = self.block2(x)
        x = self.pool2(x)  # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block2'] = x

        x = self.block3(x)
        x = self.pool3(x)  # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block3'] = x

        x = self.block4(x)
        x = self.pool4(x)  # [B, <configured>, <configured>, <configured>]
        if return_intermediate:
            features['block4'] = x

        # Channel Split: v和s
        v_features = x[:, :_cfg_require('GPN_Optimized.forward.size_or_budget'), :, :]  # [B, <configured>, <configured>, <configured>]
        s_features = x[:, _cfg_require('GPN_Optimized.forward.size_or_budget__2'):, :, :]  # [B, <configured>, <configured>, <configured>]

        if return_intermediate:
            features['v_features'] = v_features
            features['s_features'] = s_features

        # Global Average Pooling
        v = self.avgpool(v_features).flatten(1)  # [B, <configured>]
        s = self.avgpool(s_features).flatten(1)  # [B, <configured>]

        s = 1 + F.softplus(s)

        if return_intermediate:
            return v, s, features
        return v, s

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

SpectrogramBackbone = GPN_Optimized


@lru_cache(maxsize=None)
def configured_backbone(namespace):
    """Bind legacy configuration keys without copying the backbone implementation."""
    class ConfiguredBackbone(GPN_Optimized):
        def __init__(self, reduction=None):
            with backbone_scope(namespace):
                super().__init__(reduction=reduction)

        def forward(self, x, return_intermediate=False):
            with backbone_scope(namespace):
                return super().forward(x, return_intermediate=return_intermediate)
    ConfiguredBackbone.__name__ = "GPN_Optimized"
    ConfiguredBackbone.__qualname__ = "GPN_Optimized"
    return ConfiguredBackbone
