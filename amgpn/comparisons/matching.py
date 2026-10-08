'\nMatching Networks for Few-Shot Learning\n\n完全按照现有框架实现\n\nReference: Vinyals et al. "Matching Networks for One Shot Learning",\n           NeurIPS <configured>\n'
from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import gc

from amgpn.data.episodes import *
from amgpn.data.preprocessing import crop_and_rescale_symmetric
from sklearn.metrics import f1_score, precision_score, recall_score
import time

def count_parameters(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total:,} ({total/1e3:.1f}K)")
    print(f"Trainable parameters: {trainable:,}")

    return total


class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, channels, reduction=None):
        reduction = _cfg_resolve('MN_baseline.py.SEModule.__init__.reduction', reduction)
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResNeXtBlock(nn.Module):
    """ResNeXt Bottleneck Block with SE"""
    def __init__(self, in_channels, out_channels, stride=None,
                 cardinality=None, reduction=None):
        stride = _cfg_resolve('MN_baseline.py.ResNeXtBlock.__init__.stride', stride)
        cardinality = _cfg_resolve('MN_baseline.py.ResNeXtBlock.__init__.cardinality', cardinality)
        reduction = _cfg_resolve('MN_baseline.py.ResNeXtBlock.__init__.reduction', reduction)
        super().__init__()

        mid_channels = out_channels

        # 1x1 扩展
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                              kernel_size=_cfg_require('MN_baseline.py.ResNeXtBlock.__init__.kernel_size'), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 分组卷积
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=_cfg_require('MN_baseline.py.ResNeXtBlock.__init__.kernel_size__2'), stride=stride,
                              padding=_cfg_require('MN_baseline.py.ResNeXtBlock.__init__.padding'), groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 压缩
        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                              kernel_size=_cfg_require('MN_baseline.py.ResNeXtBlock.__init__.kernel_size__3'), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.SiLU(inplace=True)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=_cfg_require('MN_baseline.py.ResNeXtBlock.__init__.kernel_size__4'), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # SE注意力
        self.attention = SEModule(out_channels, reduction=reduction)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # SE注意力
        if self.attention is not None:
            out = self.attention(out)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class MN_Backbone(nn.Module):
    """
    Matching Network的特征提取骨干网络（ResNeXt+SE版本）
    """
    def __init__(self, feature_dim=None):
        feature_dim = _cfg_resolve('MN_baseline.py.MN_Backbone.__init__.feature_dim', feature_dim)
        super().__init__()

        self.feature_dim = feature_dim

        # 初始卷积
        self.conv1 = nn.Conv2d(_cfg_require('MN_baseline.py.MN_Backbone.__init__.Conv2d_arg0'), _cfg_require('MN_baseline.py.MN_Backbone.__init__.Conv2d_arg1'), kernel_size=_cfg_require('MN_baseline.py.MN_Backbone.__init__.kernel_size'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride'),
                              padding=_cfg_require('MN_baseline.py.MN_Backbone.__init__.padding'), bias=False)
        self.bn1 = nn.BatchNorm2d(_cfg_require('MN_baseline.py.MN_Backbone.__init__.BatchNorm2d_arg0'))
        self.relu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=_cfg_require('MN_baseline.py.MN_Backbone.__init__.kernel_size__2'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__2'), padding=_cfg_require('MN_baseline.py.MN_Backbone.__init__.padding__2'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # Stage <configured>: <configured>→<configured>
        self.block1 = ResNeXtBlock(
            _cfg_require('MN_baseline.py.MN_Backbone.__init__.ResNeXtBlock_arg0'), _cfg_require('MN_baseline.py.MN_Backbone.__init__.ResNeXtBlock_arg1'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__3'), cardinality=_cfg_require('MN_baseline.py.MN_Backbone.__init__.cardinality'), reduction=_cfg_require('MN_baseline.py.MN_Backbone.__init__.reduction')
        )
        self.pool1 = nn.MaxPool2d(kernel_size=_cfg_require('MN_baseline.py.MN_Backbone.__init__.kernel_size__3'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__4'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # Stage <configured>: <configured>→<configured>
        self.block2 = ResNeXtBlock(
            _cfg_require('MN_baseline.py.MN_Backbone.__init__.ResNeXtBlock_arg0__2'), _cfg_require('MN_baseline.py.MN_Backbone.__init__.ResNeXtBlock_arg1__2'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__5'), cardinality=_cfg_require('MN_baseline.py.MN_Backbone.__init__.cardinality__2'), reduction=_cfg_require('MN_baseline.py.MN_Backbone.__init__.reduction__2')
        )
        self.pool2 = nn.MaxPool2d(kernel_size=_cfg_require('MN_baseline.py.MN_Backbone.__init__.kernel_size__4'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__6'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # Stage <configured>: <configured>→<configured>
        self.block3 = ResNeXtBlock(
            _cfg_require('MN_baseline.py.MN_Backbone.__init__.ResNeXtBlock_arg0__3'), _cfg_require('MN_baseline.py.MN_Backbone.__init__.ResNeXtBlock_arg1__3'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__7'), cardinality=_cfg_require('MN_baseline.py.MN_Backbone.__init__.cardinality__3'), reduction=_cfg_require('MN_baseline.py.MN_Backbone.__init__.reduction__3')
        )
        self.pool3 = nn.MaxPool2d(kernel_size=_cfg_require('MN_baseline.py.MN_Backbone.__init__.kernel_size__5'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__8'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # Stage <configured>: <configured>→<configured>
        self.block4 = ResNeXtBlock(
            _cfg_require('MN_baseline.py.MN_Backbone.__init__.ResNeXtBlock_arg0__4'), _cfg_require('MN_baseline.py.MN_Backbone.__init__.ResNeXtBlock_arg1__4'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__9'), cardinality=_cfg_require('MN_baseline.py.MN_Backbone.__init__.cardinality__4'), reduction=_cfg_require('MN_baseline.py.MN_Backbone.__init__.reduction__4')
        )
        self.pool4 = nn.MaxPool2d(kernel_size=_cfg_require('MN_baseline.py.MN_Backbone.__init__.kernel_size__6'), stride=_cfg_require('MN_baseline.py.MN_Backbone.__init__.stride__10'))
        # 输出: [B, <configured>, <configured>, <configured>]

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 最终维度调整（如果需要）
        if feature_dim != _cfg_require('MN_baseline.py.MN_Backbone.__init__.size_or_budget'):
            self.fc = nn.Linear(_cfg_require('MN_baseline.py.MN_Backbone.__init__.Linear_arg0'), feature_dim)
        else:
            self.fc = None

        self._initialize_weights()

    def forward(self, x):
        '\n        Args:\n            x: [B, <configured>, <configured>, <configured>]\n            \n        Returns:\n            features: [B, feature_dim]\n        '
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)
        x = self.pool4(x)

        x = self.avgpool(x)
        features = x.flatten(1)

        if self.fc is not None:
            features = self.fc(features)

        return features

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AttentionModule(nn.Module):
    """
    Matching Networks的注意力模块

    计算query和support set之间的注意力权重
    添加可学习的温度参数
    """
    def __init__(self, feature_dim=None):
        feature_dim = _cfg_resolve('MN_baseline.py.AttentionModule.__init__.feature_dim', feature_dim)
        super().__init__()
        self.feature_dim = feature_dim
        # 可学习的温度参数（让模型能调整相似度的锐度）
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, query_features, support_features):
        """
        计算注意力权重（使用cosine相似度 + 可学习温度）

        Args:
            query_features: [n_query, d]
            support_features: [n_support, d]

        Returns:
            attention_weights: [n_query, n_support] - softmax归一化的权重
        """
        # 归一化特征
        query_norm = F.normalize(query_features, p=2, dim=1)
        support_norm = F.normalize(support_features, p=2, dim=1)

        # 计算cosine相似度
        similarities = torch.mm(query_norm, support_norm.t())  # [n_query, n_support]

        # 乘以可学习的温度参数（放大相似度，使softmax更有区分度）
        scaled_similarities = similarities * self.temperature

        # Softmax得到注意力权重
        attention_weights = F.softmax(scaled_similarities, dim=1)

        return attention_weights, scaled_similarities  # 返回两者

class MatchingNetworkTrainer:
    """
    Matching Network训练器

    核心思想：使用注意力机制对support set加权，而不是计算原型
    """
    def __init__(self, model, device,
                 lr=None,
                 crop_ratio_h=None,
                 crop_ratio_l=None,
                 resize=None,
                 save_middle=None):
        """
        Args:
            model: MN_Backbone模型
            device: 训练设备
            lr: 学习率
            crop_ratio_h/l: 数据增强参数
            resize: 是否resize
            save_middle: 是否中途保存
        """
        lr = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.__init__.lr', lr)
        crop_ratio_h = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.__init__.crop_ratio_h', crop_ratio_h)
        crop_ratio_l = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.__init__.crop_ratio_l', crop_ratio_l)
        resize = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.__init__.resize', resize)
        save_middle = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.__init__.save_middle', save_middle)
        self.device = device
        self.crop_ratio_h = crop_ratio_h
        self.crop_ratio_l = crop_ratio_l
        self.resize = resize
        self.save_middle = save_middle

        # 初始化模型和注意力模块
        self.model = model
        self.model.to(device)

        self.attention = AttentionModule(feature_dim=model.feature_dim)
        self.attention.to(device)

        # 优化器（同时优化backbone和attention）
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.attention.parameters()),
            lr=self.lr
        )

        # 训练统计
        self.current_epoch = 0

        print("Matching Network模式: 注意力加权的最近邻")

    def save_model(self, save_path=None):
        """保存模型"""
        save_path = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.save_model.save_path', save_path)
        filtered_state_dict = {
            k: v for k, v in self.model.state_dict().items()
            if not ("total_ops" in k or "total_params" in k)
        }
        checkpoint = {
            'model_state_dict': filtered_state_dict,
            'attention_state_dict': self.attention.state_dict(),
            'current_epoch': self.current_epoch,
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.attention.load_state_dict(checkpoint['attention_state_dict'])
        print(f"Model loaded from {model_path}")

    def _single_task_forward(self, support_signals, support_labels,
                            query_signals, query_labels):
        """
        单个任务的前向传播

        使用注意力机制进行分类

        Returns:
            loss: tensor
            accuracy: float
            output: dict
        """
        # 移动到设备并添加通道维度
        support_signals = support_signals.to(self.device).float().unsqueeze(1)
        support_labels = support_labels.to(self.device)
        query_signals = query_signals.to(self.device).float().unsqueeze(1)
        query_labels = query_labels.to(self.device)

        # 数据增强
        if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
            processed_support_list = []
            for signal in support_signals.unbind(0):
                processed_support_list.append(
                    crop_and_rescale_symmetric(signal, self.crop_ratio_h,
                                              self.crop_ratio_l, self.resize)
                )
            support_signals = torch.stack(processed_support_list, dim=0)

            processed_query_list = []
            for signal in query_signals.unbind(0):
                processed_query_list.append(
                    crop_and_rescale_symmetric(signal, self.crop_ratio_h,
                                              self.crop_ratio_l, self.resize)
                )
            query_signals = torch.stack(processed_query_list, dim=0)

        # 提取特征
        support_features = self.model(support_signals)  # [n_support, d]
        # [关键] 保留 query_features 用于 t-SNE
        query_features = self.model(query_signals)      # [n_query, d]

        unique_labels = torch.unique(support_labels)
        n_ways = len(unique_labels)
        label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

        remapped_support_labels = torch.tensor(
            [label_mapping[label.item()] for label in support_labels],
            device=self.device
        )
        remapped_query_labels = torch.tensor(
            [label_mapping[label.item()] for label in query_labels],
            device=self.device
        )

        # 计算注意力权重（关键步骤）
        # 注意力机制的核心是计算 Query 特征与 Support 特征的相似度
        attention_weights, scaled_similarities = self.attention(query_features, support_features)
        # scaled_similarities: [n_query, n_support]

        # 使用注意力权重对标签进行加权求和
        # 将标签转换为one-hot
        support_one_hot = F.one_hot(remapped_support_labels, num_classes=n_ways).float()
        # [n_support, n_ways]

        # 计算预测概率（注意力加权求和，得到类别分数）
        # 类别分数 logits = (Query到Support的相似度) @ (Support样本的One-Hot标签)
        logits = torch.mm(scaled_similarities, support_one_hot)
        # logits: [n_query, n_ways]

        # 计算损失
        loss = F.cross_entropy(logits, remapped_query_labels)

        # 计算准确率
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == remapped_query_labels).float().mean()

        # ====== 新增：返回可视化和指标所需的数据 ======

        # <configured>. 预测结果（作为 numpy 数组）
        predictions_np = predictions.cpu().numpy()

        # <configured>. 真实标签（作为 numpy 数组）
        true_labels_np = remapped_query_labels.cpu().numpy()


        # 原有的 output 字典不再需要返回，因为其内部信息已被拆分

        return loss, accuracy.item(), predictions_np, true_labels_np, query_features

    def train_step_batch(self, meta_batch, batch_idx):
        """单个batch的训练步骤"""
        support_signals, support_labels, query_signals, query_labels = meta_batch
        batch_size = support_signals.size(0)

        total_loss = 0
        total_acc = 0

        self.optimizer.zero_grad()

        for i in range(batch_size):
            loss, acc, output = self._single_task_forward(
                support_signals[i],
                support_labels[i],
                query_signals[i],
                query_labels[i]
            )

            (loss / batch_size).backward()

            total_loss += loss.item()
            total_acc += acc

        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.attention.parameters()),
            max_norm=_cfg_require('MN_baseline.py.MatchingNetworkTrainer.train_step_batch.max_norm')
        )
        self.optimizer.step()

        return total_loss / batch_size, total_acc / batch_size

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        self.attention.train()
        total_loss = 0
        total_accuracy = 0
        processed_batches = 0

        with tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]") as pbar:
            for batch_idx, meta_batch in enumerate(pbar):
                loss, acc = self.train_step_batch(meta_batch, batch_idx)

                total_loss += loss
                total_accuracy += acc
                processed_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{acc:.4f}'
                })

                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        avg_loss = total_loss / processed_batches
        avg_accuracy = total_accuracy / processed_batches

        return avg_loss, avg_accuracy

    def evaluate(self, val_loader, n_way=None, show_progress=False, show_error_stats=False,
                 return_stats=False,
                 return_features=False,      # 新增：返回特征 (用于 t-SNE)
                 return_predictions=False,
                 all_result=False            # 新增：返回所有任务的详细指标
                 ):
        """
        元学习评估函数：按任务循环进行评估，并收集多项性能指标和可视化数据。
        集成了 evaluate_single_task 的逻辑。
        """


        n_way = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.evaluate.n_way', n_way)
        self.model.eval()
        self.attention.eval()

        total_loss = 0
        total_accuracy = 0
        total_f1_macro = 0
        total_precision_macro = 0
        total_recall_macro = 0
        total_time = 0
        processed_batches = 0

        # === 数据收集容器 ===
        all_targets = []        # 收集所有样本的真实标签
        all_predictions = []    # 收集所有样本的预测结果
        collected_features = [] # 收集所有 Query 样本的特征
        collected_labels = []   # 收集 Query 样本的原始标签 (用于 t-SNE)
        all_task_metrics = []   # 收集所有任务的单次指标

        pbar = tqdm(val_loader, desc="Evaluating") if show_progress else val_loader

        with torch.no_grad():
            for batch_idx, meta_batch in enumerate(pbar):
                support_signals, support_labels, query_signals, query_labels,selected_classes = meta_batch
                batch_size = support_signals.size(0)

                batch_loss = 0
                batch_accuracy = 0
                batch_time = 0

                for i in range(batch_size):
                    # ========== 嵌入 evaluate_single_task 的核心逻辑 ==========
                    task_start_time = time.time()

                    # <configured>. 数据准备
                    support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                    support_labels = support_labels.to(self.device).flatten()
                    query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                    query_labels = query_labels.to(self.device).flatten()

                    # <configured>. 数据增强：裁剪（如果启用）
                    if hasattr(self, 'crop_ratio_h') and hasattr(self, 'crop_ratio_l'):
                        if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
                            # Support set 裁剪
                            processed_support_list = []
                            for signal in support_signals.unbind(0):
                                processed_support_list.append(
                                    crop_and_rescale_symmetric(
                                        signal, self.crop_ratio_h, self.crop_ratio_l, self.resize
                                    )
                                )
                            support_sig = torch.stack(processed_support_list, dim=0)

                            # Query set 裁剪
                            processed_query_list = []
                            for signal in query_signals.unbind(0):
                                processed_query_list.append(
                                    crop_and_rescale_symmetric(
                                        signal, self.crop_ratio_h, self.crop_ratio_l, self.resize
                                    )
                                )
                            query_signals = torch.stack(processed_query_list, dim=0)

                    # <configured>. 特征提取
                    support_features = self.model(support_sig)
                    query_features = self.model(query_signals)

                    # <configured>. 注意力机制
                    attention_weights, scaled_similarities = self.attention(
                        query_features, support_features
                    )

                    # <configured>. 计算 logits
                    support_one_hot = F.one_hot(support_labels, num_classes=n_way).float()
                    logits = torch.mm(scaled_similarities, support_one_hot)

                    # <configured>. 预测和准确率
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == query_labels).float().mean()

                    # <configured>. 计算损失（如果需要）
                    # 假设使用交叉熵损失
                    loss = F.cross_entropy(logits, query_labels)

                    # <configured>. 计算耗时
                    task_time = time.time() - task_start_time

                    # ========== 指标收集 ==========

                    # 转换为 numpy 用于 sklearn 计算
                    predictions_np = predictions.cpu().numpy()
                    targets_np = query_labels.cpu().numpy()

                    # 基础指标累加
                    batch_loss += loss.item()
                    batch_accuracy += accuracy.item()
                    batch_time += task_time

                    if all_result:
                        task_f1 = f1_score(targets_np, predictions_np, average='macro', zero_division=0)
                        task_precision = precision_score(targets_np, predictions_np, average='macro', zero_division=0)
                        task_recall = recall_score(targets_np, predictions_np, average='macro', zero_division=0)

                        all_task_metrics.append({
                            'acc': accuracy.item(),
                            'f1_macro': task_f1,
                            'precision_macro': task_precision,
                            'recall_macro': task_recall,
                            'time_ms': task_time * _cfg_require('MN_baseline.py.MatchingNetworkTrainer.evaluate.size_or_budget'),  # 转换为毫秒
                            'task_id': batch_idx * batch_size + i,
                        })

                        total_f1_macro += task_f1
                        total_precision_macro += task_precision
                        total_recall_macro += task_recall

                    if return_predictions:
                        all_targets.extend(targets_np)
                        all_predictions.extend(predictions_np)

                    if return_features:
                        features_np = query_features.cpu().numpy()
                        collected_features.append(features_np)
                        collected_labels.append(targets_np)

                # 任务批次平均
                batch_loss /= batch_size
                batch_accuracy /= batch_size
                batch_time /= batch_size

                total_loss += batch_loss
                total_accuracy += batch_accuracy
                total_time += batch_time
                processed_batches += 1

                if show_progress:
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'acc': f'{batch_accuracy:.4f}'
                    })

                # 定期清理显存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        # ========== 计算最终平均指标 ==========
        total_tasks = processed_batches * batch_size
        avg_accuracy = total_accuracy / processed_batches

        # === 结果打包 ===
        result_pack = {
            'avg_acc': avg_accuracy,
        }

        # 计算全局 F1-score（从所有样本）
        if return_predictions and all_predictions:
            global_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
            global_precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
            global_recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)

            result_pack['global_f1_macro'] = global_f1
            result_pack['global_precision_macro'] = global_precision
            result_pack['global_recall_macro'] = global_recall

        # 计算任务平均指标
        if all_result:
            result_pack['avg_f1_macro'] = total_f1_macro / total_tasks
            result_pack['avg_precision_macro'] = total_precision_macro / total_tasks
            result_pack['avg_recall_macro'] = total_recall_macro / total_tasks
            result_pack['all_task_metrics'] = all_task_metrics

        # 特征数据打包
        if return_features and collected_features:
            result_pack['features'] = np.concatenate(collected_features, axis=0)
            result_pack['feature_labels'] = np.concatenate(collected_labels, axis=0)

        # 预测数据打包
        if return_predictions and all_predictions:
            result_pack['all_predictions'] = np.array(all_predictions)
            result_pack['all_targets'] = np.array(all_targets)

        # ========== 根据参数组合返回 ==========
        # 如果仅需要准确率，则保持原样（向后兼容）
        if not any([return_features, return_predictions, all_result, return_stats]):
            return avg_accuracy
        else:
            return result_pack

    def train(self, train_loader, val_loader, num_epochs,
             save_path=None):
        """完整训练流程"""
        save_path = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.train.save_path', save_path)
        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=_cfg_require('MN_baseline.py.MatchingNetworkTrainer.train.step_size'),
            gamma=_cfg_require('MN_baseline.py.MatchingNetworkTrainer.train.gamma')
        )

        print("\n" + "="*50)
        print("Starting Matching Network Training")
        print("="*50)
        print(f"总训练轮数: {num_epochs} epochs")
        print(f"每个epoch有 {len(train_loader)} 个批次")
        print("="*50)

        best_val_acc = 0
        early_stop_counter = 0
        early_stop_patience = _cfg_require('MN_baseline.py.MatchingNetworkTrainer.train.early_stop_patience')

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_loss, train_acc = self.train_epoch(train_loader)

            if (epoch+1)%5==0:
                # 验证
                val_loss, val_acc = self.evaluate(val_loader)

                # 打印结果
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")

                # 保存最优模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if self.save_middle:
                        self.save_model(save_path)
                        print(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f"No improvement ({early_stop_counter}/{early_stop_patience})")

                # Early stopping
                if early_stop_counter >= early_stop_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best Val Acc: {best_val_acc:.4f}")
                    break

                print("-" * 50)

            # 学习率调整
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            # 清理
            torch.cuda.empty_cache()
            gc.collect()

        print(f"\n{'='*50}")
        print(f"Training completed! Best Val Acc: {best_val_acc:.4f}")
        print(f"{'='*50}\n")

        return best_val_acc

    def evaluate_single_task(self, test_loader, n_ways=None, return_details=True):
        """
        评估单个任务（用于full_evaluation）

        Args:
            test_loader: 数据加载器
            n_ways: 类别数
            return_details: 是否返回详细指标（与evaluate函数保持一致）

        Returns:
            如果 return_details=False: accuracy (float)
            如果 return_details=True: dict 包含 {acc, f1_macro, precision_macro, recall_macro, time_ms}
        """
        n_ways = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.evaluate_single_task.n_ways', n_ways)
        import time
        from sklearn.metrics import f1_score, precision_score, recall_score

        self.model.eval()
        self.attention.eval()

        with torch.no_grad():

            # 开始计时
            task_start_time = time.time()
            meta_task = next(iter(test_loader))
            support_signals, support_labels, query_signals, query_labels, selected_classes = meta_task

            # 移动到设备
            support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
            support_labels = support_labels.to(self.device).flatten()
            query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
            query_labels = query_labels.to(self.device).flatten()

            # 数据增强：裁剪
            if self.crop_ratio_h != 0 or self.crop_ratio_l != 0:
                processed_support_list = []
                for signal in support_signals.unbind(0):
                    processed_support_list.append(
                        crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                    )
                support_signals = torch.stack(processed_support_list, dim=0)

                processed_query_list = []
                for signal in query_signals.unbind(0):
                    processed_query_list.append(
                        crop_and_rescale_symmetric(signal, self.crop_ratio_h, self.crop_ratio_l, self.resize)
                    )
                query_signals = torch.stack(processed_query_list, dim=0)

            # 提取特征
            support_features = self.model(support_signals)
            query_features = self.model(query_signals)

            # ✅ 修复：正确解包两个返回值
            attention_weights, scaled_similarities = self.attention(query_features, support_features)

            # ✅ 修复：使用scaled_similarities而不是attention_weights
            support_one_hot = F.one_hot(support_labels, num_classes=n_ways).float()
            logits = torch.mm(scaled_similarities, support_one_hot)

            # 计算预测
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == query_labels).float().mean()

            # 如果不需要详细指标，直接返回准确率
            if not return_details:
                return accuracy.item()

            # ========== 计算详细指标 ==========

            # 转换为numpy用于sklearn计算
            predictions_cpu = predictions.cpu().numpy()
            labels_cpu = query_labels.cpu().numpy()

            # 计算各项指标（与evaluate函数保持一致）
            task_acc = accuracy.item()
            task_f1 = f1_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)
            task_precision = precision_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)
            task_recall = recall_score(labels_cpu, predictions_cpu, average='macro', zero_division=0)

            # 计算耗时（毫秒）
            task_duration_ms = (time.time() - task_start_time) * _cfg_require('MN_baseline.py.MatchingNetworkTrainer.evaluate_single_task.size_or_budget')

            # 返回与evaluate函数all_task_metrics相同的结构
            return {
                'acc': task_acc,
                'f1_macro': task_f1,
                'precision_macro': task_precision,
                'recall_macro': task_recall,
                'time_ms': task_duration_ms,
                'task_id': 0,  # 单任务默认为<configured>
            }

    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """完整评估协议"""
        n_trials = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('MN_baseline.py.MatchingNetworkTrainer.full_evaluation.q_query', q_query)
        from amgpn.data.episodes import MetaDataset

        self.model.eval()
        self.attention.eval()
        results = {}

        print("\n" + "="*70)
        print("Starting Matching Network Full Evaluation")
        print("="*70)

        for n_way in [8, 7, 6, 5, 4, 3]:
            for k_shot in [10, 5, 1]:
                print(f"\n{'='*50}")
                print(f"Evaluating {n_way}-way {k_shot}-shot")
                print(f"{'='*50}")

                accuracies = []

                pbar = tqdm(range(n_trials),
                           desc=f"{n_way}w{k_shot}s",
                           position=0,
                           leave=True)

                for trial in pbar:
                    meta_test = MetaDataset(
                        test_dataset,
                        n_way=n_way,
                        k_shot=k_shot,
                        q_query=q_query,
                        num_tasks=_cfg_require('MN_baseline.py.MatchingNetworkTrainer.full_evaluation.num_tasks'),
                        seed=random.randint(0, _cfg_require('MN_baseline.py.MatchingNetworkTrainer.full_evaluation.size_or_budget'))
                    )

                    loader = DataLoader(meta_test, batch_size=_cfg_require('MN_baseline.py.MatchingNetworkTrainer.full_evaluation.batch_size'), shuffle=False)
                    acc = self.evaluate(loader, n_way, show_progress=False, show_error_stats=False)
                    accuracies.append(acc)

                    if len(accuracies) > 0:
                        pbar.set_postfix({
                            'mean_acc': f"{np.mean(accuracies):.4f}",
                            'current': f"{acc:.4f}"
                        })

                    del meta_test, loader

                    if (trial + 1) % 10 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                ci_95 = 1.96 * std_acc / np.sqrt(len(accuracies))

                results[f'{n_way}w{k_shot}s'] = (mean_acc, std_acc)

                print(f"\nResult: {mean_acc*100:.2f}% ± {ci_95*100:.2f}% (std: {std_acc*100:.2f}%)")

                del accuracies, pbar
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 打印汇总
        print("\n" + "="*70)
        print("Matching Network Full Evaluation Results Summary")
        print("="*70)
        print(f"{'Configuration':<15} {'Accuracy (%)':<15} {'Std (%)':<10} {'95% CI':<10}")
        print("-"*70)
        for config, (mean_acc, std_acc) in results.items():
            ci_95 = 1.96 * std_acc / np.sqrt(n_trials)
            print(f"{config:<15} {mean_acc*100:>6.2f} ± {ci_95*100:>5.2f}   {std_acc*100:>6.2f}%")
        print("="*70)

        return results


if __name__ == "__main__":
    """使用示例"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = MN_Backbone(feature_dim=_cfg_require('MN_baseline.py.module.feature_dim'))
    count_parameters(model)

    # 创建训练器
    trainer = MatchingNetworkTrainer(
        model=model,
        device=device,
        lr=_cfg_require('MN_baseline.py.module.lr'),
        crop_ratio_h=_cfg_require('MN_baseline.py.module.crop_ratio_h'),
        crop_ratio_l=_cfg_require('MN_baseline.py.module.crop_ratio_l'),
        resize=_cfg_require('MN_baseline.py.module.resize')
    )

    # 训练

    # 评估
