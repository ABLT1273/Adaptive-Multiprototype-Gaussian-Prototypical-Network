from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
from genericpath import isfile
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pywt


class MetaDataset(Dataset):
    '\n    元学习任务数据集\n    \n    关键设计原则：\n    <configured>. MetaDataset 只负责生成单个 episode（任务）\n    <configured>. 批处理完全由 DataLoader 的 batch_size 参数控制\n    <configured>. __len__ 返回总任务数，不涉及批处理\n    \n    Args:\n        uav_dataset: 原始数据集\n        n_way: 每个任务的类别数\n        k_shot: 每类的支持集样本数\n        q_query: 每类的查询集样本数\n        num_tasks: 总任务数（用于定义 epoch 长度）\n        seed: 随机种子，决定该批次的任务\n        \n    注意：移除了 batch_size 和 epochs 参数，这些应由训练循环控制\n    '
    def __init__(self, uav_dataset, n_way, k_shot, q_query, num_tasks,seed=None):
        seed = _cfg_resolve('data_loader_clean.py.MetaDataset.__init__.seed', seed)
        super().__init__()
        self.regular_dataset = uav_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_tasks = num_tasks
        self.seed=seed
        self.class_to_indices = self._group_by_class()
        self.classes = list(self.class_to_indices.keys())


    def _group_by_class(self):
        class_dict = {}
        for idx, (data, label) in enumerate(self.regular_dataset):
            if label not in class_dict:
                class_dict[label] = []
            class_dict[label].append(idx)
        return class_dict

    def __len__(self):
        """
        返回总任务数，不考虑批处理
        DataLoader 会根据其 batch_size 参数自动进行批处理
        """
        return self.num_tasks

    def __getitem__(self, index):
        """
        生成单个元任务（episode）

        返回值:
            support_samples: [n_way * k_shot, C, H, W]
            support_labels: [n_way * k_shot]
            query_samples: [n_way * q_query, C, H, W]
            query_labels: [n_way * q_query]
        """
        task_rng = random.Random(self.seed + index)
        # 随机选择 N 个类
        selected_classes = task_rng.sample(self.classes, self.n_way)


        support_data = []
        support_labels = []
        query_data = []
        query_labels = []

        for i, class_label in enumerate(selected_classes):
            class_indices = self.class_to_indices[class_label]

            if len(class_indices) < self.k_shot + self.q_query:
                raise ValueError(
                    f'类别 {class_label} 的样本数 {len(class_indices)} 不足，无法构建 {self.k_shot}-shot {self.q_query}-query 任务'
                )

            selected_indices = task_rng.sample(
                class_indices,
                self.k_shot + self.q_query
            )

            support_indices = selected_indices[:self.k_shot]
            query_indices = selected_indices[self.k_shot:]

            new_label = i  # 重映射标签为 <configured>~(n_way-<configured>)

            for idx in support_indices:
                sample_data, _ = self.regular_dataset[idx]
                support_data.append(sample_data)
                support_labels.append(new_label)

            for idx in query_indices:
                sample_data, _ = self.regular_dataset[idx]
                query_data.append(sample_data)
                query_labels.append(new_label)

        support_samples = torch.stack(support_data)
        support_labels = torch.tensor(support_labels)
        query_samples = torch.stack(query_data)
        query_labels = torch.tensor(query_labels)

        return support_samples, support_labels, query_samples, query_labels, selected_classes

class UAVDataset(Dataset):
    """
    无人机频谱图数据集类
    """
    def __init__(self, data_dir_path, seed, is_train=True, train_ratio=None, max_sample_count=None,
                ):

        train_ratio = _cfg_resolve('data_loader_clean.py.UAVDataset.__init__.train_ratio', train_ratio)
        max_sample_count = _cfg_resolve('data_loader_clean.py.UAVDataset.__init__.max_sample_count', max_sample_count)
        super().__init__()
        random.seed(seed)
        self.data = []  # 存储 (sample, label) 元组

        label_list = os.listdir(data_dir_path)
        random.shuffle(label_list)
        split_idx = round(len(label_list) * train_ratio)

        if is_train:
            selected_labels = label_list[:split_idx]
        else:
            selected_labels = label_list[split_idx:]

        print(f"选择的标签 ({'训练集' if is_train else '测试集'}): {selected_labels}")
        print('='*20)

        for label in selected_labels:
            label_dir = os.path.join(data_dir_path, label)
            if os.path.isdir(label_dir):
                sample_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
                sample_files = random.sample(sample_files, min(max_sample_count, len(sample_files)))

                for sample_name in sample_files:
                    sample_path = os.path.join(label_dir, sample_name)
                    sample = np.load(sample_path, allow_pickle=True)
                    # 裁剪和预处理
                    start_row = (sample.shape[0] - _cfg_require('data_loader_clean.py.UAVDataset.__init__.size_or_budget')) // 2
                    sample = sample[start_row:start_row+_cfg_require('data_loader_clean.py.UAVDataset.__init__.size_or_budget__2'), :]

                    # 转换为 PyTorch Tensor
                    sample = torch.from_numpy(sample).float() # 确保是 float 类型

                    # 将样本和标签成对存储
                    self.data.append((sample, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接返回存储的元组
        return self.data[idx]
