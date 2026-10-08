from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
from genericpath import isfile
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
class MetaDataset(Dataset):
    def __init__(self, uav_dataset, n_way,k_shot,q_query,batch_size,num_tasks,epochs):
        super().__init__()
        self.regular_dataset=uav_dataset
        self.n_way=n_way
        self.k_shot=k_shot
        self.q_query=q_query
        self.batch_size=batch_size
        self.num_tasks=num_tasks
        self.epochs=epochs
        self.class_to_indices=self._group_by_class()
        self.classes=list(self.class_to_indices.keys())

    def _group_by_class(self):
        class_dict={}
        for idx,(data,label) in enumerate(self.regular_dataset):
            if label not in class_dict:
                class_dict[label]=[]
            class_dict[label].append(idx)
        return class_dict
    def __len__(self):
        # 训练阶段返回 NUM_TASKS * EPOCHS，测试阶段可自定义
        return self.num_tasks//self.batch_size  # 这里可以根据需要调整
    def __getitem__(self,index):
        """
        生成一个小样本学习的元任务（episode）。

        返回值:
            (support_samples, support_labels, query_samples, query_labels):
            分别对应支持集和查询集的图片张量和标签张量。
        """

        #随机选择 N 个类
        selected_classes = random.sample(self.classes,self.n_way)

        support_data=[]
        support_labels=[]
        query_data=[]
        query_labels=[]
        for i,class_label in enumerate(selected_classes):
            class_indices = self.class_to_indices[class_label]

            if len(class_indices) < self.k_shot + self.q_query:
                # 可以在这里抛出错误或者选择重采样
                raise ValueError(f"类别 {class_label} 的样本数{len(class_indices)}不足，无法构建元任务。")
            selected_indices=random.sample(class_indices,self.k_shot+self.q_query)

            support_indices=selected_indices[:self.k_shot]
            query_indices=selected_indices[self.k_shot:]

            new_label=i

            for idx in support_indices:
                sample_data,_=self.regular_dataset[idx]
                support_data.append(sample_data)
                support_labels.append(new_label)

            for idx in query_indices:
                sample_data,_=self.regular_dataset[idx]
                query_data.append(sample_data)
                query_labels.append(new_label)

        support_samples=torch.stack(support_data)
        support_labels=torch.tensor(support_labels)

        query_samples=torch.stack(query_data)
        query_labels=torch.tensor(query_labels)

        return support_samples,support_labels,query_samples,query_labels

class UAVDataset(Dataset):
    def __init__(self, data_dir_path, seed, max_sample_count=None):
        max_sample_count = _cfg_resolve('old/data_loader_hard.py.UAVDataset.__init__.max_sample_count', max_sample_count)
        super().__init__()
        random.seed(seed)
        self.data = []  # 存储 (sample, label) 元组
        label_list = os.listdir(data_dir_path)
        random.shuffle(label_list)

        for label in label_list:
            label_dir = os.path.join(data_dir_path, label)
            if os.path.isdir(label_dir):
                sample_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
                sample_files = random.sample(sample_files, min(max_sample_count, len(sample_files)))
                for sample_name in sample_files:
                    sample_path = os.path.join(label_dir, sample_name)
                    sample = np.load(sample_path, allow_pickle=True)
                    start_row = (sample.shape[0] - _cfg_require('old/data_loader_hard.py.UAVDataset.__init__.size_or_budget')) // 2
                    sample = sample[start_row:start_row+_cfg_require('old/data_loader_hard.py.UAVDataset.__init__.size_or_budget__2'), :]
                    sample=torch.from_numpy(sample)
                    # 将样本和标签成对存储
                    self.data.append((sample, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接返回存储的元组
        return self.data[idx]
