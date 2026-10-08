from amgpn.config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from amgpn.data.episodes import MetaDataset
from amgpn.data.preprocessing import crop_and_rescale_symmetric

def plot_multiple_assignment_heatmaps(
    trainer,
    test_dataset,
    task_indices=None,            # 支持传入多个 task 索引
    target_class_local_ids=None, # 支持传入多个类索引，None 表示该 task 的所有类
    n_way=None,
    k_shot=None,
    q_query=None,
    num_tasks=None,
    seed=None,
    temp=0.5,
    save_dir=None,
    figsize_per_proto=1.5,
    figsize_per_sample=0.25
):
    '\n    批量绘制多个任务、多个类别的样本与自适应原型分配热力图\n    \n    Args:\n        task_indices: 要可视化的 task 索引列表，如 [<configured>, <configured>, <configured>]\n        target_class_local_ids: 要可视化的局部类索引列表，如 [<configured>, <configured>, <configured>]。若为 None，则可视化该 task 所有类\n        temp: Softmax 温度参数\n        save_dir: 热力图保存目录\n    '
    task_indices = _cfg_resolve('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.task_indices', task_indices)
    n_way = _cfg_resolve('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.n_way', n_way)
    k_shot = _cfg_resolve('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.k_shot', k_shot)
    q_query = _cfg_resolve('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.q_query', q_query)
    num_tasks = _cfg_resolve('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.num_tasks', num_tasks)
    seed = _cfg_resolve('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.seed', seed)
    save_dir = _cfg_resolve('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.save_dir', save_dir)
    device = trainer.device
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    meta_test = MetaDataset(
        test_dataset, n_way, k_shot, q_query=q_query, num_tasks=num_tasks, seed=seed
    )

    # 遍历每个任务
    for task_idx in task_indices:
        if task_idx >= len(meta_test):
            print(f"Warning: task_index {task_idx} out of range. Skipping.")
            continue

        episode = meta_test[task_idx]

        if len(episode) == 5:
            support_signals, support_labels, query_signals, query_labels, selected_classes = episode
        else:
            support_signals, support_labels, query_signals, query_labels = episode
            selected_classes = list(range(n_way))

        support_signals = support_signals.to(device).float().unsqueeze(1)
        query_signals = query_signals.to(device).float().unsqueeze(1)
        support_labels = support_labels.to(device).flatten()
        query_labels = query_labels.to(device).flatten()

        # 数据增强
        if trainer.crop_ratio_h != 0 or trainer.crop_ratio_l != 0:
            support_signals = torch.stack([
                crop_and_rescale_symmetric(s, trainer.crop_ratio_h, trainer.crop_ratio_l, trainer.resize)
                for s in support_signals.unbind(0)
            ])
            query_signals = torch.stack([
                crop_and_rescale_symmetric(s, trainer.crop_ratio_h, trainer.crop_ratio_l, trainer.resize)
                for s in query_signals.unbind(0)
            ])

        trainer.model.eval()
        with torch.no_grad():
            support_v, support_s = trainer.model(support_signals)
            query_v, query_s = trainer.model(query_signals)

        # 标签映射
        unique_labels = torch.unique(support_labels)
        label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
        remapped_support_labels = torch.tensor([label_mapping[l.item()] for l in support_labels], device=device)
        remapped_query_labels = torch.tensor([label_mapping[l.item()] for l in query_labels], device=device)

        # 确定要处理的类别列表
        classes_to_process = target_class_local_ids if target_class_local_ids is not None else list(range(n_way))

        # <configured>. 遍历每个类别
        for class_id in classes_to_process:
            if class_id >= n_way:
                print(f"Warning: class_id {class_id} >= n_way {n_way}. Skipping.")
                continue

            # 提取目标类别的特征
            class_s_mask = (remapped_support_labels == class_id)
            class_q_mask = (remapped_query_labels == class_id)

            class_support_v = support_v[class_s_mask]
            class_support_s = support_s[class_s_mask]
            class_query_v = query_v[class_q_mask]

            # 合并 support 和 query 样本
            all_samples_v = torch.cat([class_support_v, class_query_v], dim=0)
            n_support = class_support_v.shape[0]
            n_samples = all_samples_v.shape[0]

            if n_samples == 0:
                print(f"Task {task_idx} Class {class_id}: No samples found. Skipping.")
                continue

            init_protos_v = class_support_v
            init_protos_s = class_support_s
            n_init = init_protos_v.shape[0]

            # 计算两两马氏距离
            dist_matrix = torch.zeros((n_init, n_init), device=device)
            for i in range(n_init):
                # 利用对角精度矩阵的特性加速
                for j in range(i+1, n_init):
                    avg_prec_diag = (init_protos_s[i] + init_protos_s[j]) / 2
                    diff = init_protos_v[i] - init_protos_v[j]
                    dist_sq = (diff * avg_prec_diag * diff).sum()
                    dist = torch.sqrt(torch.clamp(dist_sq, min=1e-10))
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist

            # 构建邻接图并找连通分量
            adj_matrix = (dist_matrix < trainer.loss_fn.merge_threshold)
            visited = torch.zeros(n_init, dtype=torch.bool, device=device)
            components = []

            for i in range(n_init):
                if not visited[i]:
                    component = []
                    queue = deque([i])
                    visited[i] = True
                    while queue:
                        node = queue.popleft()
                        component.append(node)
                        neighbors = torch.where(adj_matrix[node])[0]
                        for neighbor in neighbors:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                queue.append(neighbor)
                    components.append(component)

            # 合并原型
            merged_protos_v = []
            merged_protos_s = []
            for comp in components:
                idx = torch.tensor(comp, device=device)
                merged_protos_v.append(init_protos_v[idx].mean(dim=0))
                merged_protos_s.append(init_protos_s[idx].mean(dim=0))

            merged_protos_v = torch.stack(merged_protos_v) # [M_c, D]
            merged_protos_s = torch.stack(merged_protos_s) # [M_c, D]
            n_protos = merged_protos_v.shape[0]

            # <configured>. 计算样本到合并后原型的分配概率
            distances = torch.zeros((n_samples, n_protos), device=device)
            for j in range(n_protos):
                diff = all_samples_v - merged_protos_v[j].unsqueeze(0)
                prec_diag = merged_protos_s[j]
                dist_sq = (diff * prec_diag.unsqueeze(0) * diff).sum(dim=1)
                distances[:, j] = torch.sqrt(torch.clamp(dist_sq, min=1e-10))

            # 转换为概率分布
            logits = -distances / temp
            prob_matrix = F.softmax(logits, dim=1).cpu().numpy()

            # <configured>. 排序样本以凸显块状结构
            dominant_proto = np.argmax(prob_matrix, axis=1)
            sort_idx = np.argsort(dominant_proto)
            prob_matrix_sorted = prob_matrix[sort_idx]

            # <configured>. 绘制热力图
            fig_width = max(6, n_protos * figsize_per_proto + 2)
            fig_height = max(4, n_samples * figsize_per_sample + 2.5)
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            sns.heatmap(
                prob_matrix_sorted,
                annot=True if n_samples * n_protos < _cfg_require('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.size_or_budget__2') else False, # 样本过多时关闭数字显示
                fmt=".2f",
                cmap="YlGnBu",
                vmin=0,
                vmax=1,
                ax=ax,
                linewidths=.5,
                cbar_kws={'label': 'Assignment Probability'}
            )

            # 设置坐标轴
            ax.set_ylabel(f"Samples (S: 0~{n_support-1}, Q: {n_support}~{n_samples-1})", fontsize=12)
            ax.set_xlabel("Adaptive Prototypes", fontsize=12)
            ax.set_title(
                f'Task {task_idx} | Class {selected_classes[class_id]} | Multi-Prototype Assignment\n({n_init} initial -> {n_protos} merged prototypes)',
                fontsize=14
            )

            # 设置刻度标签
            ax.set_yticks(np.arange(n_samples) + 0.5)
            y_labels = [f"S_{i}" if i < n_support else f"Q_{i-n_support}" for i in sort_idx]
            ax.set_yticklabels(y_labels, rotation=0, fontsize=8)

            ax.set_xticks(np.arange(n_protos) + 0.5)
            ax.set_xticklabels([f"P{j+1}" for j in range(n_protos)], rotation=0)

            plt.tight_layout()

            # 保存图像
            save_path = save_dir / f'task{task_idx}_class{selected_classes[class_id]}_heatmap.pdf'
            plt.savefig(save_path, dpi=_cfg_require('evaluation_module_single_class_heatmap.py.plot_multiple_assignment_heatmaps.size_or_budget'), bbox_inches='tight')
            plt.show()
            plt.close(fig) # 及时关闭防内存溢出

    print(f"\nAll heatmaps saved to: {save_dir.resolve()}")
