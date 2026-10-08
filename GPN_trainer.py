from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
from pathlib import Path
from tqdm import tqdm
from data_loader import *
from loss import GPNLoss
import gc
from GPN_ori import *

def count_parameters(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 分别统计SE参数
    se_params = 0
    for name, module in model.named_modules():
        print(name)
        if isinstance(module, SEModule):
            se_params += sum(p.numel() for p in module.parameters())

    print(f"Total parameters: {total:,} ({total/1e3:.1f}K)")
    print(f"Trainable parameters: {trainable:,}")
    print(f"SE module parameters: {se_params:,} ({se_params/1e3:.1f}K)")
    print(f"Non-SE parameters: {total-se_params:,} ({(total-se_params)/1e3:.1f}K)")

    return total

class GPNTrainer:
    """
    GPN模型训练器
    """
    def __init__(self, model,device,feature_dim=None, lr=None):
        feature_dim = _cfg_resolve('GPN_trainer.py.GPNTrainer.__init__.feature_dim', feature_dim)
        lr = _cfg_resolve('GPN_trainer.py.GPNTrainer.__init__.lr', lr)
        self.device = device
        self.feature_dim = feature_dim

        # 初始化模型
        self.model = model
        self.model.to(device)
        self.loss_fn = GPNLoss()
        self.lr=lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"GPN模型初始化完成，设备: {self.device}")

    def save_model(self, save_path=None):
        """
        仅保存模型权重（用于部署）
        """
        save_path = _cfg_resolve('GPN_trainer.py.GPNTrainer.save_model.save_path', save_path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_dim': self.feature_dim,
        }, save_path)
        print(f"Model weights saved to {save_path}")

    def load_model(self, model_path):
        """
        仅加载模型权重
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model weights loaded from {model_path}")

    def compute_gaussian_prototypes(self, support_v, support_s, support_labels, n_ways):
        '\n        计算高斯原型和精度矩阵（论文公式<configured>-<configured>）\n        \n        Args:\n            support_v: 支持集embedding特征 [n_ways * k_shot, feature_dim]\n            support_s: 支持集precision特征 [n_ways * k_shot, feature_dim]\n            support_labels: 支持集标签 [n_ways * k_shot]\n            n_ways: 类别数\n            k_shot: 每类样本数\n            \n        Returns:\n            prototypes: 各类原型 [n_ways, feature_dim]\n            precision_matrices: 各类精度矩阵 [n_ways, feature_dim, feature_dim]\n        '
        feature_dim = support_v.shape[1]

        prototypes = torch.zeros(n_ways, feature_dim, device=self.device)
        precision_matrices = torch.zeros(n_ways, feature_dim, feature_dim, device=self.device)

        sigma = support_s

        # 为每个类别计算原型
        unique_labels = torch.unique(support_labels)

        for i, class_label in enumerate(unique_labels):
            class_mask = (support_labels == class_label)

            class_v = support_v[class_mask]  # [k_shot, feature_dim]
            class_sigma = sigma[class_mask]  # [k_shot, feature_dim]

            weighted_sum = torch.sum(class_sigma * class_v, dim=0)  # [feature_dim]
            sigma_sum = torch.sum(class_sigma, dim=0)  # [feature_dim]
            prototypes[i] = weighted_sum / (sigma_sum + 1e-8)

            # 计算精度矩阵（对角矩阵，公式<configured>）
            precision_diag = torch.mean(class_sigma, dim=0)  # [feature_dim]
            precision_matrices[i] = torch.diag(precision_diag)

        return prototypes, precision_matrices


    def train_step_batch(self, meta_batch):
        """
        批处理训练步骤 - 累积梯度后统一更新
        """
        self.model.train()

        batch_size = len(meta_batch[0])  # meta_batch中任务的数量
        total_loss = 0
        total_acc = 0

        # 清零梯度
        self.optimizer.zero_grad()

        # 处理批内所有任务，累积梯度
        for i in range(batch_size):
            support_signals, support_labels = meta_batch[0][i], meta_batch[1][i]
            query_signals, query_labels = meta_batch[2][i], meta_batch[3][i]

            # 单个任务的前向传播和损失计算
            loss, acc = self._single_task_forward(
                support_signals, support_labels,
                query_signals, query_labels
            )

            # 累积损失（自动累积梯度）
            (loss / batch_size).backward()  # 除以batch_size来平均化梯度

            total_loss += loss.item()
            total_acc += acc

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_cfg_require('GPN_trainer.py.GPNTrainer.train_step_batch.max_norm'))

        # 统一更新参数
        self.optimizer.step()

        return total_loss / batch_size, total_acc / batch_size

    def _single_task_forward(self, support_signals, support_labels, query_signals, query_labels):
        """
        单个任务的前向传播（不更新参数）
        """
        # 移动到设备
        support_signals = support_signals.to(self.device).float().unsqueeze(1)
        support_labels = support_labels.to(self.device)
        query_signals = query_signals.to(self.device).float().unsqueeze(1)
        query_labels = query_labels.to(self.device)

        # 前向传播
        support_v, support_s = self.model(support_signals)
        query_v, query_s = self.model(query_signals)

        # 标签重映射
        unique_labels = torch.unique(support_labels)
        n_ways = len(unique_labels)
        label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

        remapped_support_labels = torch.tensor([label_mapping[label.item()] for label in support_labels], device=self.device)
        remapped_query_labels = torch.tensor([label_mapping[label.item()] for label in query_labels], device=self.device)

        # 计算高斯原型
        prototypes, precision_matrices = self.compute_gaussian_prototypes(
            support_v, support_s, remapped_support_labels, n_ways
        )

        # 计算损失
        loss, probabilities, distances = self.loss_fn(
            query_v, prototypes, precision_matrices, remapped_query_labels
        )

        # 计算准确率
        predictions = torch.argmax(probabilities, dim=1)
        accuracy = (predictions == remapped_query_labels).float().mean()

        return loss, accuracy.item()

    def train(self,train_loader,test_loader, num_tasks=None, n_ways=None, epochs=None,save_path=None):
        """
        改进的元学习训练循环
        """
        num_tasks = _cfg_resolve('GPN_trainer.py.GPNTrainer.train.num_tasks', num_tasks)
        n_ways = _cfg_resolve('GPN_trainer.py.GPNTrainer.train.n_ways', n_ways)
        epochs = _cfg_resolve('GPN_trainer.py.GPNTrainer.train.epochs', epochs)
        save_path = _cfg_resolve('GPN_trainer.py.GPNTrainer.train.save_path', save_path)
        self.model.train()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=_cfg_require('GPN_trainer.py.GPNTrainer.train.step_size'), gamma=_cfg_require('GPN_trainer.py.GPNTrainer.train.gamma'))

        print(f"开始元学习训练: {epochs} epochs, {num_tasks} tasks/epoch")

        best_acc=0
        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            processed_tasks = 0
            # 使用DataLoader的自然批处理
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, meta_batch in enumerate(pbar):

                    if processed_tasks >= num_tasks:
                        break


                    # 批处理训练步骤
                    loss, acc = self.train_step_batch(meta_batch)
                    total_loss += loss
                    total_acc += acc
                    processed_tasks += len(meta_batch[0])  # 实际处理的任务数

                    pbar.set_postfix(
                        loss=f'{loss:.4f}',
                        accuracy=f'{acc:.4f}',
                        tasks=f'{processed_tasks}/{num_tasks}'
                    )

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)

            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

            if (epoch+1)%10==0:
                avg_acc=self.evaluate(test_loader=test_loader,n_ways=n_ways)
                print(f"当前测试acc:{avg_acc}")
                if avg_acc>best_acc:
                    best_acc=avg_acc
                    self.save_model(save_path)
    def evaluate(self, test_loader, n_ways, show_progress=True):
        """
        评估模型性能

        Args:
            test_loader: 数据加载器
            n_ways: 类别数
            show_progress: 是否显示进度条
        """
        self.model.eval()
        total_acc = 0

        # 根据参数决定是否使用tqdm
        iterator = tqdm(test_loader, desc="Evaluating") if show_progress else test_loader

        with torch.no_grad():
            for meta_task in iterator:
                support_signals, support_labels, query_signals, query_labels = meta_task

                # 移动到设备
                support_signals = support_signals.to(self.device).float().permute(1, 0, 2, 3)
                support_labels = support_labels.to(self.device).flatten()
                query_signals = query_signals.to(self.device).float().permute(1, 0, 2, 3)
                query_labels = query_labels.to(self.device).flatten()

                # 前向传播
                support_v, support_s = self.model(support_signals)

                # 重新映射标签
                unique_labels = torch.unique(support_labels)
                label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

                remapped_support_labels = torch.tensor(
                    [label_mapping[label.item()] for label in support_labels],
                    device=self.device
                )
                remapped_query_labels = torch.tensor(
                    [label_mapping[label.item()] for label in query_labels],
                    device=self.device
                )

                # 计算高斯原型
                prototypes, precision_matrices = self.compute_gaussian_prototypes(
                    support_v, support_s, remapped_support_labels, n_ways
                )

                # 计算距离和准确率
                query_v, query_s = self.model(query_signals)
                distances = self.loss_fn.compute_mahalanobis_distance(
                    query_v, prototypes, precision_matrices
                )
                logits = -distances
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == remapped_query_labels).float().mean()
                total_acc += accuracy.item()

        avg_acc = total_acc / len(test_loader)
        return avg_acc
    def full_evaluation(self, test_dataset, n_trials=None, q_query=None):
        """完整的评估协议（符合论文）"""
        n_trials = _cfg_resolve('GPN_trainer.py.GPNTrainer.full_evaluation.n_trials', n_trials)
        q_query = _cfg_resolve('GPN_trainer.py.GPNTrainer.full_evaluation.q_query', q_query)
        results = {}

        for n_way in [3, 4, 5, 6, 7, 8]:
            for k_shot in [1, 5, 10]:
                print(f"\n{'='*50}")
                print(f"Evaluating {n_way}-way {k_shot}-shot")
                print(f"{'='*50}")

                accuracies = []

                # 使用tqdm显示trials进度，position参数避免嵌套冲突
                pbar = tqdm(range(n_trials),
                           desc=f"{n_way}w{k_shot}s",
                           position=0,
                           leave=True)

                for trial in pbar:
                    meta_test = MetaDataset(
                        test_dataset, n_way, k_shot, q_query=q_query,
                        batch_size=_cfg_require('GPN_trainer.py.GPNTrainer.full_evaluation.batch_size'), num_tasks=_cfg_require('GPN_trainer.py.GPNTrainer.full_evaluation.num_tasks'), epochs=_cfg_require('GPN_trainer.py.GPNTrainer.full_evaluation.epochs')
                    )
                    loader = DataLoader(meta_test, batch_size=_cfg_require('GPN_trainer.py.GPNTrainer.full_evaluation.batch_size__2'))

                    # 关闭内层进度条
                    acc = self.evaluate(loader, n_way, show_progress=False)
                    accuracies.append(acc)

                    # 实时更新进度条显示当前平均准确率
                    if len(accuracies) > 0:
                        pbar.set_postfix({
                            'mean_acc': f"{np.mean(accuracies):.4f}",
                            'current': f"{acc:.4f}"
                        })

                    # ===== 关键：每个trial后立即清理 =====
                    del meta_test, loader, acc

                    # 每<configured>个trial进行一次显存清理
                    if (trial + 1) % 10 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                results[f'{n_way}w{k_shot}s'] = (mean_acc, std_acc)
                print(f"\nResult: {mean_acc:.4f} ± {std_acc:.4f}")

                del accuracies, pbar
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 确保CUDA操作完成


        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return results


def main():
    # 参数设置
    FEATURE_DIM = _cfg_require('GPN_trainer.py.main.FEATURE_DIM')
    N_WAY = _cfg_require('GPN_trainer.py.main.N_WAY')
    K_SHOT = _cfg_require('GPN_trainer.py.main.K_SHOT')
    Q_QUERY = _cfg_require('GPN_trainer.py.main.Q_QUERY')
    NUM_TASKS = _cfg_require('GPN_trainer.py.main.NUM_TASKS')
    EPOCHS = _cfg_require('GPN_trainer.py.main.EPOCHS')
    LR = _cfg_require('GPN_trainer.py.main.LR')
    BATCH_SIZE = _cfg_require('GPN_trainer.py.main.BATCH_SIZE')
    SEED = _cfg_require('GPN_trainer.py.main.SEED')
    SAVE_MODEL_PATH=_cfg_require('GPN_trainer.py.main.SAVE_MODEL_PATH')

    # 数据路径
    dataset_path = _cfg_require('GPN_trainer.py.main.dataset_path')

    # 创建原始数据集
    train_dataset=UAVDataset(data_dir_path=dataset_path,seed=SEED,is_train=True,train_ratio=_cfg_require('GPN_trainer.py.main.train_ratio'),max_sample_count=_cfg_require('GPN_trainer.py.main.max_sample_count'))
    test_dataset=UAVDataset(data_dir_path=dataset_path,seed=SEED,is_train=False,train_ratio=_cfg_require('GPN_trainer.py.main.train_ratio__2'),max_sample_count=_cfg_require('GPN_trainer.py.main.max_sample_count__2'))

    # 创建元任务划分集
    meta_train_dataset = MetaDataset(train_dataset, N_WAY, K_SHOT, Q_QUERY,batch_size=BATCH_SIZE,num_tasks=NUM_TASKS,epochs=EPOCHS)
    meta_test_dataset = MetaDataset(test_dataset, N_WAY, K_SHOT, Q_QUERY,batch_size=BATCH_SIZE,num_tasks=_cfg_require('GPN_trainer.py.main.num_tasks'),epochs=_cfg_require('GPN_trainer.py.main.epochs'))

    # DataLoader
    train_loader = DataLoader(meta_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=_cfg_require('GPN_trainer.py.main.num_workers'))
    test_loader = DataLoader(meta_test_dataset, batch_size=_cfg_require('GPN_trainer.py.main.batch_size'), shuffle=False)

    #创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_without_se = GPN(use_se=_cfg_require('GPN_trainer.py.main.use_se'))
    params_without_se = count_parameters(model_without_se)
    # 创建训练器
    meta_learner = GPNTrainer(model_without_se,device=device,feature_dim=FEATURE_DIM,lr=LR)
    print("开始训练...")
    meta_learner.train(train_loader=train_loader,test_loader=test_loader,num_tasks=NUM_TASKS,n_ways=N_WAY,epochs=EPOCHS,save_path=SAVE_MODEL_PATH)

    print("训练完成！")

    meta_learner.full_evaluation(test_dataset=test_dataset,n_trials=_cfg_require('GPN_trainer.py.main.n_trials'))

if __name__ == '__main__':
    main()
