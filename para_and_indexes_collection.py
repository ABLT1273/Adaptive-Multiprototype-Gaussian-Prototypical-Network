from private_config import require as _cfg_require, resolve as _cfg_resolve, format_value as _cfg_format
#仅<configured>0query!口径不统一，弃用
import time
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class FewShotDeploymentProfiler:
    '\n    Few-shot 模型部署指标收集器。\n\n    支持：\n        <configured>. AMGPN / 多原型聚类 GPN\n        <configured>. FEAT + GPN\n        <configured>. UNEM / UNEM-GMM\n        <configured>. 单原型 GPN / ProtoNet 风格 baseline\n\n    指标：\n        - Parameter Count\n        - Backbone FLOPs\n        - Prototype-stage FLOPs\n        - FEAT adaptor FLOPs\n        - UNEM head FLOPs\n        - Inference Speed / Deployment Latency\n        - Peak Memory / Inference Memory Increment\n    '

    def __init__(
        self,
        trainer,
        device=None,
        input_size=None,
        count_flops_with_thop=True,
    ):
        input_size = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.__init__.input_size', input_size)
        self.trainer = trainer
        self.device = device or getattr(trainer, "device", "cuda")
        self.device = torch.device(self.device)
        self.input_size = input_size
        self.count_flops_with_thop = count_flops_with_thop

        self.model = trainer.model
        self.loss_fn = trainer.loss_fn
        self.feat_adaptor = getattr(trainer, "feat_adaptor", None)

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _reset_cuda_memory(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def _memory_allocated_mb(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (_cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler._memory_allocated_mb.size_or_budget') ** 2)

    def _peak_memory_mb(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (_cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler._peak_memory_mb.size_or_budget') ** 2)

    def _set_eval(self):
        self.model.eval()
        self.loss_fn.eval()
        if self.feat_adaptor is not None:
            self.feat_adaptor.eval()

    def _module_param_count(self, module):
        if module is None:
            return 0, 0
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _unique_param_count(self):
        """
        避免重复统计同一个 parameter。
        """
        params = []
        modules = [self.model, self.loss_fn, self.feat_adaptor]

        seen = set()
        total = 0
        trainable = 0

        for module in modules:
            if module is None:
                continue
            for p in module.parameters():
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                total += p.numel()
                if p.requires_grad:
                    trainable += p.numel()

        return total, trainable

    # ------------------------------------------------------------------
    # <configured>. Parameter Count
    # ------------------------------------------------------------------

    def count_parameters(self):
        backbone_total, backbone_trainable = self._module_param_count(self.model)
        loss_total, loss_trainable = self._module_param_count(self.loss_fn)
        feat_total, feat_trainable = self._module_param_count(self.feat_adaptor)
        total, trainable = self._unique_param_count()

        return {
            "Params_Total": total,
            "Params_Trainable": trainable,

            "Backbone_Params": backbone_total,
            "Backbone_Trainable_Params": backbone_trainable,

            "Loss_Params": loss_total,
            "Loss_Trainable_Params": loss_trainable,

            "FEAT_Params": feat_total,
            "FEAT_Trainable_Params": feat_trainable,
        }

    # ------------------------------------------------------------------
    # <configured>. Backbone FLOPs
    # ------------------------------------------------------------------

    def compute_backbone_flops(self, n_way, k_shot, q_query):
        """
        统计 backbone FLOPs。

        返回：
            - 单张图像 backbone FLOPs
            - 一个 episode 的 backbone FLOPs:
                (n_way * k_shot + q_query) * single_image_flops
        """
        single_image_flops = 0
        single_image_params = 0

        if self.count_flops_with_thop:
            try:
                from thop import profile

                dummy = torch.randn(*self.input_size, device=self.device)
                self.model.eval()

                with torch.no_grad():
                    flops, params = profile(
                        self.model,
                        inputs=(dummy,),
                        verbose=False
                    )

                single_image_flops = int(flops)
                single_image_params = int(params)

            except Exception as e:
                print(f"[Warning] THOP failed for backbone FLOPs: {e}")
                single_image_flops = 0
                single_image_params = 0

        n_support = n_way * k_shot
        n_total_images = n_support + q_query
        episode_backbone_flops = single_image_flops * n_total_images

        return {
            "Backbone_FLOPs_Single_Image": single_image_flops,
            "Backbone_FLOPs_Episode": episode_backbone_flops,
            "Backbone_THOP_Params": single_image_params,
            "Episode_Input_Images": n_total_images,
        }

    # ------------------------------------------------------------------
    # <configured>. FEAT adaptor FLOPs
    # ------------------------------------------------------------------

    def compute_feat_flops(self, n_way, k_shot, q_query, feature_dim):
        """
        近似统计 FEAT adaptor FLOPs。

        你的 FEAT 路径：
            support self-attention:
                adapted_support = TransformerEncoder(support_v)
            query cross-attention:
                Q=query, K/V=adapted_support
                attn = softmax(QK^T)
                context = attn V

        这里只统计主要 matmul / linear FLOPs，LayerNorm、激活、softmax 只粗略估算。
        """
        if self.feat_adaptor is None:
            return {
                "FEAT_FLOPs": 0,
                "FEAT_Support_SelfAttn_FLOPs": 0,
                "FEAT_Query_CrossAttn_FLOPs": 0,
            }

        S = n_way * k_shot
        Q = q_query
        D = feature_dim

        # TransformerEncoderLayer:
        # Multi-head self-attention 近似：
        # Q/K/V/out projections: <configured> * S * D^<configured>
        # attention scores + weighted sum: <configured> * S^<configured> * D
        support_self_attn = 4 * S * D * D + 2 * S * S * D

        # FFN: D -> 2D -> D
        # 近似：<configured> * S * D^<configured>
        support_ffn = 4 * S * D * D

        support_total = support_self_attn + support_ffn

        # Query cross-attention:
        # q_proj: Q*D^<configured>
        # k_proj/v_proj: <configured>*S*D^<configured>
        # QK^T + AttnV: <configured>*Q*S*D
        cross_proj = (Q + 2 * S) * D * D
        cross_attn = 2 * Q * S * D
        cross_total = cross_proj + cross_attn

        # softmax 估算
        softmax_est = Q * S

        total = support_total + cross_total + softmax_est

        return {
            "FEAT_FLOPs": total,
            "FEAT_Support_SelfAttn_FLOPs": support_total,
            "FEAT_Query_CrossAttn_FLOPs": cross_total,
        }

    # ------------------------------------------------------------------
    # <configured>. Prototype-stage / AMGPN / UNEM FLOPs
    # ------------------------------------------------------------------

    def _detect_unem(self):
        return bool(
            getattr(self.trainer, "use_unem", False)
            or getattr(self.loss_fn, "use_unem", False)
            or hasattr(self.loss_fn, "unem")
        ) and getattr(self.loss_fn, "unem", None) is not None

    def _get_unem_components(self, k_shot):
        """
        兼容每类一个 Gaussian 和每类 GMM。
        """
        unem = getattr(self.loss_fn, "unem", None)
        if unem is None:
            return 0

        # UNEM-GMM 版本中通常叫 gmm_components
        if hasattr(unem, "gmm_components"):
            comp = getattr(unem, "gmm_components")
            return k_shot if comp is None else min(comp, k_shot)

        # 每类一个 Gaussian 版本
        return 1

    def compute_prototype_stage_flops(
        self,
        n_way,
        k_shot,
        q_query,
        feature_dim,
    ):
        """
        统计 Prototype-stage FLOPs。

        对不同方法分开：
            - 单原型 baseline
            - AMGPN / 多原型聚类
            - FEAT 的 prototype 分类头
            - UNEM / UNEM-GMM head
        """
        N = n_way
        K = k_shot
        Q = q_query
        D = feature_dim

        use_multi = getattr(self.trainer, "use_multi", False)
        use_unem = self._detect_unem()

        support_dist_flops = 0
        query_dist_flops = 0
        clustering_flops = 0
        unem_flops = 0
        prototype_average_flops = 0
        assign_flops = 0

        if use_unem:
            unem = self.loss_fn.unem
            layers = getattr(unem, "n_layers", getattr(unem, "em_steps", 1))
            M = self._get_unem_components(K)

            # Query 到 N*M 个 Gaussian component
            query_component_dist = Q * N * M * (2 * D)

            # Support 在每类内部到 M 个 component
            support_component_dist = N * K * M * (2 * D)

            # M-step：support + query 对 component 加权求和
            mstep = N * M * (K + Q) * (2 * D)

            # soft assignment / logsumexp / class balance 粗略估算
            assignment = Q * N * M + N * K * M

            unem_flops = layers * (
                query_component_dist
                + support_component_dist
                + mstep
                + assignment
            )

            total = unem_flops

            return {
                "Prototype_Stage_FLOPs": total,
                "Support_Dist_FLOPs": 0,
                "Query_Dist_FLOPs": 0,
                "Clustering_FLOPs": 0,
                "UNEM_FLOPs": unem_flops,
                "UNEM_Layers": layers,
                "UNEM_Components_Per_Class": M,
                "Prototype_Average_FLOPs": 0,
                "Prototype_Assign_FLOPs": assignment,
            }

        if use_multi:
            # AMGPN / 多原型聚类：
            # support 内每类 K 个原型两两距离，用于 connected/pairwise/hierarchical/dbscan 等合并
            support_dist_flops = N * (K * (K - 1) / 2) * (2 * D)

            # query 到 N*K 个原型的距离
            query_dist_flops = Q * N * K * (2 * D)

            # mask + min over prototypes
            assign_flops = Q * N * K

            # 聚类方法额外 Python/CPU 开销很难精确 FLOPs 化。
            # 这里只把距离矩阵作为主要计算量计入。
            clustering_flops = support_dist_flops

        else:
            # 单原型：每类 support 求均值 + query 到 N 个 prototype
            prototype_average_flops = N * K * D
            query_dist_flops = Q * N * (2 * D)
            assign_flops = Q * N

        total = (
            support_dist_flops
            + query_dist_flops
            + assign_flops
            + prototype_average_flops
        )

        return {
            "Prototype_Stage_FLOPs": total,
            "Support_Dist_FLOPs": support_dist_flops,
            "Query_Dist_FLOPs": query_dist_flops,
            "Clustering_FLOPs": clustering_flops,
            "UNEM_FLOPs": 0,
            "UNEM_Layers": _cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler.compute_prototype_stage_flops.UNEM_Layers'),
            "UNEM_Components_Per_Class": 0,
            "Prototype_Average_FLOPs": prototype_average_flops,
            "Prototype_Assign_FLOPs": assign_flops,
        }

    # ------------------------------------------------------------------
    # <configured>. Data preparation
    # ------------------------------------------------------------------

    def _prepare_signals(self, x):
        '\n        兼容你的 MetaDataset/DataLoader 输出：\n            DataLoader batch_size=<configured> 时常见形状：\n                support_signals: [<configured>, S, H, W]\n            evaluate() 中用：\n                support_signals.float().permute(<configured>, <configured>, <configured>, <configured>)\n            变成：\n                [S, <configured>, H, W]\n        '
        x = x.to(self.device).float()

        if x.dim() == 5:
            x = x.squeeze(0)
            if x.dim() == 4:
                return x

        if x.dim() == 4:
            # [<configured>, S, H, W] -> [S, <configured>, H, W]
            if x.shape[0] == 1 and x.shape[1] != 1:
                x = x.permute(1, 0, 2, 3)
            # 已经是 [S, <configured>, H, W]
            return x

        if x.dim() == 3:
            # [S, H, W] -> [S, <configured>, H, W]
            return x.unsqueeze(1)

        raise ValueError(f"Unsupported signal shape: {tuple(x.shape)}")

    def _prepare_labels(self, labels):
        return labels.to(self.device).flatten()

    def _remap_query_labels(self, support_labels, query_labels, n_way):
        '\n        兼容两种情况：\n            <configured>. support/query labels 已经是 <configured>..N-<configured>\n            <configured>. support/query labels 是原始类别 ID\n        '
        support_labels = support_labels.flatten()
        query_labels = query_labels.flatten()

        if (
            support_labels.numel() > 0
            and query_labels.numel() > 0
            and int(support_labels.min()) >= 0
            and int(query_labels.min()) >= 0
            and int(support_labels.max()) < n_way
            and int(query_labels.max()) < n_way
        ):
            return query_labels.long()

        unique_labels = torch.unique(support_labels)
        label_mapping = {label.item(): i for i, label in enumerate(unique_labels)}

        remapped = []
        for y in query_labels:
            y_item = y.item()
            if y_item not in label_mapping:
                # 如果 query 已经是 local label，就兜底直接用
                remapped.append(int(y_item))
            else:
                remapped.append(label_mapping[y_item])

        return torch.tensor(remapped, device=self.device, dtype=torch.long)

    # ------------------------------------------------------------------
    # <configured>. Real episode forward for timing
    # ------------------------------------------------------------------

    def _loss_forward_compatible(
        self,
        query_v,
        prototypes,
        precision_matrices,
        query_labels,
        n_way,
        k_shot,
    ):
        """
        兼容旧 loss_fn 和新版 UNEM loss_fn。
        新版可能支持 n_ways/k_shot 参数，旧版不支持。
        """
        try:
            return self.loss_fn(
                query_v=query_v,
                prototypes=prototypes,
                precision_matrices=precision_matrices,
                query_labels=query_labels,
                n_ways=n_way,
                k_shot=k_shot,
                epoch=getattr(self.trainer, "current_epoch", None),
            )
        except TypeError:
            return self.loss_fn(
                query_v,
                prototypes,
                precision_matrices,
                query_labels,
                epoch=getattr(self.trainer, "current_epoch", None),
            )

    def _single_episode_forward_timed(
        self,
        meta_task,
        n_way,
        k_shot,
    ):
        """
        真实 episode 前向路径计时。

        返回：
            timing dict, output dict
        """
        support_signals, support_labels, query_signals, query_labels, *extra = meta_task

        support_signals = self._prepare_signals(support_signals)
        query_signals = self._prepare_signals(query_signals)

        support_labels = self._prepare_labels(support_labels)
        query_labels = self._prepare_labels(query_labels)
        local_query_labels = self._remap_query_labels(
            support_labels,
            query_labels,
            n_way
        )

        timings = {}

        # --------------------
        # <configured>. Support backbone
        # --------------------
        self._sync()
        t0 = time.perf_counter()

        support_v, support_s = self.model(support_signals)

        self._sync()
        t1 = time.perf_counter()
        timings["Support_Backbone_ms"] = (t1 - t0) * _cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler._single_episode_forward_timed.size_or_budget')

        # --------------------
        # <configured>. Query backbone
        # --------------------
        self._sync()
        t0 = time.perf_counter()

        query_v, query_s = self.model(query_signals)

        self._sync()
        t1 = time.perf_counter()
        timings["Query_Backbone_ms"] = (t1 - t0) * _cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler._single_episode_forward_timed.size_or_budget__2')

        # --------------------
        # <configured>. FEAT adaptor
        # --------------------
        if self.feat_adaptor is not None:
            self._sync()
            t0 = time.perf_counter()

            support_v, query_v = self.feat_adaptor(support_v, query_v)

            self._sync()
            t1 = time.perf_counter()
            timings["FEAT_Adaptor_ms"] = (t1 - t0) * _cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler._single_episode_forward_timed.size_or_budget__4')
        else:
            timings["FEAT_Adaptor_ms"] = 0.0

        # --------------------
        # <configured>. Prototype / head
        # --------------------
        self._sync()
        t0 = time.perf_counter()

        if getattr(self.trainer, "use_multi", False):
            prototypes = support_v
            precision_matrices = self.trainer.compute_precision_matrices_from_support(
                support_s
            )

            output = self._loss_forward_compatible(
                query_v=query_v,
                prototypes=prototypes,
                precision_matrices=precision_matrices,
                query_labels=local_query_labels,
                n_way=n_way,
                k_shot=k_shot,
            )

            probabilities = output["probabilities"]
            predictions = torch.argmax(probabilities, dim=1)

        else:
            prototypes, precision_matrices = self.trainer.compute_gaussian_prototypes(
                support_v,
                support_s,
                support_labels,
                n_way
            )

            distances = self.loss_fn.distance_metric(
                query_v,
                prototypes,
                precision_matrices
            )
            logits = -distances
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            output = {
                "probabilities": probabilities,
                "distances": distances,
                "prototypes": prototypes,
                "precision_matrices": precision_matrices,
            }

        self._sync()
        t1 = time.perf_counter()
        timings["Prototype_Head_ms"] = (t1 - t0) * _cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler._single_episode_forward_timed.size_or_budget__3')

        # --------------------
        # Total
        # --------------------
        timings["Total_ms"] = (
            timings["Support_Backbone_ms"]
            + timings["Query_Backbone_ms"]
            + timings["FEAT_Adaptor_ms"]
            + timings["Prototype_Head_ms"]
        )

        output["predictions"] = predictions
        output["local_query_labels"] = local_query_labels

        return timings, output

    # ------------------------------------------------------------------
    # <configured>. Inference speed and memory
    # ------------------------------------------------------------------

    def benchmark_inference(
        self,
        test_dataset,
        n_way,
        k_shot,
        q_query=None,
        n_warmup=None,
        n_trials=None,
        num_workers=None,
    ):
        """
        收集真实 episode 推理延迟和显存。

        注意：
            - 不把 DataLoader 取数据时间计入。
            - 不把绘图、F1、混淆矩阵、t-SNE 统计计入。
            - 只计模型真实前向路径。
        """
        q_query = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.benchmark_inference.q_query', q_query)
        n_warmup = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.benchmark_inference.n_warmup', n_warmup)
        n_trials = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.benchmark_inference.n_trials', n_trials)
        num_workers = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.benchmark_inference.num_workers', num_workers)
        from data_loader_clean import MetaDataset

        self._set_eval()

        meta_test = MetaDataset(
            test_dataset,
            n_way,
            k_shot,
            q_query=q_query,
            num_tasks=n_warmup + n_trials,
            seed=_cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler.benchmark_inference.seed'),
        )
        loader = DataLoader(
            meta_test,
            batch_size=_cfg_require('para_and_indexes_collection.py.FewShotDeploymentProfiler.benchmark_inference.batch_size'),
            shuffle=False,
            num_workers=num_workers,
        )

        self._reset_cuda_memory()
        base_mem = self._memory_allocated_mb()

        stage_times = {
            "Support_Backbone_ms": [],
            "Query_Backbone_ms": [],
            "FEAT_Adaptor_ms": [],
            "Prototype_Head_ms": [],
            "Total_ms": [],
        }

        peak_mems = []
        accuracies = []

        with torch.no_grad():
            for task_id, meta_task in enumerate(loader):
                timings, output = self._single_episode_forward_timed(
                    meta_task=meta_task,
                    n_way=n_way,
                    k_shot=k_shot,
                )

                if task_id >= n_warmup:
                    for key in stage_times:
                        stage_times[key].append(timings[key])

                    pred = output["predictions"]
                    target = output["local_query_labels"]
                    acc = (pred == target).float().mean().item()
                    accuracies.append(acc)

                    peak_mems.append(self._peak_memory_mb())

        result = {}

        for key, values in stage_times.items():
            values = np.array(values, dtype=np.float64)
            result[f"{key}_Mean"] = float(values.mean())
            result[f"{key}_Std"] = float(values.std())
            result[f"{key}_P50"] = float(np.percentile(values, 50))
            result[f"{key}_P95"] = float(np.percentile(values, 95))

        result["Accuracy_Mean"] = float(np.mean(accuracies)) if accuracies else 0.0
        result["Peak_Memory_MB"] = float(np.max(peak_mems)) if peak_mems else 0.0
        result["Inference_Memory_Increment_MB"] = (
            float(np.max(peak_mems) - base_mem) if peak_mems else 0.0
        )

        return result

    # ------------------------------------------------------------------
    # <configured>. Full report
    # ------------------------------------------------------------------

    def infer_feature_dim(self):
        self._set_eval()
        dummy = torch.randn(*self.input_size, device=self.device)
        with torch.no_grad():
            v, _ = self.model(dummy)
        return int(v.shape[-1])

    def generate_profile_report(
        self,
        test_dataset,
        n_way=None,
        k_shot=None,
        q_query=None,
        n_warmup=None,
        n_trials=None,
        num_workers=None,
        print_report=True,
    ):
        n_way = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.generate_profile_report.n_way', n_way)
        k_shot = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.generate_profile_report.k_shot', k_shot)
        q_query = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.generate_profile_report.q_query', q_query)
        n_warmup = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.generate_profile_report.n_warmup', n_warmup)
        n_trials = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.generate_profile_report.n_trials', n_trials)
        num_workers = _cfg_resolve('para_and_indexes_collection.py.FewShotDeploymentProfiler.generate_profile_report.num_workers', num_workers)
        feature_dim = self.infer_feature_dim()

        report = {}
        report.update(self.count_parameters())
        report.update(self.compute_backbone_flops(n_way, k_shot, q_query))
        report.update(self.compute_feat_flops(n_way, k_shot, q_query, feature_dim))
        report.update(
            self.compute_prototype_stage_flops(
                n_way,
                k_shot,
                q_query,
                feature_dim
            )
        )
        report.update(
            self.benchmark_inference(
                test_dataset,
                n_way,
                k_shot,
                q_query=q_query,
                n_warmup=n_warmup,
                n_trials=n_trials,
                num_workers=num_workers,
            )
        )

        report["Feature_Dim"] = feature_dim
        report["N_Way"] = n_way
        report["K_Shot"] = k_shot
        report["Q_Query"] = q_query

        if print_report:
            self.print_report(report)

        return report

    def print_report(self, report):
        def fmt_num(x):
            if x >= 1e9:
                return f"{x / 1e9:.3f} G"
            if x >= 1e6:
                return f"{x / 1e6:.3f} M"
            if x >= 1e3:
                return f"{x / 1e3:.3f} K"
            return f"{x:.0f}"

        print("\n" + "=" * 72)
        print("Few-shot Model Deployment Profile")
        print("=" * 72)

        print(f"Task: {report['N_Way']}-way {report['K_Shot']}-shot, Q={report['Q_Query']}")
        print(f"Feature dim: {report['Feature_Dim']}")

        print("\n[Parameter Count]")
        print(f"  Backbone Params:        {fmt_num(report['Backbone_Params'])}")
        print(f"  FEAT Params:            {fmt_num(report['FEAT_Params'])}")
        print(f"  Loss/Head Params:       {fmt_num(report['Loss_Params'])}")
        print(f"  Total Params:           {fmt_num(report['Params_Total'])}")
        print(f"  Trainable Params:       {fmt_num(report['Params_Trainable'])}")

        print("\n[FLOPs]")
        print(f"  Backbone FLOPs/image:   {fmt_num(report['Backbone_FLOPs_Single_Image'])}")
        print(f"  Backbone FLOPs/episode: {fmt_num(report['Backbone_FLOPs_Episode'])}")
        print(f"  FEAT FLOPs/episode:     {fmt_num(report['FEAT_FLOPs'])}")
        print(f"  Prototype-stage FLOPs:  {fmt_num(report['Prototype_Stage_FLOPs'])}")

        print(f"    - Support Dist:       {fmt_num(report['Support_Dist_FLOPs'])}")
        print(f"    - Query Dist:         {fmt_num(report['Query_Dist_FLOPs'])}")
        print(f"    - Clustering:         {fmt_num(report['Clustering_FLOPs'])}")
        print(f"    - UNEM:               {fmt_num(report['UNEM_FLOPs'])}")

        if report.get("UNEM_Layers", 0) > 0:
            print(f"    - UNEM Layers:        {report['UNEM_Layers']}")
            print(f"    - GMM Comp/Class:     {report['UNEM_Components_Per_Class']}")

        print("\n[Inference Speed / Deployment Latency]")
        print(f"  Support Backbone mean:  {report['Support_Backbone_ms_Mean']:.3f} ms")
        print(f"  Query Backbone mean:    {report['Query_Backbone_ms_Mean']:.3f} ms")
        print(f"  FEAT Adaptor mean:      {report['FEAT_Adaptor_ms_Mean']:.3f} ms")
        print(f"  Prototype Head mean:    {report['Prototype_Head_ms_Mean']:.3f} ms")
        print(f"  Total mean:             {report['Total_ms_Mean']:.3f} ms")
        print(f"  Total P50:              {report['Total_ms_P50']:.3f} ms")
        print(f"  Total P95:              {report['Total_ms_P95']:.3f} ms")

        print("\n[Memory Footprint]")
        print(f"  Peak Memory:            {report['Peak_Memory_MB']:.2f} MB")
        print(f"  Inference Increment:    {report['Inference_Memory_Increment_MB']:.2f} MB")

        print("\n[Sanity]")
        print(f"  Mean Accuracy in profile trials: {report['Accuracy_Mean']:.4f}")
        print("=" * 72)
