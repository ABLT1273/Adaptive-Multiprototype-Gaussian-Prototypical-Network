"""Optional CPU integration smoke tests; all settings and inputs are synthetic."""
import contextlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

IDS = ('pn', 'gpn', 'amgpn', 'feat', 'unem', 'merge-pairwise', 'merge-hierarchical', 'merge-dbscan')


def synthetic_settings():
    """Tiny test architecture, never a reproduction of private experiment settings."""
    suffix = {
        'ResNeXtBlock.__init__.stride': 1, 'ResNeXtBlock.__init__.cardinality': 1,
        'ResNeXtBlock.__init__.reduction': 2, 'ResNeXtBlock.__init__.kernel_size': 1,
        'ResNeXtBlock.__init__.kernel_size__2': 3, 'ResNeXtBlock.__init__.padding': 1,
        'ResNeXtBlock.__init__.kernel_size__3': 1, 'ResNeXtBlock.__init__.kernel_size__4': 1,
        'GPN_Optimized.__init__.reduction': 2, 'GPN_Optimized.__init__.Conv2d_arg0': 1,
        'GPN_Optimized.__init__.Conv2d_arg1': 4, 'GPN_Optimized.__init__.BatchNorm2d_arg0': 4,
        'GPN_Optimized.__init__.kernel_size': 3, 'GPN_Optimized.__init__.stride': 1,
        'GPN_Optimized.__init__.padding': 1, 'GPN_Optimized.__init__.padding__2': 0,
        'GPN_Optimized.forward.size_or_budget': 4, 'GPN_Optimized.forward.size_or_budget__2': 4,
        'GPNTrainer.train_step_batch.max_norm': 1.0,
        'GPNTrainer.train.step_size': 1, 'GPNTrainer.train.gamma': 0.9,
    }
    for index in range(2, 7):
        suffix[f'GPN_Optimized.__init__.kernel_size__{index}'] = 2
    for index in range(2, 11):
        suffix[f'GPN_Optimized.__init__.stride__{index}'] = 2 if index % 2 == 0 else 1
    for index in range(1, 5):
        tail = '' if index == 1 else f'__{index}'
        suffix[f'GPN_Optimized.__init__.ResNeXtBlock_arg0{tail}'] = 4 if index == 1 else 8
        suffix[f'GPN_Optimized.__init__.ResNeXtBlock_arg1{tail}'] = 8
        suffix[f'GPN_Optimized.__init__.cardinality{tail}'] = 1
        suffix[f'GPN_Optimized.__init__.reduction__{index + 1}'] = 2
    scopes = ('GPN_V4_adaMulti_clean.py', 'GPN_V4_adaMulti_clean_clustering.py',
              'GPN_V5_adaMulti_clean_FEAT.py', 'GPN_V5_adaMulti_clean_UNEM.py')
    values = {f'{scope}.{key}': value for scope in scopes for key, value in suffix.items()}
    values.update({
        'GPN_V5_adaMulti_clean_FEAT.py.GPNTrainer.__init__.feature_dim': 4,
        'GPN_V5_adaMulti_clean_FEAT.py.FEATAdaptor.__init__.num_layers': 1,
        'loss_ada_num_clean_UNEM.py.UNEMClassGMMHead.__init__.init_lambda': 0.5,
        'loss_ada_num_clean_UNEM.py.UNEMClassGMMHead.__init__.init_temperature': 2.0,
    })
    return values


@unittest.skipIf(torch is None, 'Install runtime dependencies to run tensor integration tests')
class RuntimeSmokeTests(unittest.TestCase):
    def test_representative_branches(self):
        from amgpn.data.episodes import MetaDataset
        from amgpn.experiments import create_experiment
        torch.set_num_threads(1)
        torch.manual_seed(20261008)
        data = [(torch.randn(64, 64) + label * 0.2, label)
                for label in range(3) for _ in range(6)]
        episodes = MetaDataset(data, n_way=3, k_shot=3, q_query=2, num_tasks=2, seed=31)
        loader = torch.utils.data.DataLoader(episodes, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        with tempfile.TemporaryDirectory(prefix='amgpn-smoke-') as temporary:
            settings = Path(temporary) / 'synthetic.json'
            settings.write_text(json.dumps(synthetic_settings()))
            with patch.dict(os.environ, {'AMGPN_PRIVATE_CONFIG': str(settings)}):
                for identifier in IDS:
                    with self.subTest(branch=identifier), contextlib.redirect_stdout(io.StringIO()):
                        options = dict(save_middle=False, mlr=0.002, crop_ratio_h=0,
                                       crop_ratio_l=0, resize=False, merge_threshold=0.1)
                        if identifier == 'feat':
                            options.update(FEAT_heads=2, use_Mdistance=True)
                        if identifier == 'unem':
                            options.update(unem_layers=2, unem_use_diag_precision=True,
                                           unem_gmm_components=2)
                        components = create_experiment(identifier, device=torch.device('cpu'), trainer_kwargs=options)
                        trainer = components.trainer
                        modules = [value for value in vars(trainer).values() if isinstance(value, torch.nn.Module)]
                        for module in modules:
                            module.eval()
                        with torch.no_grad():
                            loss, accuracy, output = trainer._single_task_forward(*[item[0] for item in batch[:4]])
                        self.assertTrue(torch.isfinite(loss).item())
                        self.assertEqual(tuple(output['probabilities'].shape), (6, 3))
                        self.assertTrue(torch.allclose(output['probabilities'].sum(1), torch.ones(6), atol=1e-5))
                        parameters = [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]
                        before = [parameter.detach().clone() for parameter in parameters]
                        loss_value, accuracy, _ = trainer.train_step_batch(batch, 0)
                        self.assertTrue(math.isfinite(loss_value))
                        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, parameters)))
                        for parameter in parameters:
                            self.assertTrue(torch.isfinite(parameter).all().item())
                        if identifier in ('feat', 'unem'):
                            head = trainer.feat_adaptor if identifier == 'feat' else trainer.loss_fn
                            self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.parameters()))
                        if identifier == 'amgpn' or identifier.startswith('merge-'):
                            # Force a merge independently of random backbone distances.
                            embeddings = torch.zeros(9, 4)
                            precision = torch.eye(4).repeat(9, 1, 1)
                            mask = torch.ones(3, 3, dtype=torch.bool)
                            _, _, merged_mask, count = trainer.loss_fn._merge_prototypes_in_task_optimized(
                                embeddings, precision, mask, 3, 3)
                            self.assertEqual(count, 6)
                            self.assertEqual(merged_mask.sum().item(), 3)
                        result = trainer.evaluate(loader, n_ways=3, show_progress=False, show_error_stats=False)
                        self.assertTrue(math.isfinite(result))
                        self.assertTrue(0 <= result <= 1)
                        checkpoint = Path(temporary) / f'{identifier}.pth'
                        trainer.save_model(str(checkpoint))
                        expected = [{key: value.clone() for key, value in module.state_dict().items()} for module in modules]
                        with torch.no_grad():
                            for parameter in parameters:
                                parameter.add_(0.5)
                        trainer.load_model(str(checkpoint))
                        for module, state in zip(modules, expected):
                            for key, value in state.items():
                                self.assertTrue(torch.equal(value, module.state_dict()[key]), f'{identifier}: {key}')


if __name__ == '__main__':
    unittest.main()
