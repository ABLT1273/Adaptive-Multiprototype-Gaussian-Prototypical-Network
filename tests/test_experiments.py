import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from amgpn.experiments import catalog, configuration_keys, create_experiment, describe, group
from amgpn.validation import validate_repository


class ExperimentIdentityTests(unittest.TestCase):
    def test_primary_group_has_one_identity_per_method(self):
        self.assertEqual(group("paper-main")["members"], ["pn", "gpn", "amgpn", "feat", "unem"])
        self.assertNotIn("gmm", group("paper-main")["members"])
        self.assertEqual(describe("feat")["generation"], "comparison-branch")

    def test_selector_conflict_is_rejected_before_import_or_model_creation(self):
        with patch("amgpn.experiments.import_module") as importer:
            with self.assertRaisesRegex(ValueError, "requires use_multi"):
                create_experiment("amgpn", device="cpu", trainer_kwargs={"use_multi": False})
            importer.assert_not_called()

    def test_legacy_engine_cannot_be_constructed_as_maintained_method(self):
        with patch("amgpn.experiments.import_module") as importer:
            with self.assertRaisesRegex(ValueError, "reference material"):
                create_experiment("gmm", device="cpu")
            importer.assert_not_called()

    def test_factory_selects_matching_and_primary_trainer_contracts(self):
        class Model:
            def __init__(self, **kwargs):
                self.options = kwargs

        class Trainer:
            def __init__(self, model, device, **kwargs):
                self.model, self.device, self.options = model, device, kwargs

        fake = types.SimpleNamespace(GPN_Optimized=Model, GPNTrainer=Trainer, MN_Backbone=Model, MatchingNetworkTrainer=Trainer)
        with patch("amgpn.experiments.import_module", return_value=fake):
            primary = create_experiment("amgpn", device="cpu", trainer_kwargs={"mlr": "external-value"})
            self.assertEqual(primary.trainer.options["use_Mdistance"], True)
            self.assertEqual(primary.trainer.options["use_multi"], True)
            self.assertEqual(primary.trainer.options["mlr"], "external-value")
            matching = create_experiment("matching", device="cpu")
            self.assertEqual(matching.trainer.options, {})

    def test_catalog_and_namespace_ownership_are_complete(self):
        self.assertFalse(validate_repository()["problems"])
        for name in catalog()["experiments"]:
            self.assertTrue(configuration_keys(name), name)

    def test_inspection_does_not_import_model_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-c", "import runpy,sys; sys.argv=['amgpn','list']; runpy.run_module('amgpn',run_name='__main__')"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("amgpn", result.stdout)
        result = subprocess.run(
            [sys.executable, "-c", "import amgpn.experiments,sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_ids_have_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, "Unknown experiment"):
            describe("not-a-method")
        with self.assertRaisesRegex(ValueError, "Unknown experiment group"):
            group("not-a-group")
