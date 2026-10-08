import ast
from functools import lru_cache
from pathlib import Path
import unittest
from unittest.mock import patch

from amgpn.configuration import backbone_scope, require_backbone, resolve_backbone


class NamespaceTests(unittest.TestCase):
    def test_nested_scope_and_exception_restore_the_previous_scope(self):
        with patch("amgpn.configuration.require", side_effect=lambda key: key):
            before = require_backbone("setting")
            with backbone_scope("variant-a"):
                self.assertEqual(require_backbone("setting"), "variant-a.setting")
                with self.assertRaises(LookupError):
                    with backbone_scope("variant-b"):
                        self.assertEqual(require_backbone("setting"), "variant-b.setting")
                        raise LookupError("construction failed")
                self.assertEqual(require_backbone("setting"), "variant-a.setting")
            self.assertEqual(require_backbone("setting"), before)

    def test_explicit_false_and_zero_are_not_replaced_by_external_defaults(self):
        with patch("amgpn.configuration.require") as reader:
            self.assertIs(resolve_backbone("setting", False), False)
            self.assertEqual(resolve_backbone("setting", 0), 0)
            reader.assert_not_called()

    def test_configured_backbone_keeps_its_scope_during_forward(self):
        # Exercise the actual configuration wrapper without importing Torch or
        # instantiating a real model; forward-time channel splits need this scope.
        source = ast.parse(Path("amgpn/models/backbone.py").read_text())
        factory = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "configured_backbone")
        class FakeBackbone:
            def __init__(self, reduction=None):
                self.constructed_scope = require_backbone("construct")

            def forward(self, x, return_intermediate=False):
                if x == "fail":
                    raise LookupError("forward failed")
                return require_backbone("channel_split")

        namespace = {"GPN_Optimized": FakeBackbone, "backbone_scope": backbone_scope, "lru_cache": lru_cache}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), "backbone-wrapper", "exec"), namespace)
        with patch("amgpn.configuration.require", side_effect=lambda key: key):
            model = namespace["configured_backbone"]("feat-scope")()
            self.assertEqual(model.constructed_scope, "feat-scope.construct")
            self.assertEqual(model.forward("input"), "feat-scope.channel_split")
            with self.assertRaises(LookupError):
                model.forward("fail")
            self.assertEqual(require_backbone("setting"), "GPN_V4_adaMulti_clean.py.setting")
