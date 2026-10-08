import ast
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent.parent


class MigrationTests(unittest.TestCase):
    def test_data_source_directories_are_not_ignored_as_private_datasets(self):
        for relative in ["amgpn/data/episodes.py", "amgpn/data/preprocessing.py", "notebooks/data/download.ipynb"]:
            result = subprocess.run(["git", "check-ignore", "--no-index", "-q", relative], cwd=ROOT)
            self.assertEqual(result.returncode, 1, relative)
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", "data/private_notes.txt"], cwd=ROOT)
        self.assertEqual(result.returncode, 0)

    def test_all_old_locations_have_existing_replacements(self):
        mapping = json.loads((ROOT / "docs/migration_map.json").read_text())
        for section in mapping.values():
            for old, destinations in section.items():
                self.assertFalse((ROOT / old).exists(), old)
                for destination in destinations if isinstance(destinations, list) else [destinations]:
                    self.assertTrue((ROOT / destination).is_file(), destination)

    def test_notebooks_compile_and_do_not_contain_saved_outputs(self):
        for path in (ROOT / "notebooks").rglob("*.ipynb"):
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] != "code":
                    continue
                self.assertEqual(cell["outputs"], [], str(path))
                self.assertIsNone(cell["execution_count"], str(path))
                source = "\n".join("pass" if line.lstrip().startswith(("!", "%")) and not line.lstrip().startswith("!!!") else line for line in "".join(cell["source"]).splitlines())
                compile(source, str(path), "exec")

    def test_shared_backbone_is_not_copied_into_active_variants(self):
        variants = ["models/amgpn.py", "ablations/merging.py", "evaluation/analysis_trainer.py", "comparisons/feat.py", "comparisons/unem.py"]
        for variant in variants:
            tree = ast.parse((ROOT / "amgpn" / variant).read_text())
            names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
            self.assertTrue(names.isdisjoint({"SEModule", "ResNeXtBlock", "GPN_Optimized"}), variant)

    def test_each_shared_backbone_binding_has_all_frozen_configuration_keys(self):
        shared = ast.parse((ROOT / "amgpn/models/backbone.py").read_text())
        suffixes = {node.args[0].value for node in ast.walk(shared)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"_cfg_require", "_cfg_resolve"}
                    and isinstance(node.args[0], ast.Constant)}
        schema = json.loads((ROOT / "amgpn/config_templates/private_config.schema.json").read_text())
        for relative in ["models/amgpn.py", "ablations/merging.py", "evaluation/analysis_trainer.py", "comparisons/feat.py", "comparisons/unem.py"]:
            tree = ast.parse((ROOT / "amgpn" / relative).read_text())
            binding = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name) and node.func.id == "configured_backbone")
            scope = binding.args[0].value
            for suffix in suffixes:
                self.assertIn(scope + "." + suffix, schema)

    def test_templates_are_empty_and_readme_keeps_math_compatibility(self):
        template = json.loads((ROOT / "amgpn/config_templates/private_config.example.json").read_text())
        self.assertTrue(all(value is None for value in template.values()))
        readme = (ROOT / "README.md").read_text()
        self.assertNotIn(r"\operatorname", readme)
        self.assertFalse(re.search(r"[\u4e00-\u9fff]", readme))
        self.assertEqual(readme.count("$$"), 6)
