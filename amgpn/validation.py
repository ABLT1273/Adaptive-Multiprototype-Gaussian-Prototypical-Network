"""Structural checks for the maintained package and experiment catalog."""

import ast
import json
from pathlib import Path

from amgpn.experiments import catalog


def validate_repository():
    root = Path(__file__).resolve().parent.parent
    package = root / "amgpn"
    metadata = catalog()
    problems = []
    for experiment_id, spec in metadata["experiments"].items():
        module_path = root / (spec["module"].replace(".", "/") + ".py")
        if not module_path.is_file():
            problems.append(f"{experiment_id}: missing module {spec['module']}")
            continue
        tree = ast.parse(module_path.read_text())
        names = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        names.update(target.id for node in tree.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name))
        names.update(alias.asname or alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names)
        for field in ("model", "trainer"):
            if spec.get(field) and spec[field] not in names:
                problems.append(f"{experiment_id}: missing export {spec[field]}")
        for workflow in spec["workflows"]:
            if not (root / workflow).is_file():
                problems.append(f"{experiment_id}: missing workflow {workflow}")
    for group_id, spec in metadata["groups"].items():
        for member in spec["members"]:
            if member not in metadata["experiments"]:
                problems.append(f"{group_id}: unknown member {member}")
    count = 0
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text())
        compile(tree, str(path), "exec")
        count += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("amgpn."):
                target = root / node.module.replace(".", "/")
                if not target.with_suffix(".py").exists() and not (target / "__init__.py").exists():
                    problems.append(f"{path.relative_to(root)}: missing package import {node.module}")
    template = json.loads((package / "config_templates/private_config.example.json").read_text())
    schema = json.loads((package / "config_templates/private_config.schema.json").read_text())
    if template.keys() != schema.keys() or any(value is not None for value in template.values()):
        problems.append("Configuration templates must have matching keys and contain only null values.")
    return {"python_modules_checked": count, "experiments": len(metadata["experiments"]), "groups": len(metadata["groups"]), "problems": problems}
