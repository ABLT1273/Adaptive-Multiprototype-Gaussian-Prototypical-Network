"""Experiment identity and construction, independent of legacy filename numbering."""

from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent / "catalog" / "experiments.json"


def catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def describe(experiment_id):
    try:
        return catalog()["experiments"][experiment_id]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment: {experiment_id}") from exc


def group(group_id):
    try:
        return catalog()["groups"][group_id]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment group: {group_id}") from exc


def configuration_keys(experiment_id):
    from amgpn.config import _schema
    scopes = describe(experiment_id)["config_scopes"]
    return {key: kind for key, kind in _schema().items() if any(key.startswith(scope + ".") for scope in scopes)}


@dataclass
class ExperimentComponents:
    experiment_id: str
    model: object
    trainer: object
    specification: dict


def create_experiment(experiment_id, *, device, model_kwargs=None, trainer_kwargs=None):
    """Construct a named method without allowing accidental changes to its identity.

    Numerical settings remain external. Construction is explicit and imports ML
    dependencies only when this function is called; catalog inspection is lightweight.
    """
    spec = describe(experiment_id)
    if spec["status"] != "maintained":
        raise ValueError(f"{experiment_id} is reference material; use its documented historical workflow explicitly.")
    options = dict(trainer_kwargs or {})
    for key, value in spec["method_options"].items():
        if key in options and options[key] != value:
            raise ValueError(f"{experiment_id} requires {key}={value!r}; choose a different experiment identity.")
        options[key] = value
    module = import_module(spec["module"])
    model = getattr(module, spec["model"])(**dict(model_kwargs or {}))
    trainer = getattr(module, spec["trainer"])(model=model, device=device, **options)
    return ExperimentComponents(experiment_id, model, trainer, spec)
