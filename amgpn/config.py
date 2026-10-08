"""Load undisclosed settings from an external JSON file, without embedded defaults."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_TEMPLATES = Path(__file__).resolve().parent / "config_templates"


class ConfigurationError(ValueError):
    """A required private setting is missing or invalid."""


@lru_cache(maxsize=None)
def _read(path: str, modified: int) -> dict:
    del modified
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("Cannot read AMGPN_PRIVATE_CONFIG as a JSON object.") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("AMGPN_PRIVATE_CONFIG must contain a JSON object.")
    return data


def _configuration() -> dict:
    configured = os.environ.get("AMGPN_PRIVATE_CONFIG")
    if not configured:
        raise ConfigurationError(
            'Set AMGPN_PRIVATE_CONFIG to an external JSON configuration file. The public project does not contain real experimental settings.'
        )
    path = Path(configured).expanduser().resolve()
    if path.is_relative_to(PROJECT_ROOT):
        raise ConfigurationError("Keep the private configuration outside the project directory.")
    try:
        modified = path.stat().st_mtime_ns
    except OSError as exc:
        raise ConfigurationError("The external configuration file is unavailable.") from exc
    return _read(str(path), modified)


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads((CONFIG_TEMPLATES / "private_config.schema.json").read_text(encoding="utf-8"))


def require(key: str):
    """Return a private value or fail explicitly; never infer an experimental default."""
    values = _configuration()
    if key not in values or values[key] is None:
        raise ConfigurationError(f"Missing private configuration key: {key}")
    value = values[key]
    expected = _schema().get(key)
    checks = {
        "boolean": lambda v: isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "string": lambda v: isinstance(v, str) and bool(v.strip()),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    if expected in checks and not checks[expected](value):
        raise ConfigurationError(f"Invalid type for private configuration key: {key}; expected {expected}.")
    return value


def resolve(key: str, explicit=None):
    """Use a caller-supplied value, otherwise resolve the external setting lazily."""
    return explicit if explicit is not None else require(key)


def format_value(key: str, **fields) -> str:
    """Expand an external filename template using the original runtime fields."""
    template = require(key)
    if not isinstance(template, str):
        raise ConfigurationError(f"Expected a string template for private configuration key: {key}")
    try:
        return template.format(**fields)
    except (KeyError, ValueError, IndexError) as exc:
        raise ConfigurationError(f"Invalid filename template for private configuration key: {key}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate external settings without loading a model")
    parser.add_argument("--scope", default="", help="Only validate keys with this file or module prefix")
    parser.add_argument("--list-keys", action="store_true", help="List names and types without showing values")
    args = parser.parse_args()
    selected = {k: v for k, v in _schema().items() if k.startswith(args.scope)}
    if not selected:
        parser.error("No configuration keys match the requested scope.")
    if args.list_keys:
        for key, kind in selected.items():
            print(f"{key}: {kind}")
    if args.check:
        failures = []
        for key in selected:
            try:
                require(key)
            except ConfigurationError as exc:
                failures.append(str(exc))
        if failures:
            for failure in dict.fromkeys(failures):
                print(failure, file=sys.stderr)
            return 1
        print(f"Validated {len(selected)} private settings; no models or datasets were loaded.")
    elif not args.list_keys:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
