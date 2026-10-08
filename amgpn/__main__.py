"""Inspect experiment identity without importing PyTorch or starting an experiment."""

import argparse
import json
import sys

from amgpn.experiments import catalog, configuration_keys, describe, group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List named methods and their role")
    listing.add_argument("--group", help="Restrict the list to an experiment group")
    detail = commands.add_parser("describe", help="Show the implementation, options and workflow for one method")
    detail.add_argument("experiment")
    commands.add_parser("groups", help="Show comparison groups and controlled axes")
    keys = commands.add_parser("config-keys", help="List required configuration namespaces without revealing values")
    keys.add_argument("experiment")
    commands.add_parser("check", help="Validate repository structure without loading a model")
    args = parser.parse_args()
    try:
        if args.command == "list":
            data = catalog()["experiments"]
            names = group(args.group)["members"] if args.group else data
            for name in names:
                spec = data[name]
                print(f"{name:22} {spec['role']:16} {spec['status']:12} {spec['title']}")
        elif args.command == "describe":
            print(json.dumps(describe(args.experiment), indent=2))
        elif args.command == "groups":
            print(json.dumps(catalog()["groups"], indent=2))
        elif args.command == "config-keys":
            for key, kind in configuration_keys(args.experiment).items():
                print(f"{key}: {kind}")
        else:
            from amgpn.validation import validate_repository
            result = validate_repository()
            print(json.dumps(result, indent=2))
            return bool(result["problems"])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
