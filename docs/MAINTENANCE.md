# Maintenance and Migration Guide

## Ownership Rules

- Change the cleaned shared backbone in `amgpn/models/backbone.py`, not in a
  comparison trainer. Active comparison/ablation modules bind configuration
  namespaces to this one implementation.
- Change primary instance initialization and graph merging in
  `amgpn/losses/adaptive.py`.
- Add a merging mechanism in `amgpn/losses/merging.py` and register a named
  ablation, rather than copying a new version-numbered model file.
- Keep FEAT adaptation in `amgpn/comparisons/feat.py`, and UNEM heads in
  `amgpn/losses/unem.py`. UNEM head parameters belong in the optimizer.
- Keep frequency cropping in `amgpn/data/preprocessing.py`; analysis figures
  live under `amgpn/visualization/`.
- Historical implementations under `amgpn/legacy/` are reference material.
  Do not use a historical filename as the identity of a new maintained method.

## Configuration Compatibility

All real settings remain outside the repository. Empty templates and types live
under `amgpn/config_templates/`.

Original file/cell-based keys are intentionally frozen. Moving a notebook or
splitting the former V5 notebook does not renumber its external keys. Existing
private files therefore remain usable without being copied into the repository.
Use `docs/migration_map.json` to find the current owner of an old source name.

```bash
python -m amgpn config-keys amgpn
python -m amgpn.config --check --scope GPN_V4_adaMulti_clean.py
```

The shared backbone binds the original variant namespace during initialization
and forward execution. This includes the configured channel split, not just
convolution construction. Context is restored even when an exception occurs.

## Adding an Experiment

1. Choose an existing component boundary: comparison, ablation or legacy reference.
2. Define what changes and what stays controlled. Do not call a historical
   architecture change an initialization-only ablation.
3. Register a unique method ID with module/class ownership, semantic selectors,
   configuration scopes, workflow links and validation status.
4. Add the ID to the appropriate group. Keep the main table separate from
   secondary baselines and historical initialization studies.
5. Verify selector conflicts are rejected and imports do not trigger training.
6. Validate numerical behavior with appropriate data/weights before claiming
   reproduction or paper-equivalent results.

## Checks

```bash
python -m amgpn check
python -m unittest discover -s tests -v
git diff --check
```

The inspection CLI and structural tests use the standard library. They do not
import PyTorch, download datasets, load checkpoints, or perform inference.
Model execution requires the runtime dependencies and externally supplied
settings. Run `python -m pip install -e .` before opening notebooks from nested
directories.

## Historical Interface Repairs

The migration fixes imports of `GPNLoss` to its actual owning metric-loss module,
replaces nonexistent `draw_pic` imports with the existing single-model plotting
module, and connects the per-class heatmap notebook to
`plot_multiple_assignment_heatmaps`. The old non-importable GMM patch is preserved
as Markdown reference material. These are ownership/interface repairs, not changes
to prototype equations.

Legacy examples may still depend on externally supplied dataset objects or
unfinished combinations of older trainer interfaces. Their reference status is
explicit; successful syntax or catalog checks do not establish runtime validity.
