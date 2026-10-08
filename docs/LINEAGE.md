# Implementation Lineage

Version numbers in the original directory were development labels. They are not
reliable method identities, chronological release tags, or Python inheritance trees.
The relationships below are established from imports, class bodies, notebook
entry points and source similarity. Git history does not contain the original
development sequence, so dates and exact ancestry are not inferred.

| Family | Original material | Current owner | Mechanism and relationship |
| --- | --- | --- | --- |
| Original prototypes | `old/GPN_ori.py`, `old/GPN_change.py`, `old/GPN_new.py`, `GPN_trainer.py` | `amgpn/legacy/original/` | Early single-prototype, learned-metric and alternative-backbone experiments. The legacy Meta-SGD trainers actually subclass their local GPN trainer. |
| V2 | `GPN_V2.py`, `loss_Copy1.py` | `amgpn/legacy/v2/` | Single Gaussian prototypes with learned metric and attention/backbone options. Source edits build on earlier files; the GPN trainer is not a subclass of an earlier-generation trainer. |
| V3 | `GPN_V3.py`, IMP files | `amgpn/legacy/v3/` | Fixed-budget K-means prototypes and a separate supervised IMP allocation branch. Their historical backbones and contracts are not automatically controlled against the cleaned mainline. |
| Exploratory V4 | `GPN_V4.py`, `GPN_V4_adaMulti.py`, `GPN_V4_GMM.py` | `amgpn/legacy/v4/` | Fixed/adaptive prototypes, optional edge/corner features, alternative attention, and integrated iterative GMM estimation. Multiple experimental axes coexist here. |
| Clean AMGPN | `GPN_V4_adaMulti_clean.py`, `loss_ada_num_clean.py` | `amgpn/models/amgpn.py`, `amgpn/losses/adaptive.py` | Primary instance initialization and graph merging, with a reduced common backbone. |
| Merging ablations | `GPN_V4_adaMulti_clean_clustering.py`, matching loss | `amgpn/ablations/merging.py`, `amgpn/losses/merging.py` | Source adaptation of the clean trainer, adding explicit merging choices. It shares the backbone, but does not subclass the AMGPN trainer. |
| Analysis trainer | `GPN_V4_adaMulti_test.py` | `amgpn/evaluation/analysis_trainer.py` | Richer metrics, per-task outputs and assignment heatmaps. Its role is analysis, not a new algorithm generation. |
| Former V5: FEAT | `GPN_V5_adaMulti_clean_FEAT.py` | `amgpn/comparisons/feat.py` | Support self-attention and query-to-support adaptation. A separate comparison branch, not the successor to AMGPN. |
| Former V5: UNEM | `GPN_V5_adaMulti_clean_UNEM.py`, matching loss | `amgpn/comparisons/unem.py`, `amgpn/losses/unem.py` | Transductive Gaussian/GMM refinement, including trainable head parameters in the optimizer. A separate comparison branch. |

## Shared Implementation Versus Inheritance

The backbone class bodies in the clean method, merging trainer, analysis trainer,
FEAT and UNEM were identical after normalizing their configuration-key prefixes.
They now have one owner: `amgpn/models/backbone.py`.

`configured_backbone(namespace)` creates a small configuration-bound subclass of
the shared `GPN_Optimized`. The underlying SE and ResNeXt blocks remain shared.
Namespace context is active for both construction and forward execution, and is
restored after the call. Tensor operations and state-dictionary parameter names
retain the original backbone structure.

The trainer families are not collapsed into one class: FEAT adaptation, UNEM
optimizer ownership, merging choices and evaluation return formats differ.
These differences are documented through method identities and component
ownership instead of being hidden behind version numbers.

## Notebook Names Were Also Misleading

- The original `GPN_V3.ipynb` imported exploratory V4 adaptive code.
- The original `GPN_V3_clean.ipynb` used the V4 analysis trainer.
- `GPN_V5.ipynb` contained two separate FEAT and UNEM training cells; these are now
  separate notebooks in `notebooks/comparisons/`.
- `GPN_V4_test.ipynb` and `GPN_V5_test.ipynb` contained multiple experiment groups,
  including historical comparisons. Those cells are now separated by experiment axis; published-reference plots are also separate from measured evaluation. The new main comparison recipe explicitly selects the five registry IDs.

The migration map preserves all original locations. Numerical configuration
values, datasets and checkpoints remain outside the repository.
