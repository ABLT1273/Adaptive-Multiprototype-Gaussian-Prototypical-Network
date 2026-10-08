# Experiment Groups and Contracts

The authoritative method registry is `amgpn/catalog/experiments.json`. Filenames,
plot labels and notebook order must not define experiment identity.

```bash
python -m amgpn list
python -m amgpn list --group merging
python -m amgpn describe unem
python -m amgpn groups
```

## Main Method-Family Comparison

| ID | Change from the prototype backbone | Method selectors |
| --- | --- | --- |
| `pn` | One mean prototype, Euclidean scoring | Single prototype; Euclidean metric |
| `gpn` | One Gaussian prototype, Mahalanobis scoring | Single prototype; Mahalanobis metric |
| `amgpn` | One initial Gaussian per support instance, then graph merging | Multiprototype; Mahalanobis metric |
| `feat` | Support self-attention and query-to-support adaptation | Single-prototype branch; distance choice remains an explicit/private setting |
| `unem` | Transductive Gaussian/GMM refinement using unlabeled queries | Multiprototype; UNEM enabled; Mahalanobis metric |

These methods share the cleaned backbone implementation. Identical Python code
does not guarantee identical configured dimensions, preprocessing, training
budgets, class partitions or query episodes. Those settings must be matched when
running a comparison. The FEAT and UNEM implementations are style comparisons,
not claims of exact equivalence to their original external releases.

`create_experiment()` pins the selectors that define an ID. For example, asking
for `amgpn` while passing `use_multi=False` is rejected instead of silently
running another method under the AMGPN label. Numerical settings remain external.

The maintained five-method recipe is `notebooks/evaluation/paper_comparisons.ipynb`. It selects all five IDs from the registry, evaluates deterministic episode indices, checks backbone shapes and preprocessing, and saves accuracy summaries externally. Historical four-method, merging, preprocessing, threshold and initialization cells are preserved in separate notebooks, with their original configuration identities.

## Merging Ablations

| ID | Initialization | Merging |
| --- | --- | --- |
| `amgpn` | Support instances | Graph connected components |
| `merge-pairwise` | Support instances | Pairwise merging |
| `merge-hierarchical` | Support instances | Hierarchical clustering |
| `merge-dbscan` | Support instances | DBSCAN |

Hold support/query features, metric, backbone and threshold policy fixed.
`amgpn.evaluation.comprehensive.ComprehensiveEvaluator.compare_merging_strategies`
also exposes a no-merging control; it is an evaluation utility rather than a
separate training engine in the maintained factory.

Workflow: `notebooks/evaluation/merging.ipynb`.

## Initialization Studies and Historical Engines

`fixed-kmeans`, `imp` and `gmm` locate the historical implementations. They have
**reference** status in the registry. Their backbones, preprocessing options and
trainer contracts differ, so invoking these historical engines together does
not establish a controlled initialization-only ablation.

For a controlled construction comparison, inspect
`ComprehensiveEvaluator.compare_initialization_strategies`, which applies
initialization strategies to one supplied trainer. Keep its embedding model and
episode data fixed. Execution-level validation is still required before using
these results as research evidence.

The integrated GMM implementation is `amgpn/legacy/v4/gmm.py`.
`docs/reference/gmm_patch.md` is only integration reference material and must not
be imported or executed as an experiment.

## Additional Baselines

- `cnn-nearest`: an episodic CNN embedding baseline with nearest-support class
  scoring. This trainer is not a conventional supervised softmax classifier.
- `matching`: a matching-network backbone and support-query attention trainer.

These additional baselines are kept separate from the five-method main table.
Workflow: `notebooks/comparisons/cnn_and_matching.ipynb`.

## Evaluation and Result Ownership

| Component | Responsibility | Contract boundary |
| --- | --- | --- |
| `models.amgpn.GPNTrainer` | Primary training and basic evaluation | Basic accuracy and optional statistics |
| `evaluation.analysis_trainer.GPNTrainer` | Per-task metrics, features, predictions and assignment figures | Rich results selected by its evaluation arguments |
| `evaluation.compare` | Existing multi-model notebook workflow | Expects the chosen trainers' compatible return formats |
| `evaluation.multi_model` | An earlier generic evaluator | Its episode unpacking and model-specific feature path require separate validation; not the universal runner for every registered method |
| `evaluation.comprehensive` / `evaluation.per_class` | Initialization/merging/sensitivity analysis and paper figures | Utilities applied to a supplied trainer |
| `profiling.deployment` | Parameters, FLOPs and timing | Uses the selected model/trainer and compatible prototype head |

Before executing an experiment, record its ID, effective method selectors,
private configuration revision, class split, episode sampling policy and
checkpoint provenance in an external run directory. Reported paper accuracies
remain reference results; structural refactoring does not reproduce them.
