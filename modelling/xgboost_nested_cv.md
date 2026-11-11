# XGBoost nested subject-grouped evaluation

`xgboost_nested_cv.py` re-evaluates the historical XGBoost parameter configurations
from JSONL logs across the three feature sets defined in `ML-DL, Project 1.pdf`.
It starts from `dataset/full.csv`; it does not load already-pruned or PCA-transformed
CSVs, because those may have been fitted outside the CV folds.

## Run

From the repository root:

```bash
python modelling/xgboost_nested_cv.py
```

By default this reads `dataset/experiment_logs.jsonl` and writes a uniquely named
run directory under `logs/xgb_nested_cv/`. If an incomplete run with the same data,
parameter logs, configurations, fold settings, software versions, and script contents
exists, it resumes that run automatically instead. Additional JSONL logs can be
included:

```bash
python modelling/xgboost_nested_cv.py \
  --hyperparameter-log dataset/experiment_logs.jsonl \
  --hyperparameter-log path/to/another_experiment.jsonl
```

To check the runtime and output format with a small subset first:

```bash
python modelling/xgboost_nested_cv.py --max-configs 2
```

Do not use a limited run for final conclusions. The full search can require many
hours because it evaluates every distinct logged configuration with nested CV,
including historical configurations with a large `n_estimators`.

To explicitly resume a particular interrupted run:

```bash
python modelling/xgboost_nested_cv.py --output-dir logs/xgb_nested_cv/RUN_ID
```

Resume matching is strict: changed data, hyperparameter logs, effective
configurations, fold/seed settings, package versions, or script contents require a
new run. Use `--no-resume` to always start a fresh timestamped run. Completed runs
are not resumed. The run log is appended on resume, and a final incomplete JSONL
line is discarded as a possible interrupted write; malformed complete lines stop
the run with an error.

## Dataset variants and leakage safeguards

The variants mirror the report's feature-engineering stages:

| Variant | Stages | Fold-fitted steps |
|---|---:|---|
| Dataset 1 | 0-1 | Exact duplicate-column removal |
| Dataset 2 | 0-4 | Duplicate removal, fixed redundant FFT feature removal, variance threshold (`0.01`), correlation filter (`0.95`) |
| Dataset 3 | 0-6 | Dataset 2 steps, Yeo-Johnson transform, PCA retaining `95%` variance |

Duplicate detection, variance pruning, correlation pruning, Yeo-Johnson parameters,
and PCA are fitted separately inside every inner and outer training fold. The FFT
feature-name rules are fixed in advance and do not learn from validation/test values.
All rows for one subject stay together via `StratifiedGroupKFold`.

The inner CV pooled out-of-fold macro-F1 selects a parameter configuration for each
variant and also selects the overall variant/configuration pair. The outer held-out
subjects estimate those choices. There is no separate final holdout test set in this
run; `outer_fold_metrics.csv` and `summary_metrics.csv` are the outer-CV results.

**Important historical limitation:** the configurations were collected from
experiments whose results were already inspected. The set of candidate values may
therefore reflect earlier tuning on the same subjects. Nested CV protects each
outer fold from direct code-driven selection, but cannot undo researcher knowledge
from prior runs. Describe these results as an exploratory, subject-aware re-evaluation,
not as a fully untouched confirmatory test. For a genuinely independent final test,
collect new subjects or use a prospectively sealed dataset that was never inspected.

## Duplicate configuration handling

The JSONL loader reads `hyperparameters` or `parameters` objects, validates that the
remaining keys are recognized XGBoost parameters, and removes duplicate effective
parameter dictionaries while preserving every source record in the run metadata.
Legacy/non-estimator fields (`num_class`, `num_round`, objective/metric and stored
seeds) do not create separate configurations; the script sets the objective, metric,
tree method, and random seed consistently for all runs. Malformed JSON or unknown
parameter names fail explicitly rather than being silently ignored.

## Generated files

Each run directory includes:

- `REPORT.md`: ready-to-edit summary of the protocol, metrics, class results, tests,
  configuration selection, and important caveats.
- `run.log`: timestamped progress, fold subject assignments, configuration scores,
  and errors.
- `run_manifest.json`: the inputs, software versions, effective configurations, and
  planned subject assignments, written before model fitting starts.
- `checkpoint_results.jsonl` and `checkpoint_state.json`: flushed and synced after
  each completed inner fit, candidate, selection, and outer evaluation. If a run is
  interrupted, these contain all results through the last completed event; a model
  interrupted while fitting cannot have a score.
- `run_metadata.json`: command, file SHA-256 hashes, candidate provenance, fold
  counts, feature and subject counts, package versions, Git commit/worktree status,
  and model-fit count.
- `inner_candidate_scores.csv`, `selected_configurations.csv`,
  `outer_fold_metrics.csv`, `summary_metrics.csv`, and `class_metrics.csv`.
- `oof_predictions.csv`, `confusion_matrices.json`, `fold_assignments.json`, and
  `statistical_tests.json`.

The checkpoint JSONL is append-only. Every `inner_fit_completed` record contains
the fit's validation score, class-level scores, confusion matrix, subject IDs, and
compact per-row prediction arrays. `inner_candidate_completed` records contain
pooled inner out-of-fold results once all inner folds for that candidate are complete.
`outer_evaluation_completed` records contain the held-out-subject results as each
variant finishes. On restart, completed inner fits rebuild their validation
predictions, and completed outer evaluations rebuild held-out predictions, so those
model fits are not repeated. Only a fit without its durable completion record needs
to be rerun. Selection and final output tables are reconstructed from the
checkpoints; final CSV and report files are regenerated when the run completes.

The summary table reports pooled outer-fold accuracy, macro precision, macro recall,
and macro-F1 for all three variants. Per-class precision/recall/F1 and support are
also reported. The statistical analysis compares variants using per-subject
out-of-fold macro-F1: a Friedman omnibus test followed by paired Wilcoxon tests with
Holm correction. Interpret p-values cautiously: there are only 30 subjects, outer
fold models share training subjects, and the earlier search history may have seen
these subjects' results.

The emitted report explains the train/validation/test roles: inner folds select,
outer folds estimate, and no separate final test partition is used. It also explicitly
warns against presenting historical, previously inspected hyperparameter exploration
as an independent test.
