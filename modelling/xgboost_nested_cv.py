#!/usr/bin/env python3
"""Nested subject-grouped CV for the three feature sets in Project 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "full.csv"
DEFAULT_PARAMETER_LOG = PROJECT_ROOT / "dataset" / "experiment_logs.jsonl"

VARIANTS = (
    ("dataset_1", "Dataset 1 (stages 0-1)", 1),
    ("dataset_2", "Dataset 2 (stages 0-4)", 4),
    ("dataset_3", "Dataset 3 (stages 0-6)", 6),
)

IGNORED_PARAMETER_KEYS = {
    "eval_metric",
    "n_jobs",
    "num_class",
    "num_round",
    "objective",
    "random_state",
    "seed",
    "use_label_encoder",
}

XGB_CONFIG_DEFAULTS = {
    "booster": "gbtree",
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "grow_policy": "depthwise",
    "learning_rate": 0.3,
    "max_bin": 256,
    "max_depth": 6,
    "max_delta_step": 0,
    "max_leaves": 0,
    "min_child_weight": 1.0,
    "n_estimators": 100,
    "num_parallel_tree": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "sampling_method": "uniform",
    "subsample": 1.0,
    "tree_method": "hist",
}

SUPPORTED_PARAMETER_KEYS = {
    "base_score",
    "booster",
    "colsample_bylevel",
    "colsample_bynode",
    "colsample_bytree",
    "device",
    "enable_categorical",
    "gamma",
    "grow_policy",
    "interaction_constraints",
    "learning_rate",
    "max_bin",
    "max_delta_step",
    "max_depth",
    "max_leaves",
    "min_child_weight",
    "missing",
    "monotone_constraints",
    "n_estimators",
    "num_parallel_tree",
    "reg_alpha",
    "reg_lambda",
    "sampling_method",
    "scale_pos_weight",
    "subsample",
    "tree_method",
    "verbosity",
}

FFT_PRUNE_PATTERNS = (
    re.compile(r"^f.*-mean\(\)"),
    re.compile(r"^f.*-max\(\)"),
    re.compile(r"^f.*-min\(\)"),
    re.compile(r"^f.*-energy\(\)"),
)


class DuplicateColumnFilter(BaseEstimator, TransformerMixin):
    """Drop exact duplicate columns, learning which to keep from the fit rows."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> DuplicateColumnFilter:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DuplicateColumnFilter expects a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.keep_mask_ = ~X.T.duplicated(keep="first").to_numpy()
        if not self.keep_mask_.any():
            raise ValueError("Duplicate-column filtering removed every feature.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DuplicateColumnFilter expects a pandas DataFrame.")
        if not np.array_equal(np.asarray(X.columns, dtype=object), self.feature_names_in_):
            raise ValueError("Feature columns changed between fit and transform.")
        return X.loc[:, self.keep_mask_]


class FFTFeaturePruner(BaseEstimator, TransformerMixin):
    """Remove the hand-specified redundant frequency-domain feature families."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> FFTFeaturePruner:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FFTFeaturePruner expects a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.drop_columns_ = [
            column
            for column in X.columns
            if any(pattern.search(str(column)) for pattern in FFT_PRUNE_PATTERNS)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FFTFeaturePruner expects a pandas DataFrame.")
        if not np.array_equal(np.asarray(X.columns, dtype=object), self.feature_names_in_):
            raise ValueError("Feature columns changed between fit and transform.")
        return X.drop(columns=self.drop_columns_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop later columns correlated above the configured absolute threshold."""

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> CorrelationFilter:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("CorrelationFilter expects a two-dimensional array.")
        if values.shape[1] < 1:
            raise ValueError("CorrelationFilter received no features.")
        corr = pd.DataFrame(values).corr().abs().to_numpy()
        upper = np.triu(corr, k=1)
        self.drop_mask_ = np.any(upper > self.threshold, axis=0)
        self.n_features_in_ = values.shape[1]
        if self.drop_mask_.all():
            raise ValueError("Correlation filtering removed every feature.")
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Feature shape changed between correlation fit and transform.")
        return values[:, ~self.drop_mask_]


def build_preprocessor(stage: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("duplicate_columns", DuplicateColumnFilter()),
    ]
    if stage >= 2:
        steps.append(("fft_pruning", FFTFeaturePruner()))
    if stage >= 3:
        steps.append(("variance", VarianceThreshold(threshold=0.01)))
    if stage >= 4:
        steps.append(("correlation", CorrelationFilter(threshold=0.95)))
    if stage >= 5:
        steps.append(("yeo_johnson", PowerTransformer(method="yeo-johnson")))
    if stage >= 6:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    return Pipeline(steps)


def _normalize_parameter_value(key: str, value: Any) -> Any:
    integer_keys = {
        "max_bin",
        "max_depth",
        "max_leaves",
        "n_estimators",
        "num_parallel_tree",
        "verbosity",
    }
    if key in integer_keys and value is not None:
        integer = int(value)
        if float(value) != integer:
            raise ValueError(f"{key} must be an integer, got {value!r}.")
        return integer
    float_keys = {
        "base_score",
        "colsample_bylevel",
        "colsample_bynode",
        "colsample_bytree",
        "gamma",
        "learning_rate",
        "max_delta_step",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
        "subsample",
    }
    if key in float_keys and value is not None:
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_unique_configurations(
    log_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load configurations once, merging semantically irrelevant legacy keys."""
    configurations: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for path in log_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Hyperparameter log not found: {path}")
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {path}, line {line_number}: {error}"
                    ) from error
                raw_parameters = record.get(
                    "hyperparameters", record.get("parameters", {})
                )
                if not raw_parameters:
                    continue
                if not isinstance(raw_parameters, dict):
                    raise ValueError(
                        f"Parameters in {path}, line {line_number} must be an object."
                    )

                logged_parameters = {
                    key: _normalize_parameter_value(key, value)
                    for key, value in raw_parameters.items()
                    if key not in IGNORED_PARAMETER_KEYS
                }
                unsupported = set(logged_parameters) - SUPPORTED_PARAMETER_KEYS
                if unsupported:
                    raise ValueError(
                        f"Unsupported XGBoost parameters in {path}, line {line_number}: "
                        f"{sorted(unsupported)}"
                    )
                if not logged_parameters:
                    raise ValueError(
                        f"No XGBoost parameters remain in {path}, line {line_number}."
                    )
                parameters = dict(XGB_CONFIG_DEFAULTS)
                parameters.update(logged_parameters)
                key = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
                if key not in seen:
                    seen[key] = len(configurations)
                    configurations.append(parameters)
                    sources.append(
                        {
                            "config_id": f"xgb_{len(configurations):03d}",
                            "parameters": parameters,
                            "sources": [],
                            "duplicate_records_skipped": 0,
                        }
                    )
                config_index = seen[key]
                sources[config_index]["sources"].append(
                    {
                        "file": str(path.resolve()),
                        "line": line_number,
                        "timestamp": record.get("timestamp"),
                        "note": record.get("note"),
                        "logged_parameters": raw_parameters,
                    }
                )
                if len(sources[config_index]["sources"]) > 1:
                    sources[config_index]["duplicate_records_skipped"] += 1

    if not configurations:
        raise ValueError(
            "No hyperparameter configurations found. Expected JSONL records "
            "containing 'hyperparameters' or 'parameters'."
        )
    return configurations, sources


def make_model(parameters: dict[str, Any], seed: int) -> xgb.XGBClassifier:
    model_parameters = dict(parameters)
    model_parameters.update(
        {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": seed,
            "n_jobs": 1,
        }
    )
    return xgb.XGBClassifier(**model_parameters)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    labels = np.arange(n_classes)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]
) -> list[dict[str, Any]]:
    labels = np.arange(len(label_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return [
        {
            "class": label_names[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in labels
    ]


def _prediction_payload(
    row_indices: Any,
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, list[Any]]:
    return {
        "row_indices": [int(index) for index in row_indices],
        "subjects": [
            subject.item() if isinstance(subject, np.generic) else subject
            for subject in subject_ids
        ],
        "actual": [str(label_names[index]) for index in y_true],
        "predicted": [str(label_names[index]) for index in y_pred],
    }


def _restore_prediction_event(
    event: dict[str, Any],
    expected_indices: np.ndarray,
    row_indices: pd.Index,
    label_encoder: LabelEncoder,
    expected_labels: np.ndarray,
) -> np.ndarray:
    payload = event.get("predictions")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint prediction event is missing its predictions.")
    expected_row_ids = [int(row_indices[index]) for index in expected_indices]
    if payload.get("row_indices") != expected_row_ids:
        raise ValueError(
            "Checkpoint prediction rows do not match the current deterministic split."
        )
    labels = list(label_encoder.classes_)
    label_to_id = {str(label): index for index, label in enumerate(labels)}
    try:
        actual = np.asarray([label_to_id[value] for value in payload["actual"]])
        predicted = np.asarray([label_to_id[value] for value in payload["predicted"]])
    except (KeyError, TypeError) as error:
        raise ValueError("Checkpoint contains an unknown activity label.") from error
    if len(actual) != len(expected_indices) or len(predicted) != len(expected_indices):
        raise ValueError("Checkpoint prediction count does not match the current fold.")
    if not np.array_equal(actual, expected_labels):
        raise ValueError("Checkpoint validation labels do not match the dataset.")
    return predicted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_status": None}
    return {"commit": commit, "worktree_status": status}


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [1.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def statistical_comparisons(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    variant_keys = [key for key, _, _ in VARIANTS]
    subjects = np.unique(groups)
    subject_scores: dict[str, list[float]] = {}
    for variant in variant_keys:
        subject_scores[variant] = [
            float(
                f1_score(
                    y_true[groups == subject],
                    predictions[variant][groups == subject],
                    labels=np.arange(n_classes),
                    average="macro",
                    zero_division=0,
                )
            )
            for subject in subjects
        ]

    score_arrays = [np.asarray(subject_scores[key]) for key in variant_keys]
    if all(np.ptp(scores) == 0 for scores in score_arrays):
        friedman = {"statistic": 0.0, "p_value": 1.0}
    else:
        friedman_statistic, friedman_p_value = friedmanchisquare(*score_arrays)
        if not np.isfinite(friedman_statistic) or not np.isfinite(friedman_p_value):
            raise ValueError("Friedman test returned a non-finite result.")
        friedman = {
            "statistic": float(friedman_statistic),
            "p_value": float(friedman_p_value),
        }

    pairwise: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for left_index, left in enumerate(variant_keys):
        for right in variant_keys[left_index + 1 :]:
            left_scores = np.asarray(subject_scores[left])
            right_scores = np.asarray(subject_scores[right])
            differences = left_scores - right_scores
            if np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    left_scores,
                    right_scores,
                    alternative="two-sided",
                    method="auto",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
            raw_p_values.append(float(p_value))
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "subjects": len(subjects),
                    "mean_subject_f1_difference": float(np.mean(differences)),
                    "median_subject_f1_difference": float(np.median(differences)),
                    "wilcoxon_statistic": statistic,
                    "p_value": float(p_value),
                }
            )
    for result, adjusted in zip(pairwise, _holm_adjust(raw_p_values)):
        result["p_value_holm_adjusted"] = adjusted

    return {
        "unit_of_analysis": "one pooled out-of-fold macro-F1 per subject",
        "omnibus_test": {"name": "Friedman test", **friedman},
        "post_hoc_test": "paired Wilcoxon signed-rank tests with Holm correction",
        "pairwise": pairwise,
        "subject_scores": subject_scores,
        "caveat": (
            "Exploratory: only 30 subjects are available, and outer-fold models "
            "have overlapping training subjects. Historical tuning may also have "
            "been influenced by results on these same participants."
        ),
    }


def _configure_logging(output_dir: Path, resume: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("xgboost_nested_cv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_handler = logging.FileHandler(
        output_dir / "run.log",
        mode="a" if resume else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_checkpoint_events(checkpoint_path: Path) -> list[dict[str, Any]]:
    if not checkpoint_path.exists():
        return []
    contents = checkpoint_path.read_bytes()
    complete_length = contents.rfind(b"\n") + 1
    if complete_length != len(contents):
        with checkpoint_path.open("r+b") as file:
            file.truncate(complete_length)
            file.flush()
            os.fsync(file.fileno())
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        contents[:complete_length].decode("utf-8").splitlines(), start=1
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid checkpoint JSON at {checkpoint_path}, "
                f"line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict) or "event_type" not in event:
            raise ValueError(
                f"Invalid checkpoint event at {checkpoint_path}, line {line_number}."
            )
        events.append(event)
    return events


def _checkpoint_is_complete(checkpoint_path: Path, state_path: Path) -> bool:
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("last_event_type") == "run_completed":
                return True
        except (OSError, json.JSONDecodeError):
            pass
    events = _read_checkpoint_events(checkpoint_path)
    return bool(events and events[-1]["event_type"] == "run_completed")


def _resolve_output_directory(
    args: argparse.Namespace,
    run_signature: str,
) -> tuple[Path, bool]:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir = output_dir.resolve()
        manifest_path = output_dir / "run_manifest.json"
        checkpoint_path = output_dir / "checkpoint_results.jsonl"
        state_path = output_dir / "checkpoint_state.json"
        if output_dir.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Output directory already exists and --no-resume was set: "
                    f"{output_dir}"
                )
            if not manifest_path.is_file():
                if any(output_dir.iterdir()):
                    raise ValueError(
                        f"Cannot resume {output_dir}: run_manifest.json is missing."
                    )
                return output_dir, False
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_signature") != run_signature:
                raise ValueError(
                    f"Cannot resume {output_dir}: inputs, settings, or script version "
                    "do not match this run."
                )
            if _checkpoint_is_complete(checkpoint_path, state_path):
                raise ValueError(
                    f"Run in {output_dir} is already complete. Choose a new output "
                    "directory for a fresh run."
                )
            return output_dir, True
        return output_dir, False

    run_root = PROJECT_ROOT / "logs" / "xgb_nested_cv"
    if args.resume and run_root.is_dir():
        candidates = sorted(
            (path for path in run_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            manifest_path = candidate / "run_manifest.json"
            checkpoint_path = candidate / "checkpoint_results.jsonl"
            state_path = candidate / "checkpoint_state.json"
            if not manifest_path.is_file() or not checkpoint_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("run_signature") != run_signature:
                continue
            if _checkpoint_is_complete(checkpoint_path, state_path):
                continue
            return candidate.resolve(), True

    if args.resume:
        logging.getLogger("xgboost_nested_cv").info(
            "No incomplete matching run found; starting a new run."
        )
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return run_root / run_stamp, False


def _append_checkpoint_event(
    checkpoint_path: Path,
    state_path: Path,
    event: dict[str, Any],
) -> None:
    """Durably append one completed result and atomically update run progress."""
    event = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with checkpoint_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())

    state = {
        "last_recorded_at_utc": event["recorded_at_utc"],
        "last_event_type": event["event_type"],
        "last_outer_fold": event.get("outer_fold"),
        "last_inner_fold": event.get("inner_fold"),
        "last_dataset_variant": event.get("dataset_variant"),
        "last_config_id": event.get("config_id"),
        "completed_inner_fits": event.get("completed_inner_fits"),
    }
    temporary_path = state_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with temporary_path.open("rb") as file:
        os.fsync(file.fileno())
    temporary_path.replace(state_path)


def _render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(str(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    statistical_results: dict[str, Any],
) -> None:
    table_rows = [
        {
            "Dataset variant": row["dataset_label"],
            "Accuracy": f'{row["accuracy"]:.4f}',
            "Macro precision": f'{row["macro_precision"]:.4f}',
            "Macro recall": f'{row["macro_recall"]:.4f}',
            "Macro F1": f'{row["macro_f1"]:.4f}',
            "Outer-fold macro-F1 mean (SD)": (
                f'{row["outer_fold_macro_f1_mean"]:.4f} '
                f'({row["outer_fold_macro_f1_sd"]:.4f})'
            ),
        }
        for row in summary_rows
        if row["dataset_variant"] != "nested_selected"
    ]
    selected_variant_counts: dict[str, int] = defaultdict(int)
    for row in selected_rows:
        if row["selection_scope"] == "overall":
            selected_variant_counts[row["dataset_label"]] += 1

    class_table_rows = [
        {
            "Dataset variant": row["dataset_label"],
            "Activity": row["class"],
            "Precision": f'{row["precision"]:.3f}',
            "Recall": f'{row["recall"]:.3f}',
            "F1": f'{row["f1"]:.3f}',
            "Support": row["support"],
        }
        for row in class_rows
    ]
    comparison_rows = [
        {
            "Comparison": (
                f'{next(label for key, label, _ in VARIANTS if key == row["left"])} '
                f'vs {next(label for key, label, _ in VARIANTS if key == row["right"])}'
            ),
            "Mean subject-F1 difference": f'{row["mean_subject_f1_difference"]:.4f}',
            "Wilcoxon p": f'{row["p_value"]:.4g}',
            "Holm-adjusted p": f'{row["p_value_holm_adjusted"]:.4g}',
        }
        for row in statistical_results["pairwise"]
    ]
    chosen_lines = [
        f"- {variant}: selected in {count}/{metadata['outer_folds']} outer folds"
        for variant, count in selected_variant_counts.items()
    ]
    if not chosen_lines:
        chosen_lines = ["- No variant-selection records were produced."]
    selected_config_table = [
        {
            "Outer fold": row["outer_fold"],
            "Selected variant": row["dataset_label"],
            "Config ID": row["config_id"],
            "Inner macro-F1": f'{row["inner_macro_f1"]:.4f}',
            "Parameters differing from fixed defaults": json.dumps(
                {
                    key: value
                    for key, value in json.loads(row["parameters_json"]).items()
                    if key not in XGB_CONFIG_DEFAULTS
                    or value != XGB_CONFIG_DEFAULTS[key]
                },
                sort_keys=True,
            ),
        }
        for row in selected_rows
        if row["selection_scope"] == "overall"
    ]

    text = f"""# XGBoost nested subject-grouped CV run

## Run and data

- Run ID: `{metadata['run_id']}`
- Dataset source: `{metadata['data_file']}`
- SHA-256: `{metadata['data_sha256']}`
- Hyperparameter log(s): {', '.join(f'`{path}`' for path in metadata['parameter_logs'])}
- Unique configurations after duplicate removal: {metadata['unique_configurations']}
- Outer/inner folds: {metadata['outer_folds']}/{metadata['inner_folds']}
- Random seed: {metadata['seed']}
- Subjects / rows / input features: {metadata['subject_count']} / {metadata['row_count']} / {metadata['input_feature_count']}
- XGBoost / scikit-learn / pandas / SciPy: {metadata['versions']['xgboost']} / {metadata['versions']['scikit_learn']} / {metadata['versions']['pandas']} / {metadata['versions']['scipy']}

## Evaluation protocol

The script uses nested stratified subject-grouped cross-validation. In each outer fold,
all rows for the outer test subjects are withheld from preprocessing, parameter
selection, and fitting. Inner grouped folds select the hyperparameter configuration
using pooled out-of-fold macro-F1. For each dataset variant, the winning inner
configuration is refit on the complete outer training subjects and evaluated on the
outer test subjects. The inner-selected best dataset-variant/configuration pair is
also recorded and evaluated as a separate nested selection procedure.

Preprocessing is fitted from scratch inside every inner and outer training fold:

- Dataset 1: duplicate-column removal (stages 0-1).
- Dataset 2: duplicate-column removal, fixed frequency-feature pruning, variance
  filtering at 0.01, and correlation filtering above 0.95 (stages 0-4).
- Dataset 3: Dataset 2 steps, fold-fitted Yeo-Johnson transform, then PCA retaining
  95% of variance (stages 0-6).

Inner validation folds select parameters; outer folds estimate performance. This run
does not use a separate final holdout test set. **The candidate configurations come
from historical experiments, and prior manual tuning may have been influenced by
previous results on the same subjects. Nested CV prevents this script from tuning on
each outer fold, but cannot undo that history. Treat these results as an exploratory
re-evaluation, not as an untouched confirmatory test.**

## Out-of-fold comparison

The table reports pooled outer-fold predictions. Fold mean and SD describe variation
across the {metadata['outer_folds']} held-out-subject folds.

{_render_markdown_table(table_rows, ['Dataset variant', 'Accuracy', 'Macro precision', 'Macro recall', 'Macro F1', 'Outer-fold macro-F1 mean (SD)'])}

The overall inner-selected procedure chose the following variants:

{chr(10).join(chosen_lines)}

For each outer fold, the configuration below was selected because it had the highest
pooled inner out-of-fold macro-F1 across all candidate variant/configuration pairs.
Ties retain the first configuration encountered in the input logs and variant list.
The parameters shown differ from the fixed XGBoost defaults; complete effective
settings and original log records are preserved in `run_metadata.json` and
`selected_configurations.csv`.

{_render_markdown_table(selected_config_table, ['Outer fold', 'Selected variant', 'Config ID', 'Inner macro-F1', 'Parameters differing from fixed defaults'])}

Its pooled outer-fold metrics are recorded in `summary_metrics.csv` under
`nested_selected`. Configuration-level inner validation scores are in
`inner_candidate_scores.csv`; selected settings per outer fold are in
`selected_configurations.csv`.

## Per-activity results

Precision, recall, and F1 below are calculated from the pooled outer-fold predictions.
This view helps identify activity-specific weaknesses that a single aggregate score
can hide.

{_render_markdown_table(class_table_rows, ['Dataset variant', 'Activity', 'Precision', 'Recall', 'F1', 'Support'])}

## Paired statistical comparison

The omnibus test is a Friedman test on one out-of-fold macro-F1 per subject. Pairwise
comparisons use Wilcoxon signed-rank tests with Holm correction.

- Friedman statistic: {statistical_results['omnibus_test']['statistic']:.4f}
- Friedman p-value: {statistical_results['omnibus_test']['p_value']:.4g}

{_render_markdown_table(comparison_rows, ['Comparison', 'Mean subject-F1 difference', 'Wilcoxon p', 'Holm-adjusted p'])}

Interpret cautiously: there are only 30 subjects, models from different outer folds
share training subjects, and the historical search process may have seen prior scores.
The statistical test is subject-level, not a test over the much larger number of
overlapping windows.

## Reproducibility and files

The run metadata, exact parameter sources, fold subject assignments, class-level
metrics, confusion matrices, and predictions are saved beside this report. Use the
same script, input CSV, JSONL logs, seed, and fold settings to reproduce the run.
See `run_metadata.json` for file hashes and package versions; `run.log` records
progress and fold assignments.
"""
    (output_dir / "REPORT.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    log_paths = [Path(path).expanduser().resolve() for path in args.hyperparameter_log]
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    configurations, config_sources = load_unique_configurations(log_paths)
    if args.max_configs is not None:
        if args.max_configs < 1:
            raise ValueError("--max-configs must be at least 1.")
        configurations = configurations[: args.max_configs]
        config_sources = config_sources[: args.max_configs]

    data = pd.read_csv(data_path)
    required = {"Activity", "subject"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    X = data.drop(columns=["Activity", "subject", "split"], errors="ignore")
    non_numeric = list(X.select_dtypes(exclude=[np.number]).columns)
    if non_numeric:
        raise ValueError(f"Non-numeric input feature columns found: {non_numeric}")
    if X.empty or X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Features must be non-empty and contain only finite values.")
    if data["Activity"].isna().any() or data["subject"].isna().any():
        raise ValueError("Activity and subject columns must not contain missing values.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["Activity"].astype(str))
    groups = data["subject"].to_numpy()
    subjects = np.unique(groups)
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("Both outer and inner fold counts must be at least 2.")
    if args.outer_folds > len(subjects):
        raise ValueError(
            f"{args.outer_folds} outer folds exceed the {len(subjects)} subjects."
        )
    if args.inner_folds > len(subjects) - math.ceil(len(subjects) / args.outer_folds):
        raise ValueError(
            "The inner fold count exceeds the smallest expected outer-training "
            "subject count."
        )

    signature_payload = {
        "script_sha256": _sha256(Path(__file__).resolve()),
        "data_file": str(data_path),
        "data_sha256": _sha256(data_path),
        "parameter_logs": [
            {"path": str(path), "sha256": _sha256(path)} for path in log_paths
        ],
        "configurations": configurations,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "dataset_variants": VARIANTS,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "scipy": scipy.__version__,
        },
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()
    output_dir, is_resuming = _resolve_output_directory(args, run_signature)
    logger = _configure_logging(output_dir, resume=is_resuming)

    logger.info(
        "%s nested subject-grouped XGBoost evaluation.",
        "Resuming" if is_resuming else "Starting",
    )
    logger.info("Dataset: %s (%d rows, %d features).", data_path, len(X), X.shape[1])
    logger.info(
        "Subjects: %d; activities: %s.",
        len(subjects),
        list(label_encoder.classes_),
    )
    logger.info(
        "Unique configurations: %d; skipped duplicate records: %d.",
        len(configurations),
        sum(source["duplicate_records_skipped"] for source in config_sources),
    )
    logger.info(
        "Nested search size: up to %d inner fits + %d outer fits.",
        args.outer_folds
        * args.inner_folds
        * len(configurations)
        * len(VARIANTS),
        args.outer_folds * len(VARIANTS),
    )

    n_classes = len(label_encoder.classes_)
    variant_predictions = {
        key: np.full(len(y), -1, dtype=int) for key, _, _ in VARIANTS
    }
    nested_predictions = np.full(len(y), -1, dtype=int)
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    fold_assignments: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    total_fit_count = 0

    outer_splitter = StratifiedGroupKFold(
        n_splits=args.outer_folds, shuffle=True, random_state=args.seed
    )
    outer_splits = list(outer_splitter.split(X, y, groups))
    planned_folds: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, start=1):
        outer_train_groups = groups[outer_train]
        outer_test_groups = groups[outer_test]
        overlap = set(outer_train_groups) & set(outer_test_groups)
        if overlap:
            raise RuntimeError(f"Subject leakage in outer fold {outer_fold}: {overlap}")
        inner_splitter = StratifiedGroupKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.seed + outer_fold,
        )
        inner_splits = list(
            inner_splitter.split(X.iloc[outer_train], y[outer_train], outer_train_groups)
        )
        inner_subjects = []
        for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits, start=1):
            train_subjects = np.unique(outer_train_groups[inner_train]).tolist()
            valid_subjects = np.unique(outer_train_groups[inner_valid]).tolist()
            if set(train_subjects) & set(valid_subjects):
                raise RuntimeError(
                    f"Subject leakage in outer fold {outer_fold}, "
                    f"inner fold {inner_fold}."
                )
            inner_subjects.append(
                {
                    "inner_fold": inner_fold,
                    "train_subjects": train_subjects,
                    "validation_subjects": valid_subjects,
                }
            )
        planned_folds.append(
            {
                "outer_fold": outer_fold,
                "train_subjects": np.unique(outer_train_groups).tolist(),
                "test_subjects": np.unique(outer_test_groups).tolist(),
                "inner_folds": inner_subjects,
            }
        )

    checkpoint_path = output_dir / "checkpoint_results.jsonl"
    checkpoint_state_path = output_dir / "checkpoint_state.json"
    run_id = output_dir.name
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "run_id": run_id,
        "run_signature": run_signature,
        "signature_payload": signature_payload,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "script": str(Path(__file__).resolve()),
        "data_file": str(data_path),
        "data_sha256": _sha256(data_path),
        "parameter_logs": [str(path) for path in log_paths],
        "parameter_log_sha256": {str(path): _sha256(path) for path in log_paths},
        "configuration_sources": config_sources,
        "unique_configurations": len(configurations),
        "duplicate_records_skipped": sum(
            source["duplicate_records_skipped"] for source in config_sources
        ),
        "dataset_variants": [
            {
                "key": key,
                "label": label,
                "preprocessing_stage": stage,
            }
            for key, label, stage in VARIANTS
        ],
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "selection_metric": "pooled inner out-of-fold macro-F1",
        "row_count": len(data),
        "input_feature_count": X.shape[1],
        "subject_count": len(subjects),
        "subject_ids": sorted(subjects.tolist()),
        "activity_labels": list(label_encoder.classes_),
        "fold_subject_assignments": planned_folds,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "scipy": scipy.__version__,
        },
        "git": _git_provenance(),
        "caveat": (
            "Historical hyperparameter tuning may have used prior results from "
            "these subjects. Nested CV does not remove that experiment-history bias."
        ),
    }
    if is_resuming:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("fold_subject_assignments") != planned_folds:
            raise ValueError(
                f"Cannot resume {output_dir}: deterministic fold assignments changed."
            )
    else:
        _save_json(manifest_path, manifest)

    checkpoint_events = _read_checkpoint_events(checkpoint_path)
    inner_fit_events: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    outer_evaluation_events: dict[tuple[int, str], dict[str, Any]] = {}
    candidate_event_keys: set[tuple[int, str, str]] = set()
    selection_event_folds: set[int] = set()
    for event in checkpoint_events:
        event_type = event["event_type"]
        if event_type == "inner_fit_completed":
            key = (
                int(event["outer_fold"]),
                str(event["dataset_variant"]),
                int(event["inner_fold"]),
                str(event["config_id"]),
            )
            inner_fit_events[key] = event
        elif event_type == "outer_evaluation_completed":
            key = (int(event["outer_fold"]), str(event["dataset_variant"]))
            outer_evaluation_events[key] = event
        elif event_type == "inner_candidate_completed":
            candidate_event_keys.add(
                (
                    int(event["outer_fold"]),
                    str(event["dataset_variant"]),
                    str(event["config_id"]),
                )
            )
        elif event_type == "outer_fold_selection_completed":
            selection_event_folds.add(int(event["outer_fold"]))

    planned_inner_fit_count = (
        args.outer_folds
        * args.inner_folds
        * len(configurations)
        * len(VARIANTS)
    )
    completed_inner_fit_count = len(inner_fit_events)
    total_fit_count = len(inner_fit_events) + len(outer_evaluation_events)
    if not is_resuming:
        _append_checkpoint_event(
            checkpoint_path,
            checkpoint_state_path,
            {
                "event_type": "run_started",
                "run_id": run_id,
                "planned_inner_fits": planned_inner_fit_count,
                "planned_outer_fits": args.outer_folds * len(VARIANTS),
            },
        )
    else:
        logger.info(
            "Restored %d inner fits and %d outer evaluations from checkpoints.",
            len(inner_fit_events),
            len(outer_evaluation_events),
        )
        _append_checkpoint_event(
            checkpoint_path,
            checkpoint_state_path,
            {
                "event_type": "run_resumed",
                "run_id": run_id,
                "restored_inner_fits": len(inner_fit_events),
                "restored_outer_evaluations": len(outer_evaluation_events),
            },
        )
    overall_start = time.monotonic()
    for outer_fold, (outer_train, outer_test) in enumerate(outer_splits, start=1):
        outer_train_groups = groups[outer_train]
        outer_test_groups = groups[outer_test]
        overlap = set(outer_train_groups) & set(outer_test_groups)
        if overlap:
            raise RuntimeError(f"Subject leakage in outer fold {outer_fold}: {overlap}")
        logger.info(
            "Outer fold %d/%d: train subjects=%s; test subjects=%s.",
            outer_fold,
            args.outer_folds,
            sorted(np.unique(outer_train_groups).tolist()),
            sorted(np.unique(outer_test_groups).tolist()),
        )
        fold_assignments.append(
            {
                "outer_fold": outer_fold,
                "train_subjects": sorted(np.unique(outer_train_groups).tolist()),
                "test_subjects": sorted(np.unique(outer_test_groups).tolist()),
            }
        )

        X_outer = X.iloc[outer_train]
        y_outer = y[outer_train]
        groups_outer = groups[outer_train]
        inner_splitter = StratifiedGroupKFold(
            n_splits=args.inner_folds,
            shuffle=True,
            random_state=args.seed + outer_fold,
        )
        inner_splits = list(
            inner_splitter.split(X_outer, y_outer, groups_outer)
        )
        for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits, start=1):
            if set(groups_outer[inner_train]) & set(groups_outer[inner_valid]):
                raise RuntimeError(
                    f"Subject leakage in outer fold {outer_fold}, "
                    f"inner fold {inner_fold}."
                )

        scored_candidates: list[dict[str, Any]] = []
        for variant_key, variant_label, stage in VARIANTS:
            inner_oof_predictions: dict[int, np.ndarray] = {
                config_index: np.full(len(y_outer), -1, dtype=int)
                for config_index in range(len(configurations))
            }
            for inner_fold, (inner_train, inner_valid) in enumerate(
                inner_splits, start=1
            ):
                restored_for_fold = {
                    config_index: inner_fit_events.get(
                        (
                            outer_fold,
                            variant_key,
                            inner_fold,
                            config_sources[config_index]["config_id"],
                        )
                    )
                    for config_index in range(len(configurations))
                }
                missing_config_indices = [
                    index
                    for index, event in restored_for_fold.items()
                    if event is None
                ]
                if missing_config_indices:
                    preprocessor = build_preprocessor(stage)
                    X_inner_train = preprocessor.fit_transform(
                        X_outer.iloc[inner_train]
                    )
                    X_inner_valid = preprocessor.transform(X_outer.iloc[inner_valid])
                    y_inner_train = y_outer[inner_train]
                for config_index, parameters in enumerate(configurations):
                    config_id = config_sources[config_index]["config_id"]
                    previous_event = restored_for_fold[config_index]
                    if previous_event is not None:
                        expected_indices = outer_train[inner_valid]
                        inner_predictions = _restore_prediction_event(
                            previous_event,
                            expected_indices,
                            X.index,
                            label_encoder,
                            y[expected_indices],
                        )
                        inner_oof_predictions[config_index][inner_valid] = (
                            inner_predictions
                        )
                        continue
                    fit_started = time.monotonic()
                    logger.info(
                        "Starting inner fit %d/%d: outer=%d/%d, variant=%s, "
                        "inner=%d/%d, config=%s, n_estimators=%s.",
                        completed_inner_fit_count + 1,
                        planned_inner_fit_count,
                        outer_fold,
                        args.outer_folds,
                        variant_key,
                        inner_fold,
                        args.inner_folds,
                        config_id,
                        parameters.get("n_estimators", XGB_CONFIG_DEFAULTS["n_estimators"]),
                    )
                    model = make_model(parameters, args.seed)
                    model.fit(X_inner_train, y_inner_train)
                    inner_predictions = model.predict(X_inner_valid)
                    inner_oof_predictions[config_index][inner_valid] = inner_predictions
                    total_fit_count += 1
                    completed_inner_fit_count += 1
                    inner_fold_metrics = metrics(
                        y_outer[inner_valid], inner_predictions, n_classes
                    )
                    inner_confusion = confusion_matrix(
                        y_outer[inner_valid],
                        inner_predictions,
                        labels=np.arange(n_classes),
                    )
                    _append_checkpoint_event(
                        checkpoint_path,
                        checkpoint_state_path,
                        {
                            "event_type": "inner_fit_completed",
                            "completed_inner_fits": completed_inner_fit_count,
                            "planned_inner_fits": planned_inner_fit_count,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "dataset_variant": variant_key,
                            "dataset_label": variant_label,
                            "config_id": config_id,
                            "parameters": parameters,
                            "fit_elapsed_seconds": time.monotonic() - fit_started,
                            "train_subjects": np.unique(
                                groups_outer[inner_train]
                            ).tolist(),
                            "validation_subjects": np.unique(
                                groups_outer[inner_valid]
                            ).tolist(),
                            "validation_metrics": inner_fold_metrics,
                            "validation_class_metrics": per_class_metrics(
                                y_outer[inner_valid],
                                inner_predictions,
                                list(label_encoder.classes_),
                            ),
                            "confusion_matrix_counts": inner_confusion.tolist(),
                            "predictions": _prediction_payload(
                                X.index[outer_train[inner_valid]],
                                groups[outer_train[inner_valid]],
                                y[outer_train[inner_valid]],
                                inner_predictions,
                                list(label_encoder.classes_),
                            ),
                        },
                    )
                    logger.info(
                        "Completed inner fit %d/%d: outer=%d/%d, variant=%s, "
                        "inner=%d/%d, config=%s, elapsed=%.1fs.",
                        completed_inner_fit_count,
                        planned_inner_fit_count,
                        outer_fold,
                        args.outer_folds,
                        variant_key,
                        inner_fold,
                        args.inner_folds,
                        config_id,
                        time.monotonic() - fit_started,
                    )

            for config_index, parameters in enumerate(configurations):
                predictions = inner_oof_predictions[config_index]
                if np.any(predictions < 0):
                    raise RuntimeError("An inner validation sample has no prediction.")
                candidate_metrics = metrics(y_outer, predictions, n_classes)
                config_id = config_sources[config_index]["config_id"]
                result = {
                    "outer_fold": outer_fold,
                    "dataset_variant": variant_key,
                    "dataset_label": variant_label,
                    "config_id": config_id,
                    "parameters": parameters,
                    **candidate_metrics,
                }
                scored_candidates.append(result)
                candidate_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "dataset_variant": variant_key,
                        "dataset_label": variant_label,
                        "config_id": config_id,
                        "inner_macro_f1": candidate_metrics["macro_f1"],
                        "inner_accuracy": candidate_metrics["accuracy"],
                        "inner_macro_precision": candidate_metrics["macro_precision"],
                        "inner_macro_recall": candidate_metrics["macro_recall"],
                        "parameters_json": json.dumps(parameters, sort_keys=True),
                        "inner_folds": args.inner_folds,
                    }
                )
                inner_confusion = confusion_matrix(
                    y_outer, predictions, labels=np.arange(n_classes)
                )
                candidate_event_key = (outer_fold, variant_key, config_id)
                if candidate_event_key not in candidate_event_keys:
                    _append_checkpoint_event(
                        checkpoint_path,
                        checkpoint_state_path,
                        {
                            "event_type": "inner_candidate_completed",
                            "completed_inner_fits": completed_inner_fit_count,
                            "outer_fold": outer_fold,
                            "dataset_variant": variant_key,
                            "dataset_label": variant_label,
                            "config_id": config_id,
                            "parameters": parameters,
                            "inner_folds": args.inner_folds,
                            "pooled_inner_metrics": candidate_metrics,
                            "pooled_inner_class_metrics": per_class_metrics(
                                y_outer, predictions, list(label_encoder.classes_)
                            ),
                            "pooled_inner_confusion_matrix_counts": (
                                inner_confusion.tolist()
                            ),
                        },
                    )
                    candidate_event_keys.add(candidate_event_key)
                logger.info(
                    "Outer %d, %s, %s inner pooled macro-F1=%.5f.",
                    outer_fold,
                    variant_key,
                    config_id,
                    candidate_metrics["macro_f1"],
                )

        best_by_variant: dict[str, dict[str, Any]] = {}
        for variant_key, _, _ in VARIANTS:
            variant_candidates = [
                candidate
                for candidate in scored_candidates
                if candidate["dataset_variant"] == variant_key
            ]
            best_by_variant[variant_key] = max(
                variant_candidates, key=lambda candidate: candidate["macro_f1"]
            )
        best_overall = max(
            scored_candidates, key=lambda candidate: candidate["macro_f1"]
        )
        if outer_fold not in selection_event_folds:
            _append_checkpoint_event(
                checkpoint_path,
                checkpoint_state_path,
                {
                    "event_type": "outer_fold_selection_completed",
                    "outer_fold": outer_fold,
                    "selected_per_variant": {
                        variant_key: {
                            "config_id": best["config_id"],
                            "parameters": best["parameters"],
                            "inner_macro_f1": best["macro_f1"],
                        }
                        for variant_key, best in best_by_variant.items()
                    },
                    "selected_overall": {
                        "dataset_variant": best_overall["dataset_variant"],
                        "dataset_label": best_overall["dataset_label"],
                        "config_id": best_overall["config_id"],
                        "parameters": best_overall["parameters"],
                        "inner_macro_f1": best_overall["macro_f1"],
                    },
                },
            )
            selection_event_folds.add(outer_fold)

        selected_rows.extend(
            {
                "outer_fold": outer_fold,
                "selection_scope": "per_variant",
                "dataset_variant": variant_key,
                "dataset_label": best["dataset_label"],
                "config_id": best["config_id"],
                "inner_macro_f1": best["macro_f1"],
                "parameters_json": json.dumps(best["parameters"], sort_keys=True),
            }
            for variant_key, best in best_by_variant.items()
        )
        selected_rows.append(
            {
                "outer_fold": outer_fold,
                "selection_scope": "overall",
                "dataset_variant": best_overall["dataset_variant"],
                "dataset_label": best_overall["dataset_label"],
                "config_id": best_overall["config_id"],
                "inner_macro_f1": best_overall["macro_f1"],
                "parameters_json": json.dumps(
                    best_overall["parameters"], sort_keys=True
                ),
            }
        )

        for variant_key, variant_label, stage in VARIANTS:
            best = best_by_variant[variant_key]
            outer_event = outer_evaluation_events.get((outer_fold, variant_key))
            if outer_event is not None:
                if outer_event["config_id"] != best["config_id"]:
                    raise ValueError(
                        "Checkpoint selected configuration does not match the "
                        "recomputed inner-CV selection."
                    )
                predictions = _restore_prediction_event(
                    outer_event,
                    outer_test,
                    X.index,
                    label_encoder,
                    y[outer_test],
                )
                features_after_preprocessing = int(
                    outer_event["features_after_preprocessing"]
                )
                fold_scores = outer_event["outer_metrics"]
                logger.info(
                    "Restored outer %d, %s, selected %s from checkpoint.",
                    outer_fold,
                    variant_key,
                    best["config_id"],
                )
            else:
                config_index = next(
                    index
                    for index, source in enumerate(config_sources)
                    if source["config_id"] == best["config_id"]
                )
                preprocessor = build_preprocessor(stage)
                X_outer_train = preprocessor.fit_transform(X.iloc[outer_train])
                X_outer_test = preprocessor.transform(X.iloc[outer_test])
                model = make_model(configurations[config_index], args.seed)
                model.fit(X_outer_train, y[outer_train])
                predictions = model.predict(X_outer_test)
                total_fit_count += 1
                features_after_preprocessing = int(X_outer_train.shape[1])
                fold_scores = metrics(y[outer_test], predictions, n_classes)
            variant_predictions[variant_key][outer_test] = predictions

            fold_metric_rows.append(
                {
                    "outer_fold": outer_fold,
                    "dataset_variant": variant_key,
                    "dataset_label": variant_label,
                    "config_id": best["config_id"],
                    "parameters_json": json.dumps(best["parameters"], sort_keys=True),
                    "selected_inner_macro_f1": best["macro_f1"],
                    "features_after_preprocessing": features_after_preprocessing,
                    **fold_scores,
                }
            )
            if outer_event is None:
                outer_confusion = confusion_matrix(
                    y[outer_test], predictions, labels=np.arange(n_classes)
                )
                _append_checkpoint_event(
                    checkpoint_path,
                    checkpoint_state_path,
                    {
                        "event_type": "outer_evaluation_completed",
                        "completed_inner_fits": completed_inner_fit_count,
                        "outer_fold": outer_fold,
                        "dataset_variant": variant_key,
                        "dataset_label": variant_label,
                        "config_id": best["config_id"],
                        "parameters": best["parameters"],
                        "selected_inner_macro_f1": best["macro_f1"],
                        "features_after_preprocessing": (
                            features_after_preprocessing
                        ),
                        "outer_metrics": fold_scores,
                        "outer_class_metrics": per_class_metrics(
                            y[outer_test], predictions, list(label_encoder.classes_)
                        ),
                        "confusion_matrix_counts": outer_confusion.tolist(),
                        "predictions": _prediction_payload(
                            X.index[outer_test],
                            groups[outer_test],
                            y[outer_test],
                            predictions,
                            list(label_encoder.classes_),
                        ),
                    },
                )
                outer_evaluation_events[(outer_fold, variant_key)] = {
                    "event_type": "outer_evaluation_completed",
                    "outer_metrics": fold_scores,
                    "features_after_preprocessing": features_after_preprocessing,
                }
                logger.info(
                    "Outer %d, %s, selected %s, test macro-F1=%.5f (%d features).",
                    outer_fold,
                    variant_key,
                    best["config_id"],
                    fold_scores["macro_f1"],
                    features_after_preprocessing,
                )

            for local_index, global_index in enumerate(outer_test):
                oof_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "subject": groups[global_index],
                        "activity_true": label_encoder.classes_[y[global_index]],
                        "dataset_variant": variant_key,
                        "dataset_label": variant_label,
                        "activity_predicted": label_encoder.classes_[
                            predictions[local_index]
                        ],
                        "config_id": best["config_id"],
                        "selected_overall": (
                            best_overall["dataset_variant"] == variant_key
                        ),
                    }
                )

        selected_variant = best_overall["dataset_variant"]
        nested_predictions[outer_test] = variant_predictions[selected_variant][
            outer_test
        ]
        for local_index, global_index in enumerate(outer_test):
            oof_rows.append(
                {
                    "outer_fold": outer_fold,
                    "subject": groups[global_index],
                    "activity_true": label_encoder.classes_[y[global_index]],
                    "dataset_variant": "nested_selected",
                    "dataset_label": "Inner-selected variant + configuration",
                    "activity_predicted": label_encoder.classes_[
                        nested_predictions[global_index]
                    ],
                    "config_id": best_overall["config_id"],
                    "selected_overall": True,
                }
            )
        nested_fold_scores = metrics(
            y[outer_test], nested_predictions[outer_test], n_classes
        )
        fold_metric_rows.append(
            {
                "outer_fold": outer_fold,
                "dataset_variant": "nested_selected",
                "dataset_label": "Inner-selected variant + configuration",
                "config_id": best_overall["config_id"],
                "parameters_json": json.dumps(
                    best_overall["parameters"], sort_keys=True
                ),
                "selected_inner_macro_f1": best_overall["macro_f1"],
                "features_after_preprocessing": None,
                **nested_fold_scores,
            }
        )
        logger.info(
            "Outer %d, inner-selected pipeline %s/%s, test macro-F1=%.5f.",
            outer_fold,
            selected_variant,
            best_overall["config_id"],
            nested_fold_scores["macro_f1"],
        )

    if any(np.any(predictions < 0) for predictions in variant_predictions.values()):
        raise RuntimeError("Some observations did not receive an outer-fold prediction.")
    if np.any(nested_predictions < 0):
        raise RuntimeError("Some observations did not receive a nested prediction.")

    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, Any] = {}
    for variant_key, variant_label, _ in VARIANTS:
        variant_prediction = variant_predictions[variant_key]
        pooled = metrics(y, variant_prediction, n_classes)
        fold_f1 = [
            row["macro_f1"]
            for row in fold_metric_rows
            if row["dataset_variant"] == variant_key
        ]
        summary_rows.append(
            {
                "dataset_variant": variant_key,
                "dataset_label": variant_label,
                **pooled,
                "outer_fold_macro_f1_mean": float(np.mean(fold_f1)),
                "outer_fold_macro_f1_sd": float(np.std(fold_f1, ddof=1)),
                "subjects": len(subjects),
                "rows": len(y),
            }
        )
        class_rows.extend(
            {
                "dataset_variant": variant_key,
                "dataset_label": variant_label,
                **row,
            }
            for row in per_class_metrics(y, variant_prediction, list(label_encoder.classes_))
        )
        matrix = confusion_matrix(y, variant_prediction, labels=np.arange(n_classes))
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        confusion_matrices[variant_key] = {
            "dataset_label": variant_label,
            "labels": list(label_encoder.classes_),
            "counts": matrix.tolist(),
            "row_normalized": normalized.tolist(),
        }

    nested_metrics = metrics(y, nested_predictions, n_classes)
    nested_fold_f1 = [
        row["macro_f1"]
        for row in fold_metric_rows
        if row["dataset_variant"] == "nested_selected"
    ]
    summary_rows.append(
        {
            "dataset_variant": "nested_selected",
            "dataset_label": "Inner-selected variant + configuration",
            **nested_metrics,
            "outer_fold_macro_f1_mean": float(np.mean(nested_fold_f1)),
            "outer_fold_macro_f1_sd": float(np.std(nested_fold_f1, ddof=1)),
            "subjects": len(subjects),
            "rows": len(y),
        }
    )
    class_rows.extend(
        {
            "dataset_variant": "nested_selected",
            "dataset_label": "Inner-selected variant + configuration",
            **row,
        }
        for row in per_class_metrics(y, nested_predictions, list(label_encoder.classes_))
    )
    matrix = confusion_matrix(y, nested_predictions, labels=np.arange(n_classes))
    row_totals = matrix.sum(axis=1, keepdims=True)
    confusion_matrices["nested_selected"] = {
        "dataset_label": "Inner-selected variant + configuration",
        "labels": list(label_encoder.classes_),
        "counts": matrix.tolist(),
        "row_normalized": np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        ).tolist(),
    }

    statistical_results = statistical_comparisons(
        variant_predictions, y, groups, n_classes
    )
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = output_dir.name
    metadata = {
        "run_id": run_id,
        "run_signature": run_signature,
        "resumed": is_resuming,
        "restored_inner_fit_count": len(inner_fit_events),
        "restored_outer_evaluation_count": len(outer_evaluation_events),
        "created_at_utc": created_at,
        "script": str(Path(__file__).resolve()),
        "command": sys.argv,
        "data_file": str(data_path),
        "data_sha256": _sha256(data_path),
        "parameter_logs": [str(path) for path in log_paths],
        "parameter_log_sha256": {
            str(path): _sha256(path) for path in log_paths
        },
        "unique_configurations": len(configurations),
        "configuration_sources": config_sources,
        "duplicate_records_skipped": sum(
            source["duplicate_records_skipped"] for source in config_sources
        ),
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "seed": args.seed,
        "selection_metric": "pooled inner out-of-fold macro-F1",
        "outer_evaluation": "pooled out-of-fold predictions on held-out subjects",
        "final_holdout_test_set_used": False,
        "row_count": len(data),
        "input_feature_count": X.shape[1],
        "subject_count": len(subjects),
        "subject_ids": sorted(subjects.tolist()),
        "activity_labels": list(label_encoder.classes_),
        "fit_count": total_fit_count,
        "elapsed_seconds": time.monotonic() - overall_start,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "scipy": scipy.__version__,
        },
        "git": _git_provenance(),
        "caveat": (
            "Historical hyperparameter tuning may have used prior results from "
            "these subjects. Nested CV does not remove that experiment-history bias."
        ),
    }

    pd.DataFrame(candidate_rows).to_csv(
        output_dir / "inner_candidate_scores.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        output_dir / "selected_configurations.csv", index=False
    )
    pd.DataFrame(fold_metric_rows).to_csv(
        output_dir / "outer_fold_metrics.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "class_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(output_dir / "oof_predictions.csv", index=False)
    _save_json(output_dir / "confusion_matrices.json", confusion_matrices)
    _save_json(output_dir / "statistical_tests.json", statistical_results)
    _save_json(output_dir / "fold_assignments.json", fold_assignments)
    _save_json(output_dir / "run_metadata.json", metadata)
    write_report(
        output_dir,
        metadata,
        summary_rows,
        class_rows,
        selected_rows,
        statistical_results,
    )
    _append_checkpoint_event(
        checkpoint_path,
        checkpoint_state_path,
        {
            "event_type": "run_completed",
            "run_id": run_id,
            "fit_count": total_fit_count,
            "completed_inner_fits": completed_inner_fit_count,
        },
    )
    logger.info("Completed %d model fits in %.1f seconds.", total_fit_count, metadata["elapsed_seconds"])
    logger.info("Summary report: %s", output_dir / "REPORT.md")
    logger.info("Results directory: %s", output_dir)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate historical XGBoost configurations with nested, "
            "subject-grouped CV and fold-fitted feature preprocessing."
        )
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Raw feature CSV.")
    parser.add_argument(
        "--hyperparameter-log",
        action="append",
        dest="hyperparameter_log",
        help=(
            "JSONL file containing hyperparameters/parameters objects; may be "
            "specified multiple times. Defaults to dataset/experiment_logs.jsonl."
        ),
    )
    parser.add_argument(
        "--outer-folds", type=int, default=5, help="Number of held-out subject folds."
    )
    parser.add_argument(
        "--inner-folds", type=int, default=4, help="Number of grouped selection folds."
    )
    parser.add_argument("--seed", type=int, default=69, help="Reproducibility seed.")
    parser.add_argument(
        "--max-configs",
        type=int,
        help="Limit configurations for a smoke run; do not use for final reporting.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Results directory to resume, or a new directory for a fresh run. "
            "By default resumes the newest matching incomplete run under logs/."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a compatible incomplete run when available (default: enabled).",
    )
    args = parser.parse_args(argv)
    if args.hyperparameter_log is None:
        args.hyperparameter_log = [str(DEFAULT_PARAMETER_LOG)]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except Exception:
        logging.getLogger("xgboost_nested_cv").exception("Experiment failed.")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
