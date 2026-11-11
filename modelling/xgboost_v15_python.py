from datetime import datetime
import json
import os
import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def load_unique_hyperparameters(jsonl_file_path: str) -> list[dict]:
    """Scans a JSONL file line-by-line and extracts unique hyperparameter configurations."""
    unique_params = []
    seen_configs = set()

    with open(jsonl_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            params = record.get("hyperparameters", {})

            # Serialize dict with sorted keys to make it hashable for set lookup
            config_key = json.dumps(params, sort_keys=True)

            if config_key not in seen_configs:
                seen_configs.add(config_key)
                unique_params.append(params)

    return unique_params


def compute_metrics(y_true, y_pred, class_names):
    """Calculates accuracy, precision, recall, macro F1, and normalized confusion matrix dict."""
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Confusion matrix in percentage per true class
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.true_divide(cm, cm.sum(axis=1, keepdims=True)) * 100
        cm_pct = np.nan_to_num(cm_pct)

    cm_dict = {
        str(class_names[i]): {
            str(class_names[j]): float(round(cm_pct[i, j], 2))
            for j in range(len(class_names))
        }
        for i in range(len(class_names))
    }

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "confusion_matrix_pct": cm_dict,
    }


def run_cv_experiment(
    X, y, groups, group_kfold, param, eval_metric, objective, class_names
):
    """Executes GroupKFold cross-validation on a dataset permutation and collects results."""
    fold_results = []
    all_y_true = []
    all_y_pred = []

    # Clean params to prevent parameter collision
    param_clean = {
        k: v
        for k, v in param.items()
        if k not in ["eval_metric", "objective", "random_state"]
    }

    for fold, (train_idx, test_idx) in enumerate(
        group_kfold.split(X, y, groups=groups)
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Re-initialize a fresh model instance for every fold
        xgb_model = xgb.XGBClassifier(
            **param_clean,
            random_state=69,
            eval_metric=eval_metric,
            objective=objective,
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        fold_metrics = compute_metrics(y_test, y_pred, class_names)
        fold_metrics["fold"] = fold + 1
        fold_results.append(fold_metrics)

    # Compute overall out-of-fold (OOF) metrics across all combined test sets
    overall_results = compute_metrics(all_y_true, all_y_pred, class_names)
    overall_results["mean_fold_accuracy"] = round(
        float(np.mean([f["accuracy"] for f in fold_results])), 4
    )
    overall_results["std_fold_accuracy"] = round(
        float(np.std([f["accuracy"] for f in fold_results])), 4
    )

    return {
        "status": "success",
        "fold_results": fold_results,
        "final_results": overall_results,
    }


def save_progress_atomically(file_path: str, instances: list):
    """Saves output to a temporary file before renaming to prevent corrupted JSON writes on crash."""
    temp_path = f"{file_path}.tmp"
    output_payload = {
        "experiment_count": len(instances),
        "instances": instances,
    }
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4)

    os.replace(temp_path, file_path)


# --- Script Execution ---

eval_metric = "mlogloss"
objective = "multi:softprob"
n_splits = 5
json_path = r"../dataset/experiment_logs.jsonl"
output_results_path = r"../dataset/xgb_permutation_results.json"

# Load datasets
raw_data = pd.read_csv(r"../dataset/full.csv", index_col=False)
pruned_data = pd.read_csv(
    r"../dataset/full_v5.csv", index_col=False
).drop(columns=["split"], errors="ignore")
pca_data = pd.read_csv(
    r"../dataset/full_v5_pca.csv", index_col=False
).drop(columns=["split"], errors="ignore")

# Extract targets and group variables as 1D arrays/Series
groups = raw_data["subject"]
label_encoder = LabelEncoder()
raw_data_y_encoded = label_encoder.fit_transform(raw_data["Activity"])

# Separate features from meta-columns
raw_data = raw_data.drop(columns=["Activity", "subject"], errors="ignore")
pruned_data = pruned_data.drop(columns=["Activity", "subject"], errors="ignore")
pca_data = pca_data.drop(columns=["Activity", "subject"], errors="ignore")

group_kfold = GroupKFold(n_splits=n_splits)
parameters = load_unique_hyperparameters(json_path)

datasets = {
    "raw_data_results": raw_data,
    "pruned_data_results": pruned_data,
    "pca_data_results": pca_data,
}

# -----------------------------------------------------------------------------
# Resumption & Fault-Tolerance Logic
# -----------------------------------------------------------------------------
instances = []
completed_param_keys = set()

# Resume existing results if output file exists
if os.path.exists(output_results_path):
    try:
        with open(output_results_path, "r", encoding="utf-8") as f:
            existing_payload = json.load(f)
            instances = existing_payload.get("instances", [])
            for inst in instances:
                if "parameters" in inst:
                    param_key = json.dumps(
                        inst["parameters"], sort_keys=True
                    )
                    completed_param_keys.add(param_key)
        print(
            f"Found existing progress! Loaded {len(instances)} completed experiments. Resuming..."
        )
    except Exception as e:
        print(
            f"Warning: Failed to parse existing results file ({e}). Starting fresh."
        )
        instances = []

# Main Loop over hyperparameter combinations
for idx, param in enumerate(parameters):
    param_key = json.dumps(param, sort_keys=True)

    # Skip already completed hyperparameter sets
    if param_key in completed_param_keys:
        print(
            f"Skipping Parameter Set {idx + 1}/{len(parameters)} (Already executed)"
        )
        continue

    print(f"\n=== Running Parameter Set {idx + 1}/{len(parameters)} ===")

    instance = {
        "instance_id": len(instances) + 1,
        "parameters": param,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Catch any error at the instance level so individual failures do not stop the script
    try:
        for ds_name, ds_df in datasets.items():
            print(f"  -> Testing dataset permutation: {ds_name}")
            try:
                results = run_cv_experiment(
                    X=ds_df,
                    y=raw_data_y_encoded,
                    groups=groups,
                    group_kfold=group_kfold,
                    param=param,
                    eval_metric=eval_metric,
                    objective=objective,
                    class_names=label_encoder.classes_,
                )
                instance[ds_name] = results
            except Exception as e:
                print(
                    f"  [ERROR] Permutation '{ds_name}' failed for parameter set {idx + 1}: {e}"
                )
                traceback.print_exc()
                instance[ds_name] = {
                    "status": "failed",
                    "error": str(e),
                    "final_results": None,
                    "fold_results": [],
                }

    except Exception as instance_error:
        print(
            f"[CRITICAL ERROR] Failed to process Parameter Set {idx + 1}: {instance_error}"
        )
        instance["status"] = "failed"
        instance["error"] = str(instance_error)

    # Record instance and update completed set
    instances.append(instance)
    completed_param_keys.add(param_key)

    # Immediately write progress to disk after EACH hyperparameter set finishes
    save_progress_atomically(output_results_path, instances)
    print(f"--> Saved progress ({len(instances)}/{len(parameters)} complete)")

print(f"\nAll experiments finished! Final results saved to {output_results_path}")