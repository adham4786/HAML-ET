import os
import ast
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
from datetime import datetime as dt

# -------------------------------------------------------------------------
# 1. Load PCA-Processed Dataset (217 Features)
# -------------------------------------------------------------------------
# Adjust file path if your 217-feature PCA dataset is named differently
pca_data_path = '../dataset/full_v3_pca_2026-09-24_22:55:59.csv' 
df = pd.read_csv(pca_data_path)

# Separate features, target, and subject groups
X = df.drop(columns=['Activity', 'subject'])
y = df['Activity']
groups = df['subject']

# Integer-encode target labels for XGBoost compatibility
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------------------------------------------------
# 2. Train / Hold-Out Split (80% Train / 20% Test by Subject Group)
# -------------------------------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=69)
train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

# -------------------------------------------------------------------------
# 3. Parameter Search Space Optimized for 217 Orthogonal PCA Features
# -------------------------------------------------------------------------
param_distributions = {
    'n_estimators': randint(150, 600),          # Expanded tree counts for deeper learning
    'max_depth': randint(3, 10),                 # Deeper search range for complex boundaries
    'learning_rate': uniform(0.008, 0.18),       # Fine-grained step sizes
    'subsample': uniform(0.6, 0.4),              # Row sampling (60% to 100%)
    'colsample_bytree': uniform(0.5, 0.5),       # Keep higher fraction of PCA features (50% to 100%)
    'colsample_bylevel': uniform(0.5, 0.5),      # Keep higher fraction per tree level (50% to 100%)
    'min_child_weight': randint(1, 12),          # Handles noise tolerance
    'gamma': uniform(0, 1.2),                    # Minimum split loss reduction
    'reg_alpha': uniform(0, 6.0),                # L1 Regularization
    'reg_lambda': uniform(0.5, 10.0)             # L2 Regularization
}

# -------------------------------------------------------------------------
# 4. Base Estimator & CV Setup
# -------------------------------------------------------------------------
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=69,
    n_jobs=1                                     # Single-threaded per process
)

gkf = GroupKFold(n_splits=5)

# 220 iterations * 5 folds = 1,100 total model fits (~4 hours on 8 cores)
search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=220,
    scoring='f1_macro',
    cv=gkf,
    verbose=2,
    random_state=69,                             # Guarantees reproducible parameter selections
    n_jobs=8,                                    # Parallelized across 8 CPU cores
    return_train_score=True                      # Log train scores to check for overfitting
)

print(f"[{dt.now().strftime('%H:%M:%S')}] Starting ~4-hour hyperparameter search across {X_train.shape[1]} PCA features...")
print(f"Running 220 iterations with 5-fold GroupKFold (1,100 total model fits)...")

search.fit(X_train, y_train, groups=groups_train)

# -------------------------------------------------------------------------
# 5. Extract, Process & Export Complete Execution Log
# -------------------------------------------------------------------------
now_str = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs('../dataset', exist_ok=True)

# Build a clean DataFrame of all trials
cv_results = pd.DataFrame(search.cv_results_)

# Add explicit model seed metadata for complete end-to-end reproducibility
cv_results['model_random_state'] = 69
cv_results['cv_split_seed'] = 69

# Sort by best validation macro-F1 score
cv_results = cv_results.sort_values(by='rank_test_score').reset_index(drop=True)

# Save primary search results CSV as specified
output_file = rf'../dataset/pca_v3_param_seeking_{now_str}.csv'
cv_results.to_csv(output_file, index=False)

# -------------------------------------------------------------------------
# 6. Print Best Parameter Settings & Summary Stats
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("HYPERPARAMETER SEARCH COMPLETED SUCCESSFULLY")
print("="*80)
print(f"Results Log Saved To: {output_file}")
print("-" * 80)
print(f"Best Mean CV Macro-F1 Score: {search.best_score_:.5f}")
print("Best Parameter Configuration:")
print("-" * 80)
for param, val in search.best_params_.items():
    if isinstance(val, float):
        print(f"  {param:<20}: {val:.6f}")
    else:
        print(f"  {param:<20}: {val}")
print("="*80)