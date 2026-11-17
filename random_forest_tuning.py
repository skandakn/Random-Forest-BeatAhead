"""
random_forest_tuning.py - Systematic hyperparameter tuning across 5-Fold StratifiedGroupKFold.
"""

import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset import (
    FEATURES_MATRIX_A,
    load_clean_dataset,
    create_subject_level_partitions,
)
from src.models import BeatAheadRandomForestClassifier
from src.metrics import compute_classification_metrics, find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("beatahead.rf_tuning")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"


def run_tuning_grid():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Random Forest Hyperparameter Grid Search on 80 Dev cases...")

    df_clean = load_clean_dataset(DATA_DIR)
    df_dev, _, _, _ = create_subject_level_partitions(df_clean, DATA_DIR, seed=42)

    feature_cols = FEATURES_MATRIX_A
    X_dev = df_dev[feature_cols].copy()
    y_dev = df_dev["target_label"].values
    groups_dev = df_dev["caseid"].values
    case_has_pos = df_dev.groupby("caseid")["target_label"].transform(lambda s: int((s == 1).sum() > 0)).values

    param_grid = [
        {"max_depth": 4, "min_samples_leaf": 10, "n_estimators": 100},
        {"max_depth": 5, "min_samples_leaf": 5, "n_estimators": 150},
        {"max_depth": 6, "min_samples_leaf": 5, "n_estimators": 200},
        {"max_depth": 8, "min_samples_leaf": 2, "n_estimators": 200},
    ]

    results = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for i, params in enumerate(param_grid, 1):
        logger.info(f"Evaluating candidate {i}/{len(param_grid)}: {params}")
        t0 = time.time()
        oof_probs = np.zeros(len(df_dev))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, case_has_pos, groups=groups_dev), 1):
            model = BeatAheadRandomForestClassifier(
                feature_cols=feature_cols,
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                n_estimators=params["n_estimators"],
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(df_dev.iloc[train_idx], y_dev[train_idx])
            oof_probs[val_idx] = model.predict_proba(df_dev.iloc[val_idx])

        opt_thresh, opt_f1 = find_optimal_threshold(y_dev, oof_probs)
        metrics = compute_classification_metrics(y_dev, oof_probs, threshold=opt_thresh)

        res = {
            **params,
            "oof_pr_auc": round(metrics["pr_auc"], 4),
            "oof_roc_auc": round(metrics["roc_auc"], 4),
            "optimal_threshold": round(opt_thresh, 4),
            "optimal_f1": round(opt_f1, 4),
            "oof_precision": round(metrics["precision"], 4),
            "oof_recall": round(metrics["recall_sensitivity"], 4),
            "eval_time_sec": round(time.time() - t0, 2),
        }
        results.append(res)
        logger.info(f"  Result: OOF PR-AUC={res['oof_pr_auc']}, ROC-AUC={res['oof_roc_auc']}, Opt F1={res['optimal_f1']}")

    df_res = pd.DataFrame(results).sort_values("oof_pr_auc", ascending=False).reset_index(drop=True)
    csv_out = OUTPUTS_DIR / "hyperparameter_results.csv"
    df_res.to_csv(csv_out, index=False)
    logger.info(f"Saved tuning results to {csv_out}")
    print("\n" + df_res.to_string())


if __name__ == "__main__":
    run_tuning_grid()
