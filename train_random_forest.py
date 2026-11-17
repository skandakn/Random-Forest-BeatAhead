"""
train_random_forest.py - End-to-end training and evaluation pipeline for BeatAhead Random Forest model.
"""

import json
import logging
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset import (
    FEATURES_MATRIX_A,
    load_clean_dataset,
    create_subject_level_partitions,
)
from src.feature_pipeline import TabularPreprocessor
from src.models import BeatAheadRandomForestClassifier
from src.metrics import compute_classification_metrics, find_optimal_threshold
from src.plotting import (
    plot_pr_and_roc_curves,
    plot_confusion_matrix,
    plot_feature_importances,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("beatahead.rf")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"


def main():
    start_time = time.time()
    logger.info("Initializing BeatAhead Random Forest Modeling Pipeline...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("Loading and filtering BeatAhead dataset...")
    df_clean = load_clean_dataset(DATA_DIR)
    logger.info(f"Loaded {len(df_clean)} clean windows across {df_clean['caseid'].nunique()} patient cases.")

    # 2. Strict Subject-Level Split (Zero Leakage)
    logger.info("Partitioning into 80 Development and 20 Locked Test cases...")
    df_dev, df_test, dev_cases, test_cases = create_subject_level_partitions(df_clean, DATA_DIR, seed=42)
    logger.info(f"Dev Set: {len(dev_cases)} cases ({len(df_dev)} windows, {int((df_dev['target_label']==1).sum())} pos)")
    logger.info(f"Test Set: {len(test_cases)} cases ({len(df_test)} windows, {int((df_test['target_label']==1).sum())} pos)")

    # 3. 5-Fold StratifiedGroupKFold on Dev Set
    logger.info("Running 5-Fold StratifiedGroupKFold on Development Set...")
    feature_cols = FEATURES_MATRIX_A
    X_dev = df_dev[feature_cols].copy()
    y_dev = df_dev["target_label"].values
    groups_dev = df_dev["caseid"].values
    case_has_pos = df_dev.groupby("caseid")["target_label"].transform(lambda s: int((s == 1).sum() > 0)).values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(df_dev))
    fold_metrics = []

    # Model Hyperparameters
    rf_config = {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": 42,
    }

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, case_has_pos, groups=groups_dev), 1):
        assert len(set(groups_dev[train_idx]).intersection(set(groups_dev[val_idx]))) == 0

        fold_model = BeatAheadRandomForestClassifier(
            feature_cols=feature_cols,
            **rf_config,
            n_jobs=-1,
        )
        fold_model.fit(df_dev.iloc[train_idx], y_dev[train_idx])
        val_probs = fold_model.predict_proba(df_dev.iloc[val_idx])
        oof_probs[val_idx] = val_probs

        metrics_fold = compute_classification_metrics(y_dev[val_idx], val_probs, threshold=0.5)
        fold_metrics.append({
            "fold": fold,
            "pr_auc": metrics_fold["pr_auc"],
            "roc_auc": metrics_fold["roc_auc"],
            "brier_score": metrics_fold["brier_score"],
        })
        logger.info(f"  Fold {fold}: PR-AUC = {metrics_fold['pr_auc']:.4f}, ROC-AUC = {metrics_fold['roc_auc']:.4f}")

    # 4. Out-of-fold metrics and threshold calibration
    oof_metrics_raw = compute_classification_metrics(y_dev, oof_probs, threshold=0.5)
    best_threshold, best_f1 = find_optimal_threshold(y_dev, oof_probs)
    oof_metrics_calibrated = compute_classification_metrics(y_dev, oof_probs, threshold=best_threshold)

    logger.info(f"OOF PR-AUC: {oof_metrics_raw['pr_auc']:.4f}, ROC-AUC: {oof_metrics_raw['roc_auc']:.4f}")
    logger.info(f"Calibrated Optimal Decision Threshold: {best_threshold:.4f} (OOF F1 = {best_f1:.4f})")

    # 5. Fit Final Model on All 80 Development Cases
    logger.info("Training final production Random Forest model on full 80-case Development Set...")
    final_model = BeatAheadRandomForestClassifier(
        feature_cols=feature_cols,
        **rf_config,
        threshold=best_threshold,
        n_jobs=-1,
    )
    final_model.fit(df_dev, y_dev)

    # 6. Evaluate on Strictly Isolated 20 Locked Test Cases
    logger.info("Evaluating on 20 Locked Test Cases (Zero Data Leakage)...")
    test_probs = final_model.predict_proba(df_test)
    test_metrics = compute_classification_metrics(df_test["target_label"].values, test_probs, threshold=best_threshold)

    logger.info(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1_score']:.4f} (Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall_sensitivity']:.4f})")
    logger.info(f"Test Confusion Matrix: {test_metrics['confusion_matrix']}")

    # 7. Save Predictions CSV
    df_test_out = df_test[["caseid", "window_id", "start_time_min", "target_label"]].copy()
    df_test_out["predicted_probability"] = test_probs
    df_test_out["predicted_label"] = (test_probs >= best_threshold).astype(int)
    test_preds_path = OUTPUTS_DIR / "test_predictions.csv"
    df_test_out.to_csv(test_preds_path, index=False)
    logger.info(f"Saved test predictions to {test_preds_path}")

    # 8. Save Model Artifacts
    model_artifact_path = ARTIFACTS_DIR / "random_forest_model.joblib"
    joblib.dump(final_model, model_artifact_path)
    logger.info(f"Saved trained model artifact to {model_artifact_path}")

    # Feature schema
    feature_schema = {
        "schema_name": "BeatAhead_Matrix_A_Feature_Schema",
        "feature_count": len(feature_cols),
        "feature_order": feature_cols,
    }
    with open(ARTIFACTS_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    # Best RF config
    with open(ARTIFACTS_DIR / "best_rf_config.json", "w", encoding="utf-8") as f:
        json.dump(rf_config, f, indent=2)

    # Model metadata
    metadata = {
        "model_name": "BeatAhead_RandomForest_Model",
        "algorithm": "RandomForestClassifier (Balanced Subsampling)",
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "hyperparameters": rf_config,
        "development_cases": dev_cases,
        "test_cases": test_cases,
        "decision_threshold": best_threshold,
        "cv_folds": fold_metrics,
        "oof_metrics": oof_metrics_calibrated,
        "test_metrics": test_metrics,
        "training_time_sec": round(time.time() - start_time, 2),
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "cross_validation_folds": fold_metrics,
            "out_of_fold_calibrated": oof_metrics_calibrated,
            "locked_test_evaluation": test_metrics,
        }, f, indent=2)

    # 9. Generate Plots
    logger.info("Generating evaluation figures...")
    plot_pr_and_roc_curves(
        df_test["target_label"].values,
        test_probs,
        OUTPUTS_DIR,
        model_name="Random Forest",
        threshold=best_threshold,
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        OUTPUTS_DIR,
        model_name="Random Forest",
        threshold=best_threshold,
    )
    df_imp = final_model.get_feature_importances()
    plot_feature_importances(df_imp, OUTPUTS_DIR, top_n=15)
    df_imp.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)

    logger.info(f"Pipeline complete in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    main()
