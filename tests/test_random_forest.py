"""
tests/test_random_forest.py - Unit test suite for BeatAhead Random Forest pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Allow direct execution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.dataset import (
    FEATURES_MATRIX_A,
    load_clean_dataset,
    create_subject_level_partitions,
)
from src.feature_pipeline import TabularPreprocessor
from src.models import BeatAheadRandomForestClassifier
from src.metrics import compute_classification_metrics, find_optimal_threshold

DATA_DIR = BASE_DIR / "data"


def test_dataset_loading_and_filtering():
    """Validates that loaded dataset only contains SQI PASS and valid 0/1 binary labels."""
    df_clean = load_clean_dataset(DATA_DIR)
    assert len(df_clean) > 0
    assert set(df_clean["sqi_status"].unique()) == {"PASS"}
    assert set(df_clean["target_label"].unique()).issubset({0, 1})
    print("  [PASS] test_dataset_loading_and_filtering")


def test_strict_subject_level_isolation():
    """Guarantees zero subject-level data leakage between 80 Dev and 20 Test cases."""
    df_clean = load_clean_dataset(DATA_DIR)
    df_dev, df_test, dev_cases, test_cases = create_subject_level_partitions(df_clean, DATA_DIR, seed=42)

    assert len(dev_cases) == 80
    assert len(test_cases) == 20
    assert len(set(dev_cases).intersection(set(test_cases))) == 0
    assert len(df_dev) > 0
    assert len(df_test) > 0
    print("  [PASS] test_strict_subject_level_isolation")


def test_tabular_preprocessor():
    """Tests median imputation fitted on train."""
    df_train = pd.DataFrame({
        "feat1": [1.0, 2.0, np.nan, 4.0],
        "feat2": [10.0, 20.0, 30.0, 40.0],
    })
    prep = TabularPreprocessor(["feat1", "feat2"])
    X_trans = prep.fit_transform(df_train)

    assert X_trans.shape == (4, 2)
    assert not np.isnan(X_trans).any()
    print("  [PASS] test_tabular_preprocessor")


def test_rf_model_fit_and_predict():
    """Validates fitting, probability bounds, and feature importance normalization."""
    df_clean = load_clean_dataset(DATA_DIR)
    features = FEATURES_MATRIX_A
    sample_df = df_clean.head(100).copy()
    y = sample_df["target_label"].values.copy()
    if len(np.unique(y)) == 1:
        y[0] = 1  # ensure at least one positive

    clf = BeatAheadRandomForestClassifier(
        feature_cols=features,
        n_estimators=10,
        max_depth=3,
        random_state=42,
    )
    clf.fit(sample_df, y)

    probs = clf.predict_proba(sample_df)
    preds = clf.predict(sample_df, threshold=0.5)

    assert len(probs) == len(sample_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert set(np.unique(preds)).issubset({0, 1})

    df_imp = clf.get_feature_importances()
    assert len(df_imp) == len(features)
    assert np.isclose(df_imp["importance"].sum(), 1.0)
    print("  [PASS] test_rf_model_fit_and_predict")


def test_metrics_computation():
    """Tests calculation of PR-AUC, ROC-AUC, and confusion matrix."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["pr_auc"] > 0.9
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["true_negatives"] == 2
    print("  [PASS] test_metrics_computation")


if __name__ == "__main__":
    print("Running Random Forest Unit Tests...")
    test_dataset_loading_and_filtering()
    test_strict_subject_level_isolation()
    test_tabular_preprocessor()
    test_rf_model_fit_and_predict()
    test_metrics_computation()
    print("All unit tests passed successfully!")
