"""
BeatAhead Random Forest Package
"""

from .dataset import (
    FEATURES_MATRIX_A,
    FEATURES_MATRIX_B,
    load_clean_dataset,
    create_subject_level_partitions,
)
from .feature_pipeline import TabularPreprocessor
from .models import BeatAheadRandomForestClassifier
from .metrics import compute_classification_metrics, find_optimal_threshold
from .plotting import plot_pr_and_roc_curves, plot_confusion_matrix, plot_feature_importances

__all__ = [
    "FEATURES_MATRIX_A",
    "FEATURES_MATRIX_B",
    "load_clean_dataset",
    "create_subject_level_partitions",
    "TabularPreprocessor",
    "BeatAheadRandomForestClassifier",
    "compute_classification_metrics",
    "find_optimal_threshold",
    "plot_pr_and_roc_curves",
    "plot_confusion_matrix",
    "plot_feature_importances",
]
