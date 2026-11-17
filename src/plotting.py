"""
src/plotting.py - Visualization utilities for Random Forest model evaluation and feature importances.
"""

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def plot_pr_and_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    model_name: str = "Random Forest",
    threshold: Optional[float] = None,
) -> None:
    """Generates and saves PR and ROC curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Precision-Recall Curve
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)
    pr_auc_val = auc(recall, precision)

    ax1.plot(recall, precision, color="#1f77b4", lw=2, label=f"{model_name} (PR-AUC = {pr_auc_val:.3f})")
    no_skill = np.sum(y_true) / len(y_true)
    ax1.axhline(no_skill, color="#7f7f7f", linestyle="--", label=f"Baseline Prevalence ({no_skill*100:.2f}%)")

    if threshold is not None:
        idx = np.argmin(np.abs(pr_thresh - threshold)) if len(pr_thresh) > 0 else 0
        if idx < len(recall) and idx < len(precision):
            ax1.scatter([recall[idx]], [precision[idx]], color="#d62728", s=100, zorder=5, label=f"Calibrated Thresh ({threshold:.2f})")

    ax1.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax1.set_title(f"Precision-Recall Curve - {model_name}", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", frameon=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    ax2.plot(fpr, tpr, color="#2ca02c", lw=2, label=f"{model_name} (ROC-AUC = {roc_auc_val:.3f})")
    ax2.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", label="Chance (AUC = 0.50)")
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax2.set_title(f"ROC Curve - {model_name}", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_dir / "pr_roc_curves.png", dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm_dict: dict,
    output_dir: Path,
    model_name: str = "Random Forest",
    threshold: float = 0.5,
) -> None:
    """Generates annotated confusion matrix heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.array([
        [cm_dict["true_negatives"], cm_dict["false_positives"]],
        [cm_dict["false_negatives"], cm_dict["true_positives"]],
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=["Predicted No Ischemia", "Predicted Ischemia"],
        yticklabels=["Actual No Ischemia", "Actual Ischemia"],
    )
    plt.title(f"Confusion Matrix ({model_name}, Threshold={threshold:.2f})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_feature_importances(
    df_imp: pd.DataFrame,
    output_dir: Path,
    top_n: int = 15,
) -> None:
    """Plots top Gini Mean Decrease in Impurity (MDI) feature importances."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df_top = df_imp.head(top_n).copy()

    plt.figure(figsize=(9, 6))
    sns.barplot(x="importance", y="feature", data=df_top, hue="feature", palette="viridis", legend=False)
    plt.title(f"Top {top_n} Random Forest Feature Importances (Gini Impurity Reduction)", fontsize=11, fontweight="bold")
    plt.xlabel("Relative MDI Feature Importance", fontsize=10, fontweight="bold")
    plt.ylabel("Physiological Feature", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()
