"""
evaluation.py
=============
Evaluation metrics, confusion matrices, ROC/PR curves, and explainability utilities.

Functions:
- compute_metrics: Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
- plot_confusion_matrices: Multi-panel confusion matrix plot saved to outputs/figures/.
- plot_roc_pr_curves: ROC and Precision-Recall curves comparison saved to outputs/figures/.
- plot_feature_importances: Bar plot of feature importances for tree models.
- generate_shap_summary: SHAP summary beeswarm and bar plots.
"""

import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "outputs/figures"
METRICS_DIR = "outputs/metrics"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes a comprehensive dictionary of classification performance metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": report,
    }
    return metrics


def plot_confusion_matrices(results: Dict[str, Dict[str, Any]], filename: str = "06_confusion_matrices.png"):
    """
    Plots side-by-side normalized and raw confusion matrices for all evaluated models.
    """
    models = list(results.keys())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for idx, name in enumerate(models):
        cm_data = results[name]["confusion_matrix"]
        cm_array = np.array([
            [cm_data["true_negatives"], cm_data["false_positives"]],
            [cm_data["false_negatives"], cm_data["true_positives"]],
        ])
        ax = axes[idx]
        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred Normal (0)", "Pred Failure (1)"],
            yticklabels=["True Normal (0)", "True Failure (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        rec = results[name]["recall"]
        prec = results[name]["precision"]
        f1 = results[name]["f1_score"]
        ax.set_title(f"{name}\nRec: {rec:.2f} | Prec: {prec:.2f} | F1: {f1:.2f}", pad=10)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices to: {save_path}")


def plot_roc_pr_curves(
    eval_curves: Dict[str, Dict[str, np.ndarray]],
    filename: str = "07_roc_pr_curves.png",
):
    """
    Plots overlaid ROC Curves and Precision-Recall Curves for model comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    palette = ["#2b5c8f", "#2ca02c", "#d95f02", "#7570b3"]

    for idx, (name, curves) in enumerate(eval_curves.items()):
        color = palette[idx % len(palette)]
        y_true = curves["y_true"]
        y_prob = curves["y_prob"]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2)

        # PR Curve
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = average_precision_score(y_true, y_prob)
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc_val:.3f})", color=color, linewidth=2)

    # ROC formatting
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (0.50)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", pad=12)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")

    # PR formatting
    no_skill_baseline = float(np.mean(list(eval_curves.values())[0]["y_true"]))
    axes[1].axhline(no_skill_baseline, color="k", linestyle="--", alpha=0.6, label=f"Baseline Prevalence ({no_skill_baseline:.3f})")
    axes[1].set_title("Precision-Recall (PR) Curves", pad=12)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC and PR curves to: {save_path}")


def plot_feature_importances(
    importance_df: pd.DataFrame,
    filename: str = "08_feature_importances.png",
    top_n: int = 10,
):
    """
    Plots feature importances for tree models.
    """
    fig, axes = plt.subplots(1, len(importance_df.columns), figsize=(6 * len(importance_df.columns), 6))
    if len(importance_df.columns) == 1:
        axes = [axes]

    colors = ["#2b5c8f", "#d95f02"]
    for idx, col in enumerate(importance_df.columns):
        ax = axes[idx]
        sorted_imp = importance_df[col].sort_values(ascending=True).tail(top_n)
        ax.barh(sorted_imp.index, sorted_imp.values, color=colors[idx % len(colors)], edgecolor="black", alpha=0.85)
        ax.set_title(f"Feature Importance: {col}", pad=12)
        ax.set_xlabel("Relative Importance Score")

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importances plot to: {save_path}")
