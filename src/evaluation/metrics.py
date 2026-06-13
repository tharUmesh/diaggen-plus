"""
metrics.py
Evaluation utilities: accuracy, macro F1, per-class report, confusion matrix.
Used in both Phase 1 (baseline) and Phase 2 (augmented model comparison).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(y_true: list[int], y_pred: list[int],
             label_names: list[str]) -> dict:
    """
    Full evaluation report.
    Returns a dict with accuracy, macro_f1, weighted_f1, and per-class metrics.
    """
    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report     = classification_report(y_true, y_pred, target_names=label_names,
                                       output_dict=True, zero_division=0)

    logger.info(f"Accuracy:    {acc:.4f}")
    logger.info(f"Macro F1:    {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")

    return {
        "accuracy":    acc,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "report":      report,
    }


def plot_confusion_matrix(y_true: list[int], y_pred: list[int],
                          label_names: list[str],
                          save_path: str | Path | None = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_class_f1_comparison(baseline: dict, augmented: dict,
                              label_names: list[str],
                              save_path: str | Path | None = None) -> None:
    """
    Side-by-side bar chart comparing per-class F1 between baseline and augmented model.
    Useful for showing improvement on rare disease classes (Phase 2).
    """
    b_f1 = [baseline["report"].get(l, {}).get("f1-score", 0) for l in label_names]
    a_f1 = [augmented["report"].get(l, {}).get("f1-score", 0) for l in label_names]

    x    = np.arange(len(label_names))
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 0.2, b_f1, 0.4, label="Baseline", color="#5B9BD5")
    ax.bar(x + 0.2, a_f1, 0.4, label="Augmented", color="#ED7D31")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1: Baseline vs. VAE-Augmented Model")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"F1 comparison chart saved to {save_path}")
    plt.show()
