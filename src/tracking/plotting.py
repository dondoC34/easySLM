"""MLflow visual diagnostics and classification evaluation plots for easySLM.

Generates Matplotlib figures for single-label and multi-label classification tasks:
- Single-label: Normalized Confusion Matrix ($K \\times K$).
- Multi-label: Cross-Label Error / Co-activation Matrix ($K \\times K$).
- Per-Class Metrics Heatmap (Precision, Recall, F1 per class).
- Label Cardinality Distribution (Ground Truth vs Prediction label count comparison).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib

# Use headless Agg backend to guarantee thread safety and prevent GUI popups
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger("easySLM.tracking.plotting")


def _plot_annotated_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cell_texts: list[list[str]],
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "Blues",
    cbar_label: str = "",
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: tuple[float, float] = (8.0, 6.5),
) -> plt.Figure:
    """Helper that renders an annotated 2D heatmap figure."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=10)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            # Use white text on dark cells and black on light cells
            color = "white" if val > 0.55 else "black"
            txt = cell_texts[i][j]
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=color,
                fontsize=8.5,
                fontweight="normal",
            )

    fig.tight_layout()
    return fig


def create_single_label_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    epoch: int | None = None,
) -> plt.Figure:
    """Generates a standard normalized Confusion Matrix figure for single-label multiclass tasks.

    Args:
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.
        classes: Ordered sequence of class label names.
        epoch: Optional training epoch number for title labeling.

    Returns:
        plt.Figure: The Matplotlib figure.
    """
    true_idx = np.argmax(y_true, axis=-1)
    pred_idx = np.argmax(y_pred, axis=-1)
    labels = list(range(len(classes)))

    cm = confusion_matrix(true_idx, pred_idx, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums > 0,
    )

    cell_texts = [
        [f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)" for j in range(len(classes))]
        for i in range(len(classes))
    ]

    epoch_tag = f" (Epoch {epoch})" if epoch is not None else ""
    return _plot_annotated_heatmap(
        matrix=cm_norm,
        row_labels=classes,
        col_labels=classes,
        cell_texts=cell_texts,
        title=f"Single-Label Confusion Matrix{epoch_tag}",
        xlabel="Predicted Class",
        ylabel="True Class",
        cmap="Blues",
        cbar_label="Recall / Normalized Proportion",
        vmin=0.0,
        vmax=1.0,
    )


def create_cross_label_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    epoch: int | None = None,
) -> plt.Figure:
    """Generates a Cross-Label Error and Co-activation Matrix figure for multi-label tasks.

    Cell (i, j) represents P(Predicted j | True i), highlighting true recall on diagonal
    and cross-label confusions on off-diagonals.

    Args:
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.
        classes: Ordered sequence of class label names.
        epoch: Optional training epoch number for title labeling.

    Returns:
        plt.Figure: The Matplotlib figure.
    """
    # Co-occurrence counts: shape (num_classes, num_classes)
    co_matrix = (y_true.T @ y_pred).astype(int)
    support = y_true.sum(axis=0)  # Total true samples per class

    norm_matrix = np.divide(
        co_matrix.astype(float),
        support[:, None],
        out=np.zeros_like(co_matrix, dtype=float),
        where=support[:, None] > 0,
    )

    cell_texts = [
        [f"{co_matrix[i, j]}\n({norm_matrix[i, j] * 100:.1f}%)" for j in range(len(classes))]
        for i in range(len(classes))
    ]

    epoch_tag = f" (Epoch {epoch})" if epoch is not None else ""
    return _plot_annotated_heatmap(
        matrix=norm_matrix,
        row_labels=classes,
        col_labels=classes,
        cell_texts=cell_texts,
        title=f"Multi-Label Cross-Label Confusion Matrix{epoch_tag}",
        xlabel="Predicted Label",
        ylabel="True Label (Ground Truth)",
        cmap="Blues",
        cbar_label="Conditional Activation Probability P(Pred j | True i)",
        vmin=0.0,
        vmax=1.0,
    )


def create_per_class_metrics_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    epoch: int | None = None,
) -> plt.Figure:
    """Generates a Per-Class Performance Heatmap (Precision, Recall, F1-Score).

    Args:
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.
        classes: Ordered sequence of class label names.
        epoch: Optional training epoch number for title labeling.

    Returns:
        plt.Figure: The Matplotlib figure.
    """
    prec = precision_score(y_true, y_pred, average=None, zero_division=0.0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0.0)

    matrix = np.column_stack([prec, rec, f1])
    metric_names = ["Precision", "Recall", "F1-Score"]

    cell_texts = [
        [f"{matrix[i, j]:.3f}" for j in range(len(metric_names))] for i in range(len(classes))
    ]

    epoch_tag = f" (Epoch {epoch})" if epoch is not None else ""
    return _plot_annotated_heatmap(
        matrix=matrix,
        row_labels=classes,
        col_labels=metric_names,
        cell_texts=cell_texts,
        title=f"Per-Class Evaluation Metrics{epoch_tag}",
        xlabel="Metric",
        ylabel="Class Label",
        cmap="RdYlGn",
        cbar_label="Score (0.0 - 1.0)",
        vmin=0.0,
        vmax=1.0,
        figsize=(6.5, max(4.0, len(classes) * 0.7)),
    )


def create_label_cardinality_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epoch: int | None = None,
) -> plt.Figure:
    """Generates a bar chart comparing label cardinality distribution (Ground Truth vs Prediction).

    Args:
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.
        epoch: Optional training epoch number for title labeling.

    Returns:
        plt.Figure: The Matplotlib figure.
    """
    true_counts = y_true.sum(axis=1).astype(int)
    pred_counts = y_pred.sum(axis=1).astype(int)
    total_samples = max(len(y_true), 1)

    max_cardinality = max(
        int(np.max(true_counts, initial=0)), int(np.max(pred_counts, initial=0)), 1
    )
    categories = list(range(max_cardinality + 1))

    true_freqs = [int(np.sum(true_counts == k)) for k in categories]
    pred_freqs = [int(np.sum(pred_counts == k)) for k in categories]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        true_freqs,
        width,
        label="Ground Truth",
        color="#1f77b4",
        edgecolor="black",
        alpha=0.85,
    )
    rects2 = ax.bar(
        x + width / 2,
        pred_freqs,
        width,
        label="Predicted",
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.85,
    )

    # Add count labels above bars
    for rect, freq in [(rects1, true_freqs), (rects2, pred_freqs)]:
        for bar, count in zip(rect, freq, strict=False):
            height = bar.get_height()
            pct = (count / total_samples) * 100.0
            ax.annotate(
                f"{count}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} labels" for k in categories], fontsize=9)
    ax.set_xlabel(
        "Number of Labels Assigned per Sample", fontsize=11, fontweight="bold", labelpad=8
    )
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold", labelpad=8)
    epoch_tag = f" - Epoch {epoch}" if epoch is not None else ""
    ax.set_title(
        f"Label Cardinality Distribution (Validation Set: {total_samples} samples{epoch_tag})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    # Set ylim to give room for text annotations
    max_height = max(max(true_freqs, default=0), max(pred_freqs, default=0))
    ax.set_ylim(0, max_height * 1.25 + 1)

    fig.tight_layout()
    return fig
