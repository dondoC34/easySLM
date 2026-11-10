"""MLflow table generator for classification metrics, confusion matrices, and cardinality.

Prepares structured data dictionaries compatible with `mlflow.log_table` to enable
interactive comparison and inspection inside the MLflow UI 'Artifacts > Tables' section.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger("easySLM.tracking.tables")


def create_single_label_confusion_matrix_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
) -> dict[str, list[Any]]:
    """Creates a structured tabular representation of the single-label confusion matrix.

    Args:
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.
        classes: Ordered sequence of class label names.

    Returns:
        dict[str, list[Any]]: Table dictionary suitable for `mlflow.log_table`.
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

    table: dict[str, list[Any]] = {
        "true_class": [],
        "predicted_class": [],
        "count": [],
        "normalized_rate": [],
        "is_diagonal_correct": [],
    }

    for i, t_class in enumerate(classes):
        for j, p_class in enumerate(classes):
            table["true_class"].append(t_class)
            table["predicted_class"].append(p_class)
            table["count"].append(int(cm[i, j]))
            table["normalized_rate"].append(round(float(cm_norm[i, j]), 4))
            table["is_diagonal_correct"].append(i == j)

    return table


def create_cross_label_confusion_matrix_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
) -> dict[str, list[Any]]:
    """Creates a structured tabular representation of the multi-label cross-label error matrix.

    Args:
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.
        classes: Ordered sequence of class label names.

    Returns:
        dict[str, list[Any]]: Table dictionary suitable for `mlflow.log_table`.
    """
    co_matrix = (y_true.T @ y_pred).astype(int)
    support = y_true.sum(axis=0)

    norm_matrix = np.divide(
        co_matrix.astype(float),
        support[:, None],
        out=np.zeros_like(co_matrix, dtype=float),
        where=support[:, None] > 0,
    )

    table: dict[str, list[Any]] = {
        "true_class": [],
        "predicted_class": [],
        "co_occurrence_count": [],
        "conditional_probability": [],
        "true_class_support": [],
        "is_diagonal_recall": [],
    }

    for i, t_class in enumerate(classes):
        for j, p_class in enumerate(classes):
            table["true_class"].append(t_class)
            table["predicted_class"].append(p_class)
            table["co_occurrence_count"].append(int(co_matrix[i, j]))
            table["conditional_probability"].append(round(float(norm_matrix[i, j]), 4))
            table["true_class_support"].append(int(support[i]))
            table["is_diagonal_recall"].append(i == j)

    return table


def create_per_class_metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
) -> dict[str, list[Any]]:
    """Creates a per-class performance metrics table (Precision, Recall, F1, Support).

    Args:
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.
        classes: Ordered sequence of class label names.

    Returns:
        dict[str, list[Any]]: Table dictionary suitable for `mlflow.log_table`.
    """
    prec = precision_score(y_true, y_pred, average=None, zero_division=0.0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0.0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0.0)
    support = y_true.sum(axis=0).astype(int)
    pred_count = y_pred.sum(axis=0).astype(int)

    table: dict[str, list[Any]] = {
        "class_name": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "support": [],
        "predicted_count": [],
    }

    for i, c_name in enumerate(classes):
        table["class_name"].append(c_name)
        table["precision"].append(round(float(prec[i]), 4))
        table["recall"].append(round(float(rec[i]), 4))
        table["f1_score"].append(round(float(f1[i]), 4))
        table["support"].append(int(support[i]))
        table["predicted_count"].append(int(pred_count[i]))

    return table


def create_label_cardinality_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, list[Any]]:
    """Creates a label cardinality distribution comparison table.

    Args:
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.

    Returns:
        dict[str, list[Any]]: Table dictionary suitable for `mlflow.log_table`.
    """
    true_counts = y_true.sum(axis=1).astype(int)
    pred_counts = y_pred.sum(axis=1).astype(int)
    total_samples = max(len(y_true), 1)

    max_cardinality = max(
        int(np.max(true_counts, initial=0)), int(np.max(pred_counts, initial=0)), 1
    )
    categories = list(range(max_cardinality + 1))

    table: dict[str, list[Any]] = {
        "num_labels": [],
        "ground_truth_samples": [],
        "predicted_samples": [],
        "ground_truth_pct": [],
        "predicted_pct": [],
    }

    for k in categories:
        t_freq = int(np.sum(true_counts == k))
        p_freq = int(np.sum(pred_counts == k))
        table["num_labels"].append(k)
        table["ground_truth_samples"].append(t_freq)
        table["predicted_samples"].append(p_freq)
        table["ground_truth_pct"].append(round((t_freq / total_samples) * 100.0, 2))
        table["predicted_pct"].append(round((p_freq / total_samples) * 100.0, 2))

    return table
