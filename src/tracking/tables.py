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


def create_validation_errors_table(
    input_texts: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    raw_outputs: Sequence[str],
    parsed_outputs: Sequence[Any],
    error_scores: Sequence[float] | None = None,
    max_errors_per_class: int = 5,
) -> dict[str, list[Any]]:
    """Creates a structured table of model prediction errors on the validation set.

    For each class, prioritizes the most egregious mistakes (highest error_scores).

    Args:
        input_texts: Raw user input text for each validation sample.
        y_true: Binary indicator array of shape `(num_samples, num_classes)`.
        y_pred: Binary indicator array of shape `(num_samples, num_classes)`.
        classes: Ordered list of class label names.
        raw_outputs: Formatted raw outputs (probability strings for CLS, decoded strings for JSON).
        parsed_outputs: Parsed prediction label(s).
        error_scores: Optional severity ranking score per sample (e.g. max probability on wrong class).
        max_errors_per_class: Max number of error rows to retain per true class.

    Returns:
        dict[str, list[Any]]: Table dictionary suitable for `mlflow.log_table`.
    """
    table: dict[str, list[Any]] = {
        "input_text": [],
        "ground_truth": [],
        "raw_output": [],
        "parsed_output": [],
    }

    # Find mismatch indices: samples where binary indicator differs
    mismatch_mask = np.any(y_true != y_pred, axis=1)
    error_indices = np.where(mismatch_mask)[0].tolist()
    if not error_indices:
        return table

    # Group error indices by true class
    class_to_errors: dict[str, list[int]] = {c: [] for c in classes}
    unmatched_errors: list[int] = []

    for idx in error_indices:
        true_class_indices = np.where(y_true[idx] == 1)[0]
        if len(true_class_indices) > 0:
            for c_idx in true_class_indices:
                class_to_errors[classes[c_idx]].append(idx)
        else:
            unmatched_errors.append(idx)

    selected_indices: set[int] = set()

    for c in classes:
        c_errs = list(dict.fromkeys(class_to_errors[c]))
        if error_scores is not None and len(c_errs) > 0:
            c_errs.sort(key=lambda i: error_scores[i], reverse=True)
        selected_for_c = c_errs[:max_errors_per_class]
        selected_indices.update(selected_for_c)

    if unmatched_errors:
        if error_scores is not None:
            unmatched_errors.sort(key=lambda i: error_scores[i], reverse=True)
        selected_indices.update(unmatched_errors[:max_errors_per_class])

    # Sort final selected indices for deterministic ordering
    sorted_selected = sorted(selected_indices)
    if error_scores is not None:
        sorted_selected.sort(key=lambda i: error_scores[i], reverse=True)

    for idx in sorted_selected:
        true_names = [classes[ci] for ci in np.where(y_true[idx] == 1)[0]]
        table["input_text"].append(str(input_texts[idx]))
        table["ground_truth"].append(true_names if len(true_names) != 1 else true_names[0])
        table["raw_output"].append(str(raw_outputs[idx]))
        parsed = parsed_outputs[idx]
        table["parsed_output"].append(parsed if isinstance(parsed, str | list) else str(parsed))

    return table
