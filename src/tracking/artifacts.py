"""MLflow visual and tabular evaluation artifact logging module for easySLM.

Coordinates rendering and uploading of:
- Confusion Matrix plots and tables (single-label and multi-label cross-activation)
- Per-Class Metrics heatmaps and tables (Precision, Recall, F1, Support)
- Label Cardinality distribution plots and tables
- Supports both per-epoch versioned logging (`epoch_1`, `epoch_2`, ...) and root summary logging
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.tracking.plotting import (
    create_cross_label_confusion_matrix_figure,
    create_label_cardinality_figure,
    create_per_class_metrics_figure,
    create_single_label_confusion_matrix_figure,
)
from src.tracking.tables import (
    create_cross_label_confusion_matrix_table,
    create_label_cardinality_table,
    create_per_class_metrics_table,
    create_single_label_confusion_matrix_table,
)

logger = logging.getLogger("easySLM.tracking.artifacts")


def log_evaluation_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    multi_label: bool = True,
    local_output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Generates and logs confusion matrices, metrics heatmaps, and cardinality plots/tables to MLflow.

    Args:
        y_true: Binary indicator array (num_samples, num_classes).
        y_pred: Binary indicator array (num_samples, num_classes).
        classes: List of class names.
        multi_label: True if multi-label, False if single-label.
        local_output_dir: Optional local directory where figures and JSON tables will also be saved.

    Returns:
        dict[str, str]: Paths of generated artifacts.
    """
    import mlflow

    has_active_run = mlflow.active_run() is not None
    generated: dict[str, str] = {}

    local_dir = Path(local_output_dir) if local_output_dir else None
    local_plots_dir = (local_dir / "plots") if local_dir else None
    local_tables_dir = (local_dir / "tables") if local_dir else None

    if local_plots_dir:
        local_plots_dir.mkdir(parents=True, exist_ok=True)
    if local_tables_dir:
        local_tables_dir.mkdir(parents=True, exist_ok=True)

    if multi_label:
        # 1. Multi-label Cross-Label Confusion Matrix Plot
        fig_cross = create_cross_label_confusion_matrix_figure(y_true, y_pred, classes)
        if has_active_run:
            mlflow.log_figure(fig_cross, "plots/cross_label_confusion_matrix.png")
        if local_plots_dir:
            p = local_plots_dir / "cross_label_confusion_matrix.png"
            fig_cross.savefig(p, dpi=150)
            generated["plot_cross_cm"] = str(p)
        plt.close(fig_cross)

        # 2. Multi-label Cross-Label Confusion Matrix Table
        tbl_cross = create_cross_label_confusion_matrix_table(y_true, y_pred, classes)
        if has_active_run:
            mlflow.log_table(tbl_cross, "tables/cross_label_confusion_matrix.json")
        if local_tables_dir:
            p = local_tables_dir / "cross_label_confusion_matrix.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(tbl_cross, f, indent=2)
            generated["table_cross_cm"] = str(p)

    else:
        # 1. Single-label Confusion Matrix Plot
        fig_cm = create_single_label_confusion_matrix_figure(y_true, y_pred, classes)
        if has_active_run:
            mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")
        if local_plots_dir:
            p = local_plots_dir / "confusion_matrix.png"
            fig_cm.savefig(p, dpi=150)
            generated["plot_cm"] = str(p)
        plt.close(fig_cm)

        # 2. Single-label Confusion Matrix Table
        tbl_cm = create_single_label_confusion_matrix_table(y_true, y_pred, classes)
        if has_active_run:
            mlflow.log_table(tbl_cm, "tables/confusion_matrix.json")
        if local_tables_dir:
            p = local_tables_dir / "confusion_matrix.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(tbl_cm, f, indent=2)
            generated["table_cm"] = str(p)

    # 3. Per-Class Metrics Heatmap Plot
    fig_metrics = create_per_class_metrics_figure(y_true, y_pred, classes)
    if has_active_run:
        mlflow.log_figure(fig_metrics, "plots/per_class_metrics.png")
    if local_plots_dir:
        p = local_plots_dir / "per_class_metrics.png"
        fig_metrics.savefig(p, dpi=150)
        generated["plot_metrics"] = str(p)
    plt.close(fig_metrics)

    # 4. Per-Class Metrics Table
    tbl_metrics = create_per_class_metrics_table(y_true, y_pred, classes)
    if has_active_run:
        mlflow.log_table(tbl_metrics, "tables/per_class_metrics.json")
    if local_tables_dir:
        p = local_tables_dir / "per_class_metrics.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(tbl_metrics, f, indent=2)
        generated["table_metrics"] = str(p)

    # 5. Label Cardinality Plot
    fig_card = create_label_cardinality_figure(y_true, y_pred)
    if has_active_run:
        mlflow.log_figure(fig_card, "plots/label_cardinality.png")
    if local_plots_dir:
        p = local_plots_dir / "label_cardinality.png"
        fig_card.savefig(p, dpi=150)
        generated["plot_cardinality"] = str(p)
    plt.close(fig_card)

    # 6. Label Cardinality Table
    tbl_card = create_label_cardinality_table(y_true, y_pred)
    if has_active_run:
        mlflow.log_table(tbl_card, "tables/label_cardinality.json")
    if local_tables_dir:
        p = local_tables_dir / "label_cardinality.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(tbl_card, f, indent=2)
        generated["table_cardinality"] = str(p)

    logger.info("Evaluation artifacts successfully logged (items: %d).", len(generated))
    return generated
