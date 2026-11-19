"""MLflow visual and tabular evaluation artifact logging module for easySLM.

Coordinates rendering and uploading of:
- Confusion Matrix plots and tables (single-label and multi-label cross-activation)
- Per-Class Metrics heatmaps and tables (Precision, Recall, F1, Support)
- Label Cardinality distribution plots and tables
- Supports both per-epoch versioned logging (`epoch_1`, `epoch_2`, ...) and root summary logging
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    validation_errors_table: dict[str, list[Any]] | None = None,
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

    # 7. Validation Errors Table (if provided)
    if validation_errors_table is not None:
        if has_active_run:
            mlflow.log_table(validation_errors_table, "tables/validation_errors.json")
        if local_tables_dir:
            p = local_tables_dir / "validation_errors.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(validation_errors_table, f, indent=2)
            generated["table_errors"] = str(p)

    logger.info("Evaluation artifacts successfully logged (items: %d).", len(generated))
    return generated


def _compute_file_digest(path: Path | None) -> str | None:
    """Returns a short SHA-256 hex digest for a local dataset file without copying data."""
    if path is None or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:8]


def log_run_metadata(
    train_data_path: str | Path,
    base_dataset: Any,
    mode: str,
    prompts_config_path: str | Path,
    classes: Sequence[str],
    multi_label: bool,
    train_dataset: Any | None = None,
    eval_dataset: Any | None = None,
) -> None:
    """Logs dataset inputs (with Profile & Schema, no data copy), params, and prompt.json to MLflow."""
    import mlflow

    dataset_name = Path(train_data_path).resolve().name
    task_type = "multi_label" if multi_label else "single_label"
    train_count = len(train_dataset) if train_dataset is not None else len(base_dataset)

    try:
        from mlflow.data.meta_dataset import MetaDataset
        from mlflow.data.sources import LocalArtifactDatasetSource  # type: ignore[attr-defined]
        from mlflow.types.schema import ColSpec, Schema

        class _ProfiledMetaDataset(MetaDataset):
            def __init__(self, profile_data: dict[str, Any] | None = None, **kwargs: Any) -> None:
                self._profile = profile_data
                super().__init__(**kwargs)

            @property
            def profile(self) -> dict[str, Any] | None:
                return self._profile

            def to_dict(self) -> dict[str, str]:
                config = super().to_dict()
                if self._profile is not None:
                    config["profile"] = json.dumps(self._profile)
                return config

        schema = Schema([ColSpec("string", "text"), ColSpec("string", "labels")])
        train_source = Path(getattr(base_dataset, "data_path", None) or train_data_path)
        common_meta = {"num_classes": len(classes), "task_type": task_type}

        train_profile = {"num_rows": train_count, **common_meta}
        mlflow.log_input(
            _ProfiledMetaDataset(
                profile_data=train_profile,
                source=LocalArtifactDatasetSource(str(train_source)),
                name=dataset_name,
                digest=_compute_file_digest(train_source),
                schema=schema,
            ),
            context="training",
            tags={k: str(v) for k, v in train_profile.items()},
        )

        if eval_dataset is not None:
            val_source = Path(getattr(eval_dataset, "data_path", None) or train_source)
            is_explicit = val_source.resolve() != train_source.resolve()
            val_name = (
                f"{dataset_name}/{val_source.name}"
                if (is_explicit and Path(train_data_path).is_dir())
                else (val_source.name if is_explicit else f"{dataset_name} (val split)")
            )
            val_profile = {
                "num_rows": len(eval_dataset),
                "split_strategy": "explicit_file" if is_explicit else "random_split",
                **common_meta,
            }
            mlflow.log_input(
                _ProfiledMetaDataset(
                    profile_data=val_profile,
                    source=LocalArtifactDatasetSource(str(val_source)),
                    name=val_name,
                    digest=_compute_file_digest(val_source),
                    schema=schema,
                ),
                context="validation",
                tags={k: str(v) for k, v in val_profile.items()},
            )
    except Exception as e:
        logger.warning("Failed logging MLflow dataset input '%s': %s", dataset_name, e)

    try:
        mlflow.log_params({"dataset": dataset_name, "task_type": task_type})
    except Exception as e:
        logger.warning("Failed logging MLflow dataset params: %s", e)

    try:
        mlflow.log_dict(
            {
                "mode": mode,
                "prompts_config_path": str(prompts_config_path),
                "system_prompt": base_dataset.system_prompt,
                "user_prompt_template": base_dataset.user_prompt_template,
                "prompt_template": base_dataset.prompt_template,
                "classes": list(classes),
                "multi_label": multi_label,
                "example_rendered_prompt": base_dataset.build_prompt("<TEXT>"),
            },
            "prompt.json",
        )
    except Exception as e:
        logger.warning("Failed logging MLflow 'prompt.json' artifact: %s", e)
