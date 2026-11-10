"""MLflow tracking, experiment monitoring, and server lifecycle package for easySLM.

This package provides a modular architecture for MLflow experiment tracking:
- `config`: Configuration loading and default settings management.
- `server`: Server lifecycle controls (process launching and port health checks).
- `manager`: Storage inspection, run query tables, best run discovery, and cleanup.
- `plotting`: Diagnostic Matplotlib visualization generators (confusion matrices, heatmaps, cardinality).
- `tables`: Tabular evaluation artifacts for interactive MLflow UI tables inspection.
- `artifacts`: High-level evaluation artifact logging coordinator.
- `cli`: Subcommand handler for `easyslm mlflow`.
"""

from __future__ import annotations

from src.tracking.artifacts import log_evaluation_artifacts
from src.tracking.config import DEFAULT_CONFIG, get_mlflow_config, load_mlflow_config, mlflow_config
from src.tracking.manager import (
    artifact_uri_to_path,
    clear_mlflow_data,
    export_run_artifacts,
    get_best_mlflow_run,
    get_mlflow_status,
    list_mlflow_runs,
    resolve_storage_paths,
)
from src.tracking.plotting import (
    create_cross_label_confusion_matrix_figure,
    create_label_cardinality_figure,
    create_per_class_metrics_figure,
    create_single_label_confusion_matrix_figure,
)
from src.tracking.server import (
    check_mlflow_server_online,
    start_mlflow_server,
)
from src.tracking.tables import (
    create_cross_label_confusion_matrix_table,
    create_label_cardinality_table,
    create_per_class_metrics_table,
    create_single_label_confusion_matrix_table,
)

__all__ = [
    "DEFAULT_CONFIG",
    "artifact_uri_to_path",
    "check_mlflow_server_online",
    "clear_mlflow_data",
    "create_cross_label_confusion_matrix_figure",
    "create_cross_label_confusion_matrix_table",
    "create_label_cardinality_figure",
    "create_label_cardinality_table",
    "create_per_class_metrics_figure",
    "create_per_class_metrics_table",
    "create_single_label_confusion_matrix_figure",
    "create_single_label_confusion_matrix_table",
    "export_run_artifacts",
    "get_best_mlflow_run",
    "get_mlflow_config",
    "get_mlflow_status",
    "list_mlflow_runs",
    "load_mlflow_config",
    "log_evaluation_artifacts",
    "mlflow_config",
    "resolve_storage_paths",
    "start_mlflow_server",
]
