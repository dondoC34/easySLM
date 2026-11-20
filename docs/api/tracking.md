# MLflow Tracking & Plots

Local MLflow server lifecycle, experiment querying/cleanup, diagnostic plots, and structured evaluation tables.

---

## Configuration

::: src.tracking.config.get_default_mlflow_config

::: src.tracking.config.load_mlflow_config

::: src.tracking.config.get_mlflow_config

::: src.tracking.config.configure_mlflow_env

---

## Server Lifecycle

::: src.tracking.server.check_mlflow_server_online

::: src.tracking.server.start_mlflow_server

---

## Storage & Run Management

::: src.tracking.manager.resolve_storage_paths

::: src.tracking.manager.get_mlflow_status

::: src.tracking.manager.list_mlflow_runs

::: src.tracking.manager.get_best_mlflow_run

::: src.tracking.manager.export_run_artifacts

::: src.tracking.manager.clear_mlflow_data

---

## Artifacts, Plots & Tables

::: src.tracking.artifacts.log_evaluation_artifacts

::: src.tracking.artifacts.log_run_metadata

::: src.tracking.plotting.create_single_label_confusion_matrix_figure

::: src.tracking.plotting.create_cross_label_confusion_matrix_figure

::: src.tracking.plotting.create_per_class_metrics_figure

::: src.tracking.plotting.create_label_cardinality_figure

::: src.tracking.tables.create_per_class_metrics_table

::: src.tracking.tables.create_validation_errors_table
