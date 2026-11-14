"""MLflow tracking configuration management module for easySLM.

This module handles:
- Default configuration settings for the local MLflow SQLite backend and artifact store.
- Parsing and validation of custom YAML configuration files (e.g., 'mlflow-config.yml').
- Transparent cascading fallback to project defaults when no configuration is provided.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("easySLM.tracking.config")

# Default settings for the MLflow tracking environment
DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 5000,
        "backend_store_uri": "sqlite:///mlflow.db",
        "default_artifact_root": "./mlruns",
    },
    "tracking": {
        "experiment_prefix": "easySLM",
        "autolog_figures": True,
        "autolog_model": True,
        "system_metrics": True,
        "system_metrics_sampling_interval": 10,
    },
}


def load_mlflow_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads MLflow configuration from a YAML file or falls back to standard defaults.

    Resolution order:
    1. The explicit path supplied via `config_path` (if provided).
    2. 'mlflow-config.yml' in the current working directory.
    3. 'mlflow-config.yml' in the project repository root.
    4. Built-in `DEFAULT_CONFIG` fallback dictionary.

    Args:
        config_path: Optional file path or string pointing to a YAML configuration file.

    Returns:
        dict[str, Any]: Resolved configuration dictionary containing 'server' and 'tracking' blocks.
    """
    candidate_paths: list[Path] = []
    if config_path:
        candidate_paths.append(Path(config_path))
    else:
        candidate_paths.append(Path("mlflow-config.yml"))
        candidate_paths.append(Path(__file__).resolve().parent.parent.parent / "mlflow-config.yml")

    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    # Deep merge with DEFAULT_CONFIG to guarantee complete keys
                    cfg = {
                        "server": {**DEFAULT_CONFIG["server"], **data.get("server", {})},
                        "tracking": {**DEFAULT_CONFIG["tracking"], **data.get("tracking", {})},
                    }
                    logger.debug("Loaded MLflow config from '%s'.", p)
                    return cfg
            except Exception as e:
                logger.warning("Error reading config file '%s': %s. Using default config.", p, e)

    return DEFAULT_CONFIG.copy()


# Pre-loaded default configuration instance (loaded once on module import)
mlflow_config: dict[str, Any] = load_mlflow_config()


def get_mlflow_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Returns the pre-loaded `mlflow_config` singleton if `config_path` is None,

    or loads and returns configuration from the explicit custom path.
    """
    if config_path is None:
        return mlflow_config
    return load_mlflow_config(config_path)


def configure_mlflow_env(
    mlflow_cfg: dict[str, Any],
    tracking_uri_override: str | None = None,
    experiment_name_override: str | None = None,
) -> list[str]:
    """Configures MLflow environment variables and system metrics logging for Hugging Face `Trainer`.

    Args:
        mlflow_cfg: Dictionary loaded via `get_mlflow_config`.
        tracking_uri_override: Optional override for `MLFLOW_TRACKING_URI`.
        experiment_name_override: Optional override for `MLFLOW_EXPERIMENT_NAME`.

    Returns:
        `["mlflow"]` list suitable for `TrainingArguments(report_to=...)`.
    """
    server_cfg = mlflow_cfg.get("server", {})
    tracking_cfg = mlflow_cfg.get("tracking", {})

    tracking_uri = tracking_uri_override or server_cfg.get(
        "backend_store_uri", "sqlite:///mlflow.db"
    )
    experiment_name = (
        experiment_name_override or f"{tracking_cfg.get('experiment_prefix', 'easySLM')}_finetuning"
    )

    # Convert relative local directory paths into clean file:// URIs for MLflow
    if not tracking_uri.startswith(("http://", "https://", "sqlite://", "file:")):
        tracking_path = Path(tracking_uri).resolve()
        tracking_path.mkdir(parents=True, exist_ok=True)
        tracking_uri = tracking_path.as_uri()

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_FLATTEN_PARAMS"] = "True"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    enable_system_metrics = tracking_cfg.get("system_metrics", True)
    if enable_system_metrics:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        interval = str(tracking_cfg.get("system_metrics_sampling_interval", 10))
        os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = interval
        os.environ["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"] = "1"
        try:
            import mlflow

            mlflow.enable_system_metrics_logging()
        except Exception as e:
            logger.debug("Could not enable system metrics logging: %s", e)

    logger.info(
        "MLflow configured | URI: %s | Experiment: %s | System Metrics: %s",
        tracking_uri,
        experiment_name,
        enable_system_metrics,
    )
    return ["mlflow"]
