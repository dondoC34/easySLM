"""MLflow experiment queries, diagnostics, and storage management module for easySLM.

This module provides high-level utilities to manage the local MLflow environment:
- Resolving storage filesystem paths for the SQLite backend and artifact directories.
- Querying server health, storage size, and run diagnostics (`get_mlflow_status`).
- Searching, filtering, and tabulating recent runs (`list_mlflow_runs`).
- Discovering and extracting the best performing model checkpoint (`get_best_mlflow_run`).
- Safely wiping SQLite database records and artifact directories (`clear_mlflow_data`).
- Exporting artifacts and model files for a given run ID to an external folder (`export_run_artifacts`).
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

from src.tracking.config import get_mlflow_config
from src.tracking.server import check_mlflow_server_online

logger = logging.getLogger("easySLM.tracking.manager")


def resolve_storage_paths(config_path: str | Path | None = None) -> tuple[Path | None, Path]:
    """Resolves the local filesystem paths for the SQLite database and default artifact root.

    Args:
        config_path: Optional custom path to YAML config file.

    Returns:
        tuple[Path | None, Path]: A pair (db_path, artifact_dir).
    """
    cfg = get_mlflow_config(config_path)
    server_cfg = cfg.get("server", {})
    backend_uri = server_cfg.get("backend_store_uri", "sqlite:///mlflow.db")
    artifact_root = server_cfg.get("default_artifact_root", "./mlruns")

    db_path: Path | None = None
    if backend_uri.startswith("sqlite:///"):
        db_raw = backend_uri[len("sqlite:///") :]
        db_path = Path(db_raw).resolve()

    artifact_path = Path(artifact_root).resolve()
    return db_path, artifact_path


def artifact_uri_to_path(artifact_uri: str) -> Path:
    """Converts a file:// URI or local filesystem path string to a Path object."""
    if artifact_uri.startswith("file:"):
        parsed = urlparse(artifact_uri)
        return Path(url2pathname(parsed.path))
    return Path(artifact_uri)


def clear_mlflow_data(
    config_path: str | Path | None = None,
    artifacts_only: bool = False,
) -> tuple[bool, str]:
    """Cleans local MLflow storage by deleting artifacts and SQLite database records.

    Args:
        config_path: Optional custom path to YAML config file.
        artifacts_only: When True, only clears the artifacts folder and leaves SQLite DB intact.

    Returns:
        tuple[bool, str]: Status flag and human-readable confirmation or error message.
    """
    db_path, artifact_path = resolve_storage_paths(config_path)
    deleted_items: list[str] = []

    # 1. Clean artifacts directory
    if artifact_path.exists():
        try:
            for item in artifact_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            deleted_items.append(f"artifact directory '{artifact_path}'")
        except Exception as e:
            logger.error("Failed cleaning artifact directory '%s': %s", artifact_path, e)
            return False, f"Failed cleaning artifact directory '{artifact_path}': {e}"
    else:
        artifact_path.mkdir(parents=True, exist_ok=True)

    # 2. Clean SQLite database file and auxiliary journal/WAL files
    if not artifacts_only and db_path is not None:
        if db_path.exists():
            try:
                db_path.unlink()
                deleted_items.append(f"database file '{db_path.name}'")
            except PermissionError:
                msg = (
                    f"Database file '{db_path}' is currently locked by another active process. "
                    "If the MLflow tracking server is currently running, please stop it first."
                )
                logger.error(msg)
                return False, msg
            except Exception as e:
                logger.error("Failed deleting database file '%s': %s", db_path, e)
                return False, f"Failed deleting database file '{db_path}': {e}"

        # Clean auxiliary SQLite write-ahead log files
        for suffix in ("-wal", "-shm"):
            aux_file = db_path.with_name(db_path.name + suffix)
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except Exception:
                    pass

    if not deleted_items:
        return True, "MLflow storage is already empty."
    return True, f"Successfully cleared {', '.join(deleted_items)}."


def get_mlflow_status(
    config_path: str | Path | None = None,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    """Collects diagnostic metrics on server availability and storage usage.

    Args:
        config_path: Optional path to YAML config file.
        host: Optional server host override to probe.
        port: Optional server port override to probe.

    Returns:
        dict[str, Any]: Structured dictionary with server health, storage stats, and run counts.
    """
    cfg = get_mlflow_config(config_path)
    server_cfg = cfg.get("server", {})
    resolved_host = host or server_cfg.get("host", "127.0.0.1")
    resolved_port = int(port or server_cfg.get("port", 5000))
    backend_uri = server_cfg.get("backend_store_uri", "sqlite:///mlflow.db")
    db_path, artifact_path = resolve_storage_paths(config_path)

    server_online = check_mlflow_server_online(resolved_host, resolved_port)

    db_exists = db_path.exists() if db_path else False
    db_size = db_path.stat().st_size if db_exists and db_path else 0

    artifact_exists = artifact_path.exists()
    artifact_files = list(artifact_path.rglob("*")) if artifact_exists else []
    artifact_file_count = sum(1 for f in artifact_files if f.is_file())
    artifact_total_size = sum(f.stat().st_size for f in artifact_files if f.is_file())

    experiment_count = 0
    total_runs = 0
    runs_by_status: dict[str, int] = {}
    best_f1_by_experiment: dict[str, tuple[float, str]] = {}
    best_loss_by_experiment: dict[str, tuple[float, str]] = {}

    if db_exists or not backend_uri.startswith("sqlite:///"):
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=backend_uri)
            experiments = client.search_experiments()
            experiment_count = len(experiments)

            exp_ids = [e.experiment_id for e in experiments]
            if exp_ids:
                all_runs = client.search_runs(experiment_ids=exp_ids, max_results=5000)
                total_runs = len(all_runs)
                for r in all_runs:
                    status = r.info.status
                    runs_by_status[status] = runs_by_status.get(status, 0) + 1
                    exp_name = next(
                        (e.name for e in experiments if e.experiment_id == r.info.experiment_id),
                        r.info.experiment_id,
                    )
                    f1 = r.data.metrics.get("eval_f1_macro")
                    if f1 is not None:
                        curr_f1 = best_f1_by_experiment.get(exp_name)
                        if curr_f1 is None or f1 > curr_f1[0]:
                            best_f1_by_experiment[exp_name] = (f1, r.info.run_id[:8])

                    loss = r.data.metrics.get("eval_loss") or r.data.metrics.get("train_loss")
                    if loss is not None:
                        curr_loss = best_loss_by_experiment.get(exp_name)
                        if curr_loss is None or loss < curr_loss[0]:
                            best_loss_by_experiment[exp_name] = (loss, r.info.run_id[:8])
        except Exception as e:
            logger.debug("Failed querying MlflowClient for status diagnostics: %s", e)

    return {
        "host": resolved_host,
        "port": resolved_port,
        "server_online": server_online,
        "backend_store_uri": backend_uri,
        "db_path": str(db_path) if db_path else None,
        "db_exists": db_exists,
        "db_size_bytes": db_size,
        "artifact_dir": str(artifact_path),
        "artifact_exists": artifact_exists,
        "artifact_file_count": artifact_file_count,
        "artifact_size_bytes": artifact_total_size,
        "experiment_count": experiment_count,
        "total_runs": total_runs,
        "runs_by_status": runs_by_status,
        "best_f1_by_experiment": best_f1_by_experiment,
        "best_loss_by_experiment": best_loss_by_experiment,
    }


def _extract_model_name(params: dict[str, str], tags: dict[str, str]) -> str:
    """Extracts a readable SLM model identifier from params and tags."""
    if "model_name" in params:
        return params["model_name"]
    raw_path = params.get("_name_or_path")
    if raw_path:
        base = Path(raw_path).name
        if "--" in base:
            return base.replace("--", "/")
        return base
    run_name = params.get("run_name") or tags.get("mlflow.runName")
    if run_name:
        for prefix in ("easyslm-cls-", "easyslm-json-", "easyslm-"):
            if run_name.startswith(prefix):
                return run_name[len(prefix) :]
        return run_name
    return "-"


def _extract_mode(params: dict[str, str], tags: dict[str, str]) -> str:
    """Extracts the architecture mode ('cls' or 'json')."""
    if "mode" in params:
        return params["mode"]
    run_name = params.get("run_name") or tags.get("mlflow.runName") or ""
    if "cls" in run_name.lower():
        return "cls"
    if "json" in run_name.lower():
        return "json"
    return "-"


def list_mlflow_runs(
    config_path: str | Path | None = None,
    experiment_name: str | None = None,
    model: str | None = None,
    limit: int = 10,
    sort_by: str = "date",
) -> list[dict[str, Any]]:
    """Retrieves and formats a structured list of recent MLflow runs for easySLM.

    Args:
        config_path: Optional path to YAML config file.
        experiment_name: Optional filter for a specific experiment name.
        model: Optional filter for model architecture (substring match).
        limit: Maximum number of run records to retrieve.
        sort_by: Sort strategy: 'date' (most recent first), 'loss' (lowest first), or 'f1' (highest first).

    Returns:
        list[dict[str, Any]]: List of dictionary summaries per matching run.
    """
    cfg = get_mlflow_config(config_path)
    backend_uri = cfg.get("server", {}).get("backend_store_uri", "sqlite:///mlflow.db")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=backend_uri)
    try:
        experiments = client.search_experiments()
    except Exception as e:
        logger.debug("No experiments found or failed connecting: %s", e)
        return []

    if not experiments:
        return []

    exp_map = {e.experiment_id: e.name for e in experiments}
    target_exp_ids = [
        e.experiment_id
        for e in experiments
        if not experiment_name
        or e.name.lower() == experiment_name.lower()
        or experiment_name.lower() in e.name.lower()
    ]
    if not target_exp_ids:
        return []

    if sort_by == "loss":
        order_by = ["metrics.eval_loss ASC"]
    elif sort_by == "f1":
        order_by = ["metrics.eval_f1_macro DESC"]
    else:
        order_by = ["attributes.start_time DESC"]

    try:
        runs = client.search_runs(
            experiment_ids=target_exp_ids,
            order_by=order_by,
            max_results=limit * 4 if model else limit,
        )
    except Exception as e:
        logger.debug("Failed querying runs: %s", e)
        return []

    results: list[dict[str, Any]] = []
    for r in runs:
        run_model = _extract_model_name(r.data.params, r.data.tags)
        if model and model.lower() not in run_model.lower():
            continue

        mode = _extract_mode(r.data.params, r.data.tags)
        eval_loss = r.data.metrics.get("eval_loss")
        eval_f1 = r.data.metrics.get("eval_f1_macro")
        train_loss = r.data.metrics.get("train_loss") or r.data.metrics.get("loss")
        epochs = r.data.params.get("num_train_epochs") or str(r.data.metrics.get("epoch", "-"))

        start_ts = r.info.start_time
        date_str = (
            datetime.fromtimestamp(start_ts / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
            if start_ts
            else "-"
        )

        results.append(
            {
                "run_id": r.info.run_id,
                "run_name": r.data.params.get("run_name") or r.data.tags.get("mlflow.runName", "-"),
                "experiment_name": exp_map.get(r.info.experiment_id, r.info.experiment_id),
                "model": run_model,
                "mode": mode,
                "epochs": epochs,
                "eval_loss": f"{eval_loss:.4f}" if eval_loss is not None else "-",
                "eval_f1": f"{eval_f1:.4f}" if eval_f1 is not None else "-",
                "train_loss": f"{train_loss:.4f}" if train_loss is not None else "-",
                "status": r.info.status,
                "created_at": date_str,
                "artifact_uri": r.info.artifact_uri,
            }
        )
        if len(results) >= limit:
            break

    return results


def get_best_mlflow_run(
    config_path: str | Path | None = None,
    experiment_name: str | None = None,
    model: str | None = None,
    metric: str = "eval_f1_macro",
) -> dict[str, Any] | None:
    """Finds the run with the best performance on the given metric.

    If metric contains 'loss', looks for the lowest score. Otherwise, looks for the highest score.

    Args:
        config_path: Optional path to YAML config file.
        experiment_name: Optional filter for a specific experiment name.
        model: Optional filter for model architecture or name.
        metric: Metric name to rank by (default: 'eval_f1_macro').

    Returns:
        dict[str, Any] | None: Detailed dictionary containing parameters, metrics, and artifact paths.
    """
    cfg = get_mlflow_config(config_path)
    backend_uri = cfg.get("server", {}).get("backend_store_uri", "sqlite:///mlflow.db")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=backend_uri)
    try:
        experiments = client.search_experiments()
    except Exception as e:
        logger.debug("Failed querying experiments: %s", e)
        return None

    target_exp_ids = [
        e.experiment_id
        for e in experiments
        if not experiment_name
        or e.name.lower() == experiment_name.lower()
        or experiment_name.lower() in e.name.lower()
    ]
    if not target_exp_ids:
        return None

    is_loss = "loss" in metric.lower()
    sort_order = "ASC" if is_loss else "DESC"
    order_by = [f"metrics.{metric} {sort_order}"]

    try:
        runs = client.search_runs(
            experiment_ids=target_exp_ids,
            order_by=order_by,
            max_results=50 if model else 1,
        )
    except Exception as e:
        logger.debug("Failed searching best run by metric '%s': %s", metric, e)
        return None

    best_r = None
    for r in runs:
        val = r.data.metrics.get(metric)
        if val is None:
            continue
        run_model = _extract_model_name(r.data.params, r.data.tags)
        if model and model.lower() not in run_model.lower():
            continue
        best_r = r
        break

    if not best_r:
        return None

    exp_map = {e.experiment_id: e.name for e in experiments}
    start_ts = best_r.info.start_time
    date_str = (
        datetime.fromtimestamp(start_ts / 1000.0).strftime("%Y-%m-%d %H:%M:%S") if start_ts else "-"
    )

    art_path = artifact_uri_to_path(best_r.info.artifact_uri)
    return {
        "run_id": best_r.info.run_id,
        "run_name": best_r.data.params.get("run_name")
        or best_r.data.tags.get("mlflow.runName", "-"),
        "experiment_name": exp_map.get(best_r.info.experiment_id, best_r.info.experiment_id),
        "model": _extract_model_name(best_r.data.params, best_r.data.tags),
        "mode": _extract_mode(best_r.data.params, best_r.data.tags),
        "status": best_r.info.status,
        "created_at": date_str,
        "metric_name": metric,
        "metric_value": best_r.data.metrics.get(metric),
        "all_metrics": best_r.data.metrics,
        "params": best_r.data.params,
        "artifact_uri": best_r.info.artifact_uri,
        "artifacts_dir": str(art_path) if art_path.exists() else None,
    }


def export_run_artifacts(
    run_id: str,
    dest_dir: str | Path,
    config_path: str | Path | None = None,
) -> list[Path]:
    """Copies artifacts from a specified run ID to a local destination directory.

    Args:
        run_id: The MLflow Run ID string.
        dest_dir: Target local directory where files will be copied.
        config_path: Optional custom path to YAML config file.

    Returns:
        list[Path]: List of copied file paths.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    cfg = get_mlflow_config(config_path)
    backend_uri = cfg.get("server", {}).get("backend_store_uri", "sqlite:///mlflow.db")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=backend_uri)
    run = client.get_run(run_id)
    art_path = artifact_uri_to_path(run.info.artifact_uri)

    if not art_path.exists():
        raise FileNotFoundError(f"Artifact directory for run '{run_id}' not found at '{art_path}'.")

    copied: list[Path] = []
    for item in art_path.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied.append(target)
    return copied
