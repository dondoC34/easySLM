"""CLI Subcommand Controller for 'easyslm mlflow'.

Provides commands to start servers, inspect runs, display status, and wipe data.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from src.tracking.config import get_mlflow_config
from src.tracking.manager import (
    clear_mlflow_data,
    export_run_artifacts,
    get_best_mlflow_run,
    get_mlflow_status,
    list_mlflow_runs,
    resolve_storage_paths,
)
from src.tracking.server import start_mlflow_server

logger = logging.getLogger("easySLM.tracking.cli")


def format_cli_table(headers: list[str], rows: list[list[str]]) -> str:
    """Formats headers and string rows into a clean ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(col_widths)))
    row_lines = [
        " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(r)) for r in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def handle_mlflow_cli(args: Sequence[str]) -> int:
    """Subcommand controller for 'easyslm mlflow' commands:

    - start / run: Launch the tracking server daemon/subprocess
    - clear / clean: Clean SQLite database and/or artifact directories
    - status: Probe server health and storage usage metrics
    - list / runs: Display formatted table of runs and metrics
    - best: Display best run by metric (default: eval_f1_macro) and optionally export assets
    """
    mlflow_parser = argparse.ArgumentParser(
        prog="easyslm mlflow",
        description="MLflow experiment and tracking management for easySLM.",
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to mlflow-config.yml).",
    )

    subparsers = mlflow_parser.add_subparsers(dest="subcommand", required=True)

    # 1. start / run
    start_parser = subparsers.add_parser(
        "start",
        aliases=["run"],
        parents=[common_parser],
        help="Start the MLflow tracking server and web dashboard.",
    )
    start_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host IP address to bind to (overrides config).",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to bind to (overrides config).",
    )
    start_parser.add_argument(
        "--backend-store-uri",
        type=str,
        default=None,
        help="Storage database URI (e.g. sqlite:///mlflow.db, overrides config).",
    )
    start_parser.add_argument(
        "--default-artifact-root",
        type=str,
        default=None,
        help="Directory where artifacts are stored (overrides config).",
    )

    # 2. clear / clean
    clear_parser = subparsers.add_parser(
        "clear",
        aliases=["clean"],
        parents=[common_parser],
        help="Clear MLflow tracking database records and artifact files.",
    )
    clear_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Confirm deletion automatically without interactive prompt.",
    )
    clear_parser.add_argument(
        "--artifacts-only",
        action="store_true",
        default=False,
        help="Only delete artifact files, leaving the SQLite database intact.",
    )

    # 3. status
    status_parser = subparsers.add_parser(
        "status",
        parents=[common_parser],
        help="Display MLflow server status, database size, and storage diagnostics.",
    )
    status_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host IP address to probe (overrides config).",
    )
    status_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to probe (overrides config).",
    )

    # 4. list / runs
    list_parser = subparsers.add_parser(
        "list",
        aliases=["runs"],
        parents=[common_parser],
        help="List recent MLflow runs with metrics, architecture, and status.",
    )
    list_parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Filter runs by experiment name (e.g. 'easySLM_finetuning').",
    )
    list_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter runs by model name (e.g. 'Qwen', 'SmolLM').",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of runs to display (default: 10).",
    )
    list_parser.add_argument(
        "--sort-by",
        type=str,
        choices=["date", "loss", "f1"],
        default="date",
        help="Sort runs by 'date' (most recent first), 'loss' (lowest first), or 'f1' (highest first).",
    )

    # 5. best
    best_parser = subparsers.add_parser(
        "best",
        parents=[common_parser],
        help="Display details of the best performing run and export assets.",
    )
    best_parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Filter by experiment name.",
    )
    best_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter by model name.",
    )
    best_parser.add_argument(
        "--metric",
        type=str,
        default="eval_f1_macro",
        help="Metric to rank by (default: 'eval_f1_macro').",
    )
    best_parser.add_argument(
        "--export-to",
        type=str,
        default=None,
        help="Optional directory to copy the best run's artifacts.",
    )

    parsed = mlflow_parser.parse_args(args)

    # ---------------- Subcommand Dispatchers ----------------

    # Command: start / run
    if parsed.subcommand in ("start", "run"):
        cfg = get_mlflow_config(parsed.config)
        server_cfg = cfg.get("server", {})
        host = parsed.host or server_cfg.get("host", "127.0.0.1")
        port = parsed.port or server_cfg.get("port", 5000)
        backend_store_uri = parsed.backend_store_uri or server_cfg.get(
            "backend_store_uri", "sqlite:///mlflow.db"
        )
        default_artifact_root = parsed.default_artifact_root or server_cfg.get(
            "default_artifact_root", "./mlruns"
        )

        return start_mlflow_server(
            host=host,
            port=port,
            backend_store_uri=backend_store_uri,
            default_artifact_root=default_artifact_root,
        )

    # Command: clear / clean
    if parsed.subcommand in ("clear", "clean"):
        db_path, artifact_path = resolve_storage_paths(parsed.config)
        db_size = db_path.stat().st_size if db_path and db_path.exists() else 0
        art_size = (
            sum(f.stat().st_size for f in artifact_path.rglob("*") if f.is_file())
            if artifact_path.exists()
            else 0
        )

        print("\n" + "=" * 65)
        print("easySLM MLflow: Storage Cleanup")
        print("=" * 65)
        if not parsed.artifacts_only and db_path:
            print(f"Target Database : {db_path} ({db_size / 1024:.1f} KB)")
        print(f"Target Artifacts: {artifact_path} ({art_size / (1024 * 1024):.2f} MB)")
        print("=" * 65)

        if not parsed.yes:
            target_desc = (
                "all artifacts"
                if parsed.artifacts_only
                else "all MLflow runs, metrics, and artifacts"
            )
            try:
                confirm = (
                    input(f"Are you sure you want to permanently delete {target_desc}? [y/N]: ")
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user.")
                return 1
            if confirm not in ("y", "yes"):
                print("Cleanup aborted by user.\n")
                return 0

        success, msg = clear_mlflow_data(parsed.config, artifacts_only=parsed.artifacts_only)
        if success:
            logger.info("MLflow storage cleanup: %s", msg)
            print(f"[easySLM] {msg}\n")
            return 0
        logger.error("MLflow storage cleanup failed: %s", msg)
        print(f"[easySLM ERROR] {msg}\n")
        return 1

    # Command: status
    if parsed.subcommand == "status":
        stats = get_mlflow_status(parsed.config, host=parsed.host, port=parsed.port)
        print("\n" + "=" * 65)
        print("easySLM MLflow: System & Storage Status")
        print("=" * 65)
        server_badge = "[ONLINE]" if stats["server_online"] else "[OFFLINE]"
        print(f"Dashboard URL       : http://{stats['host']}:{stats['port']} {server_badge}")
        print(f"Backend Store URI   : {stats['backend_store_uri']}")
        if stats["db_path"]:
            db_status = (
                f"Exists ({stats['db_size_bytes'] / 1024:.1f} KB)"
                if stats["db_exists"]
                else "Not created yet"
            )
            print(f"Database File       : {stats['db_path']} -> {db_status}")
        art_status = (
            f"Exists ({stats['artifact_file_count']} files, {stats['artifact_size_bytes'] / (1024 * 1024):.2f} MB)"
            if stats["artifact_exists"]
            else "Not created yet"
        )
        print(f"Artifact Directory  : {stats['artifact_dir']} -> {art_status}")
        print("-" * 65)
        print(f"Total Experiments   : {stats['experiment_count']}")
        print(f"Total Runs Recorded : {stats['total_runs']}")
        if stats["runs_by_status"]:
            status_summary = ", ".join(f"{k}: {v}" for k, v in stats["runs_by_status"].items())
            print(f"Runs by Status      : {status_summary}")
        if stats.get("best_f1_by_experiment"):
            print("Best F1 Macro by Exp:")
            for exp, (bf1, run_id) in stats["best_f1_by_experiment"].items():
                print(f"  * {exp.ljust(22)}: {bf1:.4f} (Run: {run_id})")
        if stats.get("best_loss_by_experiment"):
            print("Best Loss Recorded  :")
            for exp, (bloss, run_id) in stats["best_loss_by_experiment"].items():
                print(f"  * {exp.ljust(22)}: {bloss:.4f} (Run: {run_id})")
        print("=" * 65 + "\n")
        return 0

    # Command: list / runs
    if parsed.subcommand in ("list", "runs"):
        runs = list_mlflow_runs(
            config_path=parsed.config,
            experiment_name=parsed.experiment,
            model=parsed.model,
            limit=parsed.limit,
            sort_by=parsed.sort_by,
        )
        if not runs:
            print("\n[easySLM] No MLflow runs found matching specified criteria.\n")
            return 0

        print("\n" + "=" * 90)
        print(f"easySLM MLflow Runs (Showing top {len(runs)}, sorted by {parsed.sort_by}):")
        print("=" * 90)
        headers = [
            "Run ID",
            "Model",
            "Mode",
            "Epochs",
            "Eval Loss",
            "F1 Macro",
            "Train Loss",
            "Status",
            "Created At",
        ]
        table_rows = [
            [
                r["run_id"][:8],
                r["model"][:22],
                r["mode"],
                str(r["epochs"]),
                str(r["eval_loss"]),
                str(r["eval_f1"]),
                str(r["train_loss"]),
                r["status"],
                r["created_at"],
            ]
            for r in runs
        ]
        print(format_cli_table(headers, table_rows))
        print("=" * 90 + "\n")
        return 0

    # Command: best
    if parsed.subcommand == "best":
        best_run = get_best_mlflow_run(
            config_path=parsed.config,
            experiment_name=parsed.experiment,
            model=parsed.model,
            metric=parsed.metric,
        )
        if not best_run:
            print(f"\n[easySLM] No runs with recorded metric '{parsed.metric}' found.\n")
            return 0

        print("\n" + "=" * 65)
        print(f"easySLM MLflow: Best Performing Run ({parsed.metric})")
        print("=" * 65)
        print(f"Run ID        : {best_run['run_id']}")
        print(f"Run Name      : {best_run['run_name']}")
        print(f"Experiment    : {best_run['experiment_name']}")
        print(f"Model         : {best_run['model']}")
        print(f"Mode          : {best_run['mode']}")
        print(f"Metric ({best_run['metric_name']}): {best_run['metric_value']}")
        print(f"Status        : {best_run['status']}")
        print(f"Created At    : {best_run['created_at']}")
        print("-" * 65)
        print("Key Metrics:")
        for mk in ("eval_loss", "eval_f1_macro", "eval_accuracy", "train_loss"):
            if mk in best_run["all_metrics"]:
                print(f"  * {mk.ljust(18)}: {best_run['all_metrics'][mk]}")
        if best_run.get("artifacts_dir"):
            print(f"Artifacts Dir : {best_run['artifacts_dir']}")

        if parsed.export_to:
            try:
                copied = export_run_artifacts(best_run["run_id"], parsed.export_to, parsed.config)
                print("-" * 65)
                print(f"Exported {len(copied)} artifact file(s) to '{parsed.export_to}':")
                for f in copied:
                    print(f"  -> {f.name}")
            except Exception as e:
                print(f"[easySLM ERROR] Failed exporting artifacts: {e}")
                return 1

        print("=" * 65 + "\n")
        return 0

    return 0
