import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.fine_tuning import FineTuningConfig, SLMFineTuner
from src.slm import SLMBase
from src.tracking.cli import handle_mlflow_cli
from src.utils.config import (
    get_default_finetuned_dir,
    get_default_predictions_dir,
    get_project_root,
)
from src.utils.device import print_cuda_info
from src.utils.help_tables import (
    get_combined_help_tables,
    get_docs_help_table,
    get_ftune_help_table,
    get_mlflow_help_table,
    get_predict_help_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easyslm.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easyslm",
        description="easySLM: Generic CLI for Fine-Tuning and Inference with Small Language Models",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    # --------------------------------------------------------------------------
    # Subcommand: ftune
    # --------------------------------------------------------------------------
    ftune_parser = subparsers.add_parser(
        "ftune",
        help="Fine-tune an SLM on a classification dataset",
    )
    ftune_parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model ID (e.g. 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'meta-llama/Llama-3.2-1B', 'google/gemma-2-2b')",
    )
    ftune_parser.add_argument(
        "--data",
        type=str,
        default="data/agnews_tiny",
        help="Path to training dataset (.txt file or folder containing train.txt and optional val.txt)",
    )
    ftune_parser.add_argument(
        "--type",
        dest="mode",
        type=str,
        choices=["cls", "json"],
        default=None,
        help="Classification mode override: 'cls' (linear classification head) or 'json' (generative list output)",
    )
    ftune_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to training configuration YAML file (defaults to FineTuningConfig built-in defaults)",
    )
    ftune_parser.add_argument(
        "--prompts-config",
        type=str,
        default=None,
        help="Optional override path to prompts configuration YAML file",
    )
    ftune_parser.add_argument(
        "--epochs",
        type=float,
        default=None,
        help="Optional override for number of training epochs",
    )

    # --------------------------------------------------------------------------
    # Subcommand: predict
    # --------------------------------------------------------------------------
    pred_parser = subparsers.add_parser(
        "predict",
        help="Run inference on text files with a fine-tuned model (or the best saved model)",
    )
    pred_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a specific model checkpoint. If omitted, automatically selects the best model from --models-dir",
    )
    pred_parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to search for model cards when --model is not specified (defaults to ~/.easyslm/finetuned)",
    )
    pred_parser.add_argument(
        "--type",
        dest="mode",
        type=str,
        choices=["cls", "json"],
        default="cls",
        help="Model type to use or search for ('cls' or 'json')",
    )
    pred_parser.add_argument(
        "--data",
        type=str,
        default="data/agnews_tiny/val.txt",
        help="Path to .txt file or folder containing .txt files to predict on",
    )
    pred_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where predictions will be saved (.txt and .json, defaults to ~/.easyslm/predictions)",
    )
    pred_parser.add_argument(
        "--metric",
        type=str,
        default="eval_f1_macro",
        help="Performance metric used to select best model if --model is omitted (e.g. 'eval_f1_macro', 'eval_exact_match_accuracy')",
    )
    pred_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size",
    )
    pred_parser.add_argument(
        "--prompts-config",
        type=str,
        default=None,
        help="Path to prompts configuration YAML file (defaults to checkpoint prompt_config.yml if present, else configs/prompts.yml)",
    )

    # --------------------------------------------------------------------------
    # Subcommand: mlflow
    # --------------------------------------------------------------------------
    subparsers.add_parser(
        "mlflow",
        help="Manage MLflow tracking server, runs, metrics, and storage",
    )

    # --------------------------------------------------------------------------
    # Subcommand: docs
    # --------------------------------------------------------------------------
    docs_parser = subparsers.add_parser(
        "docs",
        help="Serve local MkDocs documentation and open in browser",
    )
    docs_subparsers = docs_parser.add_subparsers(dest="docs_subcommand")
    docs_start_parser = docs_subparsers.add_parser(
        "start",
        aliases=["serve"],
        help="Start local MkDocs development server and open the browser",
    )
    docs_start_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP address to bind the docs server to (default: 127.0.0.1)",
    )
    docs_start_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to serve documentation on (default: 8000)",
    )
    docs_start_parser.add_argument(
        "--no-open",
        action="store_true",
        default=False,
        help="Do not automatically open the documentation URL in a web browser",
    )

    # --------------------------------------------------------------------------
    # Subcommand: info
    # --------------------------------------------------------------------------
    subparsers.add_parser("info", help="Print GPU / CUDA diagnostic information")

    return parser


def run_ftune(args: argparse.Namespace) -> None:
    overrides: dict[str, Any] = {}
    if args.mode is not None:
        overrides["mode"] = args.mode
    if args.prompts_config is not None:
        overrides["prompts_config_path"] = args.prompts_config
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs

    if args.config is not None:
        logger.info(
            "Initializing fine-tuning with config '%s' and model '%s'...", args.config, args.model
        )
        config = FineTuningConfig.from_yaml(args.config, **overrides)
    else:
        logger.info("Initializing fine-tuning with built-in defaults and model '%s'...", args.model)
        config = FineTuningConfig(**overrides)
    tuner = SLMFineTuner(model_name=args.model, config=config)
    results = tuner.train(train_data_path=args.data)
    logger.info(
        "Fine-tuning completed successfully! Checkpoint saved at: %s", results.get("checkpoint_dir")
    )


def run_predict(args: argparse.Namespace) -> None:
    if args.model is not None:
        logger.info("Loading specified model checkpoint from: %s", args.model)
        model = SLMBase.from_checkpoint(
            checkpoint_dir=args.model,
            prompts_config_path=args.prompts_config,
        )
    else:
        resolved_models_dir = args.models_dir or str(get_default_finetuned_dir())
        logger.info(
            "No --model specified. Searching for best '%s' model in '%s' (ranking by '%s')...",
            args.mode,
            resolved_models_dir,
            args.metric,
        )
        model = SLMBase.from_best_model(
            models_dir=resolved_models_dir,
            model_type=args.mode,
            metric=args.metric,
            prompts_config_path=args.prompts_config,
        )
        best_card = model.metadata.get("model_card", {})
        logger.info(
            "Loaded best model from: %s (as of %s)",
            model.metadata.get("checkpoint_dir", "N/A"),
            best_card.get("as_of", "N/A"),
        )

    resolved_output_dir = args.output_dir or str(get_default_predictions_dir())
    logger.info("Predicting on data: '%s'...", args.data)
    output_files = model.predict_file_or_dir(
        data_path=args.data,
        output_dir=resolved_output_dir,
        batch_size=args.batch_size,
    )
    logger.info("Predictions successfully written to:")
    for f in output_files:
        logger.info("  - %s", f)


def start_docs_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> int:
    """Starts the local MkDocs development server and optionally opens the browser."""
    if importlib.util.find_spec("mkdocs") is None:
        logger.error(
            "MkDocs is not installed. Install docs dependencies with: uv sync --group docs"
        )
        return 1

    config_file = Path("mkdocs.yml")
    if not config_file.is_file():
        config_file = get_project_root() / "mkdocs.yml"
    if not config_file.is_file():
        logger.error("Could not locate 'mkdocs.yml' in current directory or repository root.")
        return 1

    dev_addr = f"{host}:{port}"
    cmd = [
        sys.executable,
        "-m",
        "mkdocs",
        "serve",
        "-a",
        dev_addr,
        "-f",
        str(config_file),
    ]
    if open_browser:
        cmd.append("--open")

    url = f"http://{dev_addr}"
    logger.info("Starting easySLM documentation server at %s ...", url)
    print(f"\n[easySLM] Documentation running at: {url}")
    print("[easySLM] Press Ctrl+C to terminate the server.\n")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        logger.info("Documentation server stopped by user.")
        return 0


def run_docs(args: argparse.Namespace) -> int:
    if getattr(args, "docs_subcommand", None) in ("start", "serve"):
        return start_docs_server(
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
    print(get_docs_help_table())
    return 0


def main() -> None:
    args_list = sys.argv[1:]

    # Delegate 'mlflow' subcommand directly to tracking CLI handler
    if args_list and args_list[0] == "mlflow":
        if len(args_list) == 1:
            print(get_mlflow_help_table())
            sys.exit(0)
        sys.exit(handle_mlflow_cli(args_list[1:]))

    is_help = any(arg in ("-h", "--help", "help") for arg in args_list)

    if is_help:
        if "ftune" in args_list:
            print(get_ftune_help_table())
            sys.exit(0)
        elif "predict" in args_list:
            print(get_predict_help_table())
            sys.exit(0)
        elif "mlflow" in args_list:
            print(get_mlflow_help_table())
            sys.exit(0)
        elif "docs" in args_list:
            print(get_docs_help_table())
            sys.exit(0)
        else:
            print(get_combined_help_tables())
            sys.exit(0)

    # When no arguments are provided, print full reference table
    if not args_list:
        print(get_combined_help_tables())
        sys.exit(0)

    parser = build_parser()
    args = parser.parse_args()

    if args.subcommand == "ftune":
        run_ftune(args)
    elif args.subcommand == "predict":
        run_predict(args)
    elif args.subcommand == "mlflow":
        print(get_mlflow_help_table())
    elif args.subcommand == "docs":
        sys.exit(run_docs(args))
    elif args.subcommand == "info":
        logger.info("=== easySLM System Information ===")
        print_cuda_info()
    else:
        print(get_combined_help_tables())


if __name__ == "__main__":
    main()
