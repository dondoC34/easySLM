import argparse
import logging
import sys
from typing import Any

from src.fine_tuning import FineTuningConfig, SLMFineTuner
from src.slm import SLMBase
from src.tracking.cli import handle_mlflow_cli
from src.utils.device import print_cuda_info
from src.utils.help_tables import (
    get_combined_help_tables,
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
        default="data/sample_news.txt",
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
        default="configs/training_config.yml",
        help="Path to training configuration YAML file",
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
        default="./models/finetuned",
        help="Directory to search for model cards when --model is not specified",
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
        default="data/sample_news_validation.txt",
        help="Path to .txt file or folder containing .txt files to predict on",
    )
    pred_parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Directory where predictions will be saved (.txt and .json)",
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

    logger.info(
        "Initializing fine-tuning with config '%s' and model '%s'...", args.config, args.model
    )
    config = FineTuningConfig.from_yaml(args.config, **overrides)
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
        logger.info(
            "No --model specified. Searching for best '%s' model in '%s' (ranking by '%s')...",
            args.mode,
            args.models_dir,
            args.metric,
        )
        model = SLMBase.from_best_model(
            models_dir=args.models_dir,
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

    logger.info("Predicting on data: '%s'...", args.data)
    output_files = model.predict_file_or_dir(
        data_path=args.data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    logger.info("Predictions successfully written to:")
    for f in output_files:
        logger.info("  - %s", f)


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
    elif args.subcommand == "info":
        logger.info("=== easySLM System Information ===")
        print_cuda_info()
    else:
        print(get_combined_help_tables())


if __name__ == "__main__":
    main()
