import textwrap

FTUNE_OPTIONS: list[tuple[str, str, str]] = [
    (
        "--model",
        "str (default: 'Qwen/Qwen2.5-0.5B')",
        "Hugging Face model ID. Examples: 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'meta-llama/Llama-3.2-1B', 'google/gemma-2-2b'.",
    ),
    (
        "--type",
        "'cls' | 'json' (default: 'cls')",
        "Classification mode: 'cls' (dense linear head) or 'json' (generative list).",
    ),
    (
        "--data",
        "path (default: 'data/sample_news.txt')",
        "Path to dataset file (.txt) or directory containing .txt files.",
    ),
    (
        "--val-data",
        "path (default: None)",
        "Optional path to separate validation dataset (.txt file or directory). If omitted, 15% is randomly split from training data.",
    ),
    (
        "--output-dir",
        "path (default: './models/finetuned')",
        "Directory where model checkpoints and Markdown Model Cards are saved.",
    ),
    (
        "--config",
        "path (default: 'configs/training_config.yml')",
        "Path to base YAML training configuration file.",
    ),
    (
        "--prompts-config",
        "path (default: 'configs/prompts.yml')",
        "Path to YAML prompts configuration file.",
    ),
    ("--epochs", "float (default: 3.0)", "Total training epochs (overrides config value)."),
    ("--batch-size", "int (default: 2)", "Per-device training batch size."),
    ("--eval-batch-size", "int (default: 4)", "Per-device validation batch size."),
    ("--lr", "float (default: 2e-4)", "Peak learning rate for AdamW optimizer."),
    (
        "--grad-accum",
        "int (default: 4)",
        "Gradient accumulation steps (effective batch = batch_size * grad_accum).",
    ),
    (
        "--max-length",
        "int (default: 512)",
        "Maximum sequence token length (truncates longer sequences).",
    ),
    (
        "--single-label",
        "flag (default: False)",
        "Enforces single-label classification (CrossEntropy + Softmax / argmax).",
    ),
    (
        "--pooling",
        "'eos'|'mean'|'max' (default: 'eos')",
        "Sequence pooling strategy in 'cls' mode.",
    ),
    (
        "--no-lora",
        "flag (default: False)",
        "Disables LoRA and performs full parameter fine-tuning.",
    ),
    ("--lora-r", "int (default: 8)", "Low-rank dimension (rank r) for LoRA adapters."),
    ("--lora-alpha", "int (default: 16)", "Scaling factor alpha for LoRA adapters."),
    ("--no-mlflow", "flag (default: False)", "Disables MLflow experiment tracking."),
    ("--run-name", "str (default: auto)", "Custom run name for MLflow and checkpoint directory."),
]

PREDICT_OPTIONS: list[tuple[str, str, str]] = [
    (
        "--model",
        "path (default: auto-best)",
        "Path to specific checkpoint. If omitted, picks best model from --models-dir.",
    ),
    (
        "--models-dir",
        "path (default: './models/finetuned')",
        "Directory searched for model cards when --model is omitted.",
    ),
    (
        "--type",
        "'cls' | 'json' (default: 'cls')",
        "Architecture mode to use or search for ('cls' or 'json').",
    ),
    (
        "--data",
        "path (default: 'data/sample_news_validation.txt')",
        "Path to .txt file or directory containing .txt lines to predict on.",
    ),
    (
        "--output-dir",
        "path (default: 'predictions')",
        "Directory where prediction results (.txt and .json) are saved.",
    ),
    (
        "--metric",
        "str (default: 'eval_f1_macro')",
        "Model card metric used to rank best model if --model is omitted.",
    ),
    ("--batch-size", "int (default: 8)", "Batch size during inference."),
    (
        "--prompts-config",
        "path (default: 'configs/prompts.yml')",
        "Path to YAML prompts configuration file.",
    ),
]

MODEL_EXAMPLES: list[tuple[str, str, str]] = [
    (
        "Qwen/Qwen2.5-0.5B",
        "0.5B (Default)",
        "Ultra-lightweight, multilingual (29+ langs), runs in < 2 GB VRAM.",
    ),
    (
        "Qwen/Qwen2.5-1.5B",
        "1.5B",
        "Higher capacity multilingual SLM, great accuracy/speed balance (< 4 GB VRAM).",
    ),
    (
        "meta-llama/Llama-3.2-1B",
        "1.2B",
        "Meta compact SLM, excellent English generalist with 128k context window.",
    ),
    (
        "google/gemma-2-2b",
        "2.6B",
        "Google Gemma 2 architecture with sliding-window attention (< 6 GB VRAM).",
    ),
    (
        "HuggingFaceTB/SmolLM2-1.7B",
        "1.7B",
        "High-efficiency reasoning SLM trained on curated synthetic data (< 4 GB VRAM).",
    ),
]

MLFLOW_COMMANDS: list[tuple[str, str, str]] = [
    (
        "start (run)",
        "easyslm mlflow start [OPTIONS]",
        "Start MLflow tracking server and Web UI dashboard (--host, --port, --backend-store-uri).",
    ),
    (
        "status",
        "easyslm mlflow status [OPTIONS]",
        "Probe server health, database size, and storage diagnostics (--host, --port).",
    ),
    (
        "list (runs)",
        "easyslm mlflow list [OPTIONS]",
        "List recent training runs with metrics, epochs, and status (--sort-by, --limit, --model).",
    ),
    (
        "best",
        "easyslm mlflow best [OPTIONS]",
        "Display details of best run by metric and optionally export artifacts (--metric, --export-to).",
    ),
    (
        "clear (clean)",
        "easyslm mlflow clear [OPTIONS]",
        "Clear MLflow database records and/or artifact files (-y, --artifacts-only).",
    ),
    (
        "ui (open)",
        "easyslm mlflow ui [OPTIONS]",
        "Open the MLflow Web dashboard in default browser.",
    ),
]


def render_table(
    title: str,
    subtitle: str,
    rows: list[tuple[str, str, str]],
    headers: tuple[str, str, str] = ("Parameter", "Type / Default", "Description"),
    w_param: int = 20,
    w_type: int = 30,
    w_desc: int = 50,
) -> str:
    """Renders a clean ASCII table for CLI parameter documentation (100% compatible across OS / code pages)."""
    h_sep = f"+{'-' * (w_param + 2)}+{'-' * (w_type + 2)}+{'-' * (w_desc + 2)}+"

    header_line = f"| {headers[0]:<{w_param}} | {headers[1]:<{w_type}} | {headers[2]:<{w_desc}} |"

    table_lines: list[str] = [
        "",
        f"=== {title.upper()} ===",
        subtitle,
        "",
        h_sep,
        header_line,
        h_sep,
    ]

    for i, (param, ptype, desc) in enumerate(rows):
        param_wrapped = textwrap.wrap(param, width=w_param) or [""]
        type_wrapped = textwrap.wrap(ptype, width=w_type) or [""]
        desc_wrapped = textwrap.wrap(desc, width=w_desc) or [""]

        num_sublines = max(len(param_wrapped), len(type_wrapped), len(desc_wrapped))
        for line_idx in range(num_sublines):
            p_chunk = param_wrapped[line_idx] if line_idx < len(param_wrapped) else ""
            t_chunk = type_wrapped[line_idx] if line_idx < len(type_wrapped) else ""
            d_chunk = desc_wrapped[line_idx] if line_idx < len(desc_wrapped) else ""
            table_lines.append(
                f"| {p_chunk:<{w_param}} | {t_chunk:<{w_type}} | {d_chunk:<{w_desc}} |"
            )

        if i < len(rows) - 1:
            table_lines.append(h_sep)

    table_lines.append(h_sep)
    table_lines.append("")
    return "\n".join(table_lines)


def get_model_examples_table() -> str:
    """Generates the formatted table for recommended base SLMs."""
    return render_table(
        title="RECOMMENDED BASE SLM MODELS (HUGGING FACE IDS)",
        subtitle="Popular Small Language Models compatible with easySLM (LoRA supported, runnable on consumer GPUs):",
        rows=MODEL_EXAMPLES,
        headers=("Model ID (Hugging Face)", "Size / VRAM", "Description"),
        w_param=28,
        w_type=18,
        w_desc=54,
    )


def get_ftune_help_table() -> str:
    """Generates the formatted table for `easyslm ftune`."""
    ftune_table = render_table(
        title="COMMAND: easyslm ftune",
        subtitle="Fine-tunes a Small Language Model (SLM) for Direct ('cls') or Generative ('json') classification.",
        rows=FTUNE_OPTIONS,
    )
    return ftune_table + "\n" + get_model_examples_table()


def get_predict_help_table() -> str:
    """Generates the formatted table for `easyslm predict`."""
    return render_table(
        title="COMMAND: easyslm predict",
        subtitle="Runs inference on text files. Automatically discovers the best saved model if --model is omitted.",
        rows=PREDICT_OPTIONS,
    )


def get_mlflow_help_table() -> str:
    """Generates the formatted table for `easyslm mlflow`."""
    return render_table(
        title="COMMAND: easyslm mlflow",
        subtitle="Manage MLflow experiment tracking server, runs, metrics, and storage.",
        rows=MLFLOW_COMMANDS,
        headers=("Subcommand", "Syntax", "Description"),
        w_param=16,
        w_type=34,
        w_desc=50,
    )


def get_combined_help_tables() -> str:
    """Generates the combined tables for ftune, predict, and mlflow."""
    banner = [
        "==========================================================================================================",
        "                                     easySLM CLI - PARAMETER REFERENCE                                    ",
        "==========================================================================================================",
        "Usage:",
        "  easyslm ftune [OPTIONS]      Run fine-tuning with PEFT (LoRA), HF Trainer, and MLflow",
        "  easyslm predict [OPTIONS]    Run batch inference (with optional automatic best-model selection)",
        "  easyslm mlflow [COMMAND]     Manage MLflow tracking server, runs, metrics, and storage",
        "  easyslm info                 Display CUDA and GPU system diagnostics",
        "",
        "Run 'easyslm <command> --help' to inspect an individual command.",
    ]
    return (
        "\n".join(banner)
        + "\n"
        + get_ftune_help_table()
        + "\n"
        + get_predict_help_table()
        + "\n"
        + get_mlflow_help_table()
    )
