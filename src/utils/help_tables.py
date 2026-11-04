import textwrap

FTUNE_OPTIONS: list[tuple[str, str, str]] = [
    (
        "--model",
        "str (default: 'Qwen/Qwen2.5-0.5B')",
        "Hugging Face model ID or path to local model snapshot.",
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
    (
        "--chat-template",
        "path (default: 'configs/chat_template.yml')",
        "Path to YAML Jinja2 chat template configuration.",
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
    (
        "--chat-template",
        "path (default: 'configs/chat_template.yml')",
        "Path to YAML Jinja2 chat template configuration.",
    ),
]


def render_table(
    title: str,
    subtitle: str,
    rows: list[tuple[str, str, str]],
    w_param: int = 20,
    w_type: int = 30,
    w_desc: int = 50,
) -> str:
    """Renders a clean ASCII table for CLI parameter documentation (100% compatible across OS / code pages)."""
    h_sep = f"+{'-' * (w_param + 2)}+{'-' * (w_type + 2)}+{'-' * (w_desc + 2)}+"

    header_line = (
        f"| {'Parameter':<{w_param}} | {'Type / Default':<{w_type}} | {'Description':<{w_desc}} |"
    )

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
    return "\n".join(table_lines)


def get_ftune_help_table() -> str:
    """Generates the formatted table for `easyslm ftune`."""
    return render_table(
        title="COMMAND: easyslm ftune",
        subtitle="Fine-tunes a Small Language Model (SLM) for Direct ('cls') or Generative ('json') classification.",
        rows=FTUNE_OPTIONS,
    )


def get_predict_help_table() -> str:
    """Generates the formatted table for `easyslm predict`."""
    return render_table(
        title="COMMAND: easyslm predict",
        subtitle="Runs inference on text files. Automatically discovers the best saved model if --model is omitted.",
        rows=PREDICT_OPTIONS,
    )


def get_combined_help_tables() -> str:
    """Generates the combined tables for both `ftune` and `predict`."""
    banner = [
        "==========================================================================================================",
        "                                     easySLM CLI - PARAMETER REFERENCE                                    ",
        "==========================================================================================================",
        "Usage:",
        "  easyslm ftune [OPTIONS]      Run fine-tuning with PEFT (LoRA), HF Trainer, and MLflow",
        "  easyslm predict [OPTIONS]    Run batch inference (with optional automatic best-model selection)",
        "  easyslm info                 Display CUDA and GPU system diagnostics",
        "",
        "Run 'easyslm ftune --help' or 'easyslm predict --help' to inspect an individual command.",
    ]
    return "\n".join(banner) + "\n" + get_ftune_help_table() + "\n" + get_predict_help_table()
