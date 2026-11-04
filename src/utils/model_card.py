import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

METRIC_DESCRIPTIONS: dict[str, str] = {
    "eval_loss": "Validation set loss (CrossEntropy / BCE; lower is better)",
    "loss": "Training loss on mini-batches (lower is better)",
    "eval_f1_macro": "Unweighted average of F1 scores across classes (evaluates class balance)",
    "eval_f1_micro": "Global aggregate F1 score across all classes and predictions",
    "eval_f1_weighted": "F1 score weighted by the number of true instances (support) of each class",
    "eval_f1_samples": "Sample-averaged F1 score across multi-label instances",
    "eval_accuracy": "Standard top-1 classification accuracy on single-label multiclass predictions",
    "eval_balanced_accuracy": "Balanced accuracy (macro-average of class recalls, robust to class imbalance)",
    "eval_matthews_corrcoef": "Matthews correlation coefficient (MCC, range -1 to +1, robust multiclass agreement)",
    "eval_cohen_kappa": "Cohen's kappa score measuring chance-adjusted agreement between predictions and truth",
    "eval_exact_match_accuracy": "Subset accuracy: fraction of samples where all predicted labels match ground truth exactly",
    "eval_hamming_accuracy": "Fraction of correct individual binary label assignments across all classes",
    "eval_hamming_loss": "Fraction of wrong individual binary label assignments (lower is better)",
    "eval_precision_macro": "Unweighted average precision across all classes",
    "eval_precision_micro": "Global aggregate precision across all classes",
    "eval_precision_weighted": "Precision weighted by true instances of each class",
    "eval_recall_macro": "Unweighted average recall across all classes",
    "eval_recall_micro": "Global aggregate recall across all classes",
    "eval_recall_weighted": "Recall weighted by true instances of each class",
    "eval_token_accuracy": "Token-level next-token prediction accuracy on assistant output tokens",
    "eval_json_valid_parse_rate": "Fraction of model generations successfully parsed as valid JSON lists",
    "eval_runtime": "Total evaluation time on validation dataset in seconds",
    "eval_samples_per_second": "Evaluation throughput in samples per second",
    "eval_steps_per_second": "Evaluation throughput in batches/steps per second",
    "epoch": "Total training epochs elapsed",
    "train_loss": "Average loss over all training iterations",
    "train_runtime": "Total training execution time in seconds",
    "train_samples_per_second": "Training throughput in samples per second",
    "train_steps_per_second": "Training throughput in batches/steps per second",
    "total_flos": "Total floating point operations executed during training",
}


def get_metric_description(metric_name: str) -> str:
    """Returns a concise description for a given metric name."""
    if metric_name in METRIC_DESCRIPTIONS:
        return METRIC_DESCRIPTIONS[metric_name]
    clean_key = metric_name.lower().removeprefix("eval_").removeprefix("train_")
    for key, desc in METRIC_DESCRIPTIONS.items():
        if clean_key == key.lower().removeprefix("eval_").removeprefix("train_"):
            return desc
    return "Custom evaluation metric"


def clean_slug(name: str) -> str:
    """Creates a filesystem-safe slug from a model name or string."""
    name = Path(name).name
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name)


def generate_model_card_markdown(
    model_name: str,
    model_type: str,
    as_of: str,
    metrics: dict[str, Any],
    training_params: dict[str, Any],
    classes: list[str],
    checkpoint_dir: str | Path,
) -> str:
    """
    Generates a Markdown document with YAML frontmatter containing structured metadata
    and human-readable tables detailing performance and training configuration.
    """
    frontmatter = {
        "model": model_name,
        "type": model_type.lower(),
        "as_of": as_of,
        "checkpoint_dir": str(Path(checkpoint_dir).as_posix()),
        "classes": classes,
        "metrics": {
            k: (float(v) if isinstance(v, int | float) else str(v)) for k, v in metrics.items()
        },
        "training_params": {
            k: (v if isinstance(v, int | float | bool | str | list) else str(v))
            for k, v in training_params.items()
        },
    }

    frontmatter_yaml = yaml.dump(frontmatter, sort_keys=False, default_flow_style=False)

    # Markdown formatting
    f1_macro = metrics.get("eval_f1_macro", metrics.get("f1_macro", "N/A"))
    f1_micro = metrics.get("eval_f1_micro", metrics.get("f1_micro", "N/A"))
    exact_match = metrics.get(
        "eval_exact_match_accuracy", metrics.get("exact_match_accuracy", "N/A")
    )
    eval_loss = metrics.get("eval_loss", "N/A")

    lines = [
        "---",
        frontmatter_yaml.strip(),
        "---",
        "",
        f"# Model Card: {model_name} ({model_type.upper()})",
        "",
        f"- **Architecture Mode**: `{model_type.upper()}`",
        f"- **Base Model**: `{model_name}`",
        f"- **Created At**: `{as_of}`",
        f"- **Checkpoint Path**: `{checkpoint_dir}`",
        f"- **Number of Classes**: `{len(classes)}`",
        f"- **Target Classes**: `{', '.join(classes)}`",
        "",
        "## Evaluation Metrics",
        "",
        "| Metric | Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **Eval Loss** | `{eval_loss if isinstance(eval_loss, str) else f'{eval_loss:.4f}'}` | {get_metric_description('eval_loss')} |",
        f"| **F1 Macro** | `{f1_macro if isinstance(f1_macro, str) else f'{f1_macro:.4f}'}` | {get_metric_description('eval_f1_macro')} |",
        f"| **F1 Micro** | `{f1_micro if isinstance(f1_micro, str) else f'{f1_micro:.4f}'}` | {get_metric_description('eval_f1_micro')} |",
        f"| **Exact Match Accuracy** | `{exact_match if isinstance(exact_match, str) else f'{exact_match:.4f}'}` | {get_metric_description('eval_exact_match_accuracy')} |",
    ]

    for k, v in metrics.items():
        if k not in (
            "eval_loss",
            "eval_f1_macro",
            "eval_f1_micro",
            "eval_exact_match_accuracy",
            "f1_macro",
            "f1_micro",
            "exact_match_accuracy",
        ):
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            desc = get_metric_description(k)
            lines.append(f"| {k} | `{val_str}` | {desc} |")

    lines.extend(
        [
            "",
            "## Training Hyperparameters",
            "",
            "| Parameter | Value |",
            "| :--- | :--- |",
        ]
    )
    for k, v in training_params.items():
        lines.append(f"| `{k}` | `{v}` |")

    lines.append("")
    return "\n".join(lines)


def parse_model_card(card_path: str | Path) -> dict[str, Any]:
    """Parses the YAML frontmatter from a model card Markdown file."""
    path = Path(card_path)
    if not path.exists():
        raise FileNotFoundError(f"Model card not found: {path}")

    content = path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        raise ValueError(f"Model card at '{path}' is missing YAML frontmatter ('---').")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Model card at '{path}' has malformed YAML frontmatter.")

    frontmatter_raw = parts[1]
    data = yaml.safe_load(frontmatter_raw) or {}
    data["_card_path"] = str(path)
    return data


def find_best_model(
    models_dir: str | Path,
    model_type: str,
    metric: str = "eval_f1_macro",
) -> tuple[Path, dict[str, Any]]:
    """
    Scans `models_dir` for all `model_card_*.md` files, parses their evaluation metrics,
    and returns the checkpoint directory and metadata of the best performing model.

    Args:
        models_dir: Directory containing fine-tuned model checkpoints and model cards.
        model_type: Target mode to filter by ('cls' or 'json').
        metric: Performance metric to maximize (e.g. 'eval_f1_macro', 'eval_exact_match_accuracy')
                or minimize ('eval_loss').

    Returns:
        `(best_checkpoint_dir, best_model_card_data)`
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: '{models_dir}'")

    card_files = list(models_path.rglob("model_card_*.md"))
    if not card_files:
        raise FileNotFoundError(
            f"No model cards found in '{models_dir}' matching 'model_card_*.md'."
        )

    matching_candidates: list[dict[str, Any]] = []
    target_type = model_type.lower().strip()

    for card_file in card_files:
        try:
            card_data = parse_model_card(card_file)
            if card_data.get("type", "").lower() == target_type:
                matching_candidates.append(card_data)
        except Exception as e:
            logger.warning("Failed to parse model card '%s': %s", card_file, e)

    if not matching_candidates:
        raise FileNotFoundError(
            f"No model cards found in '{models_dir}' matching type '{model_type}'."
        )

    # Determine whether metric should be maximized or minimized
    minimize = "loss" in metric.lower()

    def get_metric_val(card: dict[str, Any]) -> float:
        m = card.get("metrics", {})
        # check both exact key and fallback key without 'eval_' prefix
        val = m.get(metric)
        if val is None and metric.startswith("eval_"):
            val = m.get(metric[5:])
        elif val is None and not metric.startswith("eval_"):
            val = m.get(f"eval_{metric}")

        if val is None:
            return float("inf") if minimize else float("-inf")
        try:
            return float(val)
        except (ValueError, TypeError):
            return float("inf") if minimize else float("-inf")

    best_card = sorted(matching_candidates, key=get_metric_val, reverse=not minimize)[0]
    best_checkpoint = Path(best_card["checkpoint_dir"])

    # Fallback to the directory containing the model card if checkpoint_dir doesn't exist directly
    if not best_checkpoint.exists():
        parent_dir = Path(best_card["_card_path"]).parent
        if (parent_dir / "easyslm_config.json").exists():
            best_checkpoint = parent_dir

    logger.info(
        "Selected best model for type '%s' using metric '%s'=%.4f: %s",
        target_type,
        metric,
        get_metric_val(best_card),
        best_checkpoint,
    )
    return best_checkpoint, best_card
