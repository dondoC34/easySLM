import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from transformers import EvalPrediction, PreTrainedTokenizerBase

from src.slm.json_slm import SLMJson

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multi_label: bool = True,
) -> dict[str, float]:
    """Computes classification metrics given binary indicator matrices `y_true` and `y_pred`

    of shape `(num_samples, num_classes)`.

    Metrics computed:
    - `accuracy` & `exact_match_accuracy`: top-1 / subset accuracy.
    - `balanced_accuracy`: macro-averaged recall across classes (for single-label).
    - `matthews_corrcoef`: multiclass correlation coefficient (for single-label).
    - `cohen_kappa`: chance-adjusted agreement statistic (for single-label).
    - `hamming_accuracy`, `hamming_loss`
    - `precision_macro`, `precision_micro`, `precision_weighted`
    - `recall_macro`, `recall_micro`, `recall_weighted`
    - `f1_macro`, `f1_micro`, `f1_weighted`, `f1_samples`
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    h_loss = float(hamming_loss(y_true, y_pred))
    metrics = {
        "accuracy": acc,
        "exact_match_accuracy": acc,
        "hamming_accuracy": float(1.0 - h_loss),
        "hamming_loss": h_loss,
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }

    # In single-label scenario, compute multiclass-specific metrics
    is_single = (not multi_label) or (
        y_true.ndim == 2 and np.all(y_true.sum(axis=1) <= 1) and np.all(y_pred.sum(axis=1) <= 1)
    )
    if is_single and len(y_true) > 0:
        true_indices = np.argmax(y_true, axis=1)
        pred_indices = np.argmax(y_pred, axis=1)
        try:
            metrics["balanced_accuracy"] = float(
                balanced_accuracy_score(true_indices, pred_indices)
            )
        except Exception:
            metrics["balanced_accuracy"] = 0.0
        try:
            metrics["matthews_corrcoef"] = float(matthews_corrcoef(true_indices, pred_indices))
        except Exception:
            metrics["matthews_corrcoef"] = 0.0
        try:
            metrics["cohen_kappa"] = float(cohen_kappa_score(true_indices, pred_indices))
        except Exception:
            metrics["cohen_kappa"] = 0.0

    return metrics


def extract_cls_binary_arrays(
    logits: Any,
    labels: Any,
    multi_label: bool = True,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Converts raw model logits and ground truth label_ids into binary indicator matrices `(y_true, y_pred)`."""
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels)

    num_classes = logits.shape[-1]

    if multi_label:
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        y_pred = (probs >= threshold).astype(np.int32)
        y_true = (labels >= 0.5).astype(np.int32)
    else:
        pred_indices = np.argmax(logits, axis=-1)
        y_pred = np.zeros((len(pred_indices), num_classes), dtype=np.int32)
        y_pred[np.arange(len(pred_indices)), pred_indices] = 1

        if labels.ndim > 1 and labels.shape[-1] == num_classes:
            y_true = (labels >= 0.5).astype(np.int32)
        else:
            y_true = np.zeros((len(labels), num_classes), dtype=np.int32)
            y_true[np.arange(len(labels)), labels.astype(np.int64)] = 1

    return y_true, y_pred


def build_cls_compute_metrics(
    multi_label: bool = True,
    threshold: float = 0.5,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Builds a `compute_metrics` callable for Hugging Face `Trainer` in `SLMCls` mode."""

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        y_true, y_pred = extract_cls_binary_arrays(
            logits=eval_pred.predictions,
            labels=eval_pred.label_ids,
            multi_label=multi_label,
            threshold=threshold,
        )

        metrics = compute_classification_metrics(y_true, y_pred, multi_label=multi_label)
        logger.info(
            "CLS Evaluation Metrics | f1_macro=%.4f | f1_micro=%.4f | exact_match=%.4f | hamming_acc=%.4f",
            metrics["f1_macro"],
            metrics["f1_micro"],
            metrics["exact_match_accuracy"],
            metrics["hamming_accuracy"],
        )
        return metrics

    return compute_metrics


def preprocess_logits_for_json_metrics(
    logits: torch.Tensor | tuple[torch.Tensor, ...],
    labels: torch.Tensor,
) -> torch.Tensor:
    """Reduces CausalLM logits from `(batch_size, seq_len, vocab_size)` to `(batch_size, seq_len)`

    token IDs via `argmax(dim=-1)` immediately on GPU during evaluation to save VRAM.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def _parse_class_list_string(
    raw_text: str,
    class_to_idx: dict[str, int],
) -> tuple[list[str], bool]:
    """Parses a decoded string (e.g. `"['economy', 'politics']"`) into a sorted list of valid classes

    and a boolean flag indicating whether a bracketed list was successfully parsed.
    """
    return SLMJson.parse_class_list(raw_text, class_to_idx)


def extract_json_binary_arrays(
    pred_ids: Any,
    label_ids: Any,
    tokenizer: PreTrainedTokenizerBase,
    classes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Decodes token sequences and parses JSON class lists into `(y_true, y_pred, correct_tokens, total_tokens, valid_json_count)`."""
    sorted_classes = sorted(list(dict.fromkeys(classes)))
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    num_classes = len(sorted_classes)

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_ids = np.asarray(pred_ids)
    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    label_ids = np.asarray(label_ids)

    shift_preds = pred_ids[:, :-1]
    shift_labels = label_ids[:, 1:]

    num_samples = shift_labels.shape[0]
    y_true = np.zeros((num_samples, num_classes), dtype=np.int32)
    y_pred = np.zeros((num_samples, num_classes), dtype=np.int32)

    total_tokens = 0
    correct_tokens = 0
    valid_json_count = 0

    for i in range(num_samples):
        mask = shift_labels[i] != -100
        sample_true_ids = shift_labels[i][mask]
        sample_pred_ids = shift_preds[i][mask]

        total_tokens += len(sample_true_ids)
        correct_tokens += int(np.sum(sample_true_ids == sample_pred_ids))

        true_str = str(tokenizer.decode(sample_true_ids.tolist(), skip_special_tokens=True))
        pred_str = str(tokenizer.decode(sample_pred_ids.tolist(), skip_special_tokens=True))

        true_classes, _ = _parse_class_list_string(true_str, class_to_idx)
        pred_classes, is_valid_json = _parse_class_list_string(pred_str, class_to_idx)

        if is_valid_json:
            valid_json_count += 1

        for c in true_classes:
            y_true[i, class_to_idx[c]] = 1
        for c in pred_classes:
            y_pred[i, class_to_idx[c]] = 1

    return y_true, y_pred, correct_tokens, total_tokens, valid_json_count


def build_json_compute_metrics(
    tokenizer: PreTrainedTokenizerBase,
    classes: Sequence[str],
    multi_label: bool = True,
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Builds a `compute_metrics` callable for Hugging Face `Trainer` in `SLMJson` mode.

    Computes both:
    1. `token_accuracy`: next-token prediction accuracy strictly on the unmasked assistant tokens.
    2. Full classification metrics (`f1_macro`, `f1_micro`, `exact_match_accuracy`, etc.) by decoding
       the predicted assistant tokens vs target assistant tokens and parsing the JSON class lists.
    """
    sorted_classes = sorted(list(dict.fromkeys(classes)))

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        (
            y_true,
            y_pred,
            correct_tokens,
            total_tokens,
            valid_json_count,
        ) = extract_json_binary_arrays(
            pred_ids=eval_pred.predictions,
            label_ids=eval_pred.label_ids,
            tokenizer=tokenizer,
            classes=sorted_classes,
        )

        num_samples = len(y_true)
        metrics = compute_classification_metrics(y_true, y_pred, multi_label=multi_label)
        metrics["token_accuracy"] = float(correct_tokens / max(total_tokens, 1))
        metrics["json_valid_parse_rate"] = float(valid_json_count / max(num_samples, 1))

        logger.info(
            "JSON Evaluation Metrics | token_acc=%.4f | valid_json=%.2f | f1_macro=%.4f | f1_micro=%.4f",
            metrics["token_accuracy"],
            metrics["json_valid_parse_rate"],
            metrics["f1_macro"],
            metrics["f1_micro"],
        )
        return metrics

    return compute_metrics
