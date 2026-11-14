"""Fine-tuning package for easySLM: configuration, LoRA setup, callbacks, evaluation, and orchestrator."""

from src.fine_tuning.callbacks import EarlyStoppingWindowCallback
from src.fine_tuning.config import FineTuningConfig
from src.fine_tuning.evaluation import (
    build_cls_compute_metrics,
    build_json_compute_metrics,
    compute_classification_metrics,
    extract_cls_binary_arrays,
    extract_json_binary_arrays,
    extract_validation_prediction_details,
    preprocess_logits_for_json_metrics,
)
from src.fine_tuning.lora import apply_lora_to_model, get_parameter_counts, log_gpu_memory
from src.fine_tuning.trainer import SLMFineTuner

__all__ = [
    "EarlyStoppingWindowCallback",
    "FineTuningConfig",
    "SLMFineTuner",
    "apply_lora_to_model",
    "build_cls_compute_metrics",
    "build_json_compute_metrics",
    "compute_classification_metrics",
    "extract_cls_binary_arrays",
    "extract_json_binary_arrays",
    "extract_validation_prediction_details",
    "get_parameter_counts",
    "log_gpu_memory",
    "preprocess_logits_for_json_metrics",
]
