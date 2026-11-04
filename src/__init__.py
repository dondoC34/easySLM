"""easySLM: A generic tool for fine-tuning Small Language Models on classification tasks."""

from src.data import SLMClsDataset, SLMCollator, SLMDatasetBase, SLMJsonDataset
from src.evaluation import (
    build_cls_compute_metrics,
    build_json_compute_metrics,
    compute_classification_metrics,
    preprocess_logits_for_json_metrics,
)
from src.fine_tuning import FineTuningConfig, SLMFineTuner, apply_lora_to_model
from src.inference import SLMPredictor
from src.slm import (
    ModelFactory,
    PoolingStrategy,
    SLMBase,
    SLMCls,
    SLMJson,
    load_finetuned_model,
    save_finetuned_model,
)
from src.utils.config import get_project_root, load_chat_template, load_prompts_config, load_yaml
from src.utils.device import get_default_device, get_default_dtype, print_cuda_info
from src.utils.model_card import find_best_model

__version__ = "0.1.0"
__all__ = [
    "FineTuningConfig",
    "ModelFactory",
    "PoolingStrategy",
    "SLMBase",
    "SLMCls",
    "SLMClsDataset",
    "SLMCollator",
    "SLMDatasetBase",
    "SLMFineTuner",
    "SLMJson",
    "SLMJsonDataset",
    "SLMPredictor",
    "apply_lora_to_model",
    "build_cls_compute_metrics",
    "build_json_compute_metrics",
    "compute_classification_metrics",
    "find_best_model",
    "get_default_device",
    "get_default_dtype",
    "get_project_root",
    "load_chat_template",
    "load_finetuned_model",
    "load_prompts_config",
    "load_yaml",
    "preprocess_logits_for_json_metrics",
    "print_cuda_info",
    "save_finetuned_model",
]
