"""LoRA / PEFT adapter setup and GPU memory telemetry utilities for SLM fine-tuning."""

import logging

import torch
from peft import LoraConfig, TaskType, get_peft_model

from src.fine_tuning.config import FineTuningConfig
from src.slm import SLMBase, SLMCls, SLMJson

logger = logging.getLogger(__name__)


def apply_lora_to_model(
    model: SLMBase,
    config: FineTuningConfig,
) -> SLMBase:
    """
    Wraps `model.backbone` with LoRA adapters using `peft` so that only ~0.5% of parameters
    require optimizer states, keeping GPU VRAM usage well within 8 GB.
    In `SLMCls` mode, keeps the `norm` and linear `classifier` head fully trainable alongside LoRA.
    """
    task_type = TaskType.CAUSAL_LM if isinstance(model, SLMJson) else TaskType.FEATURE_EXTRACTION
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=task_type,
    )
    model.backbone = get_peft_model(model.backbone, lora_config)  # type: ignore[assignment]

    # Ensure the LayerNorm and dense classification head in SLMCls remain trainable
    if isinstance(model, SLMCls):
        for param in model.norm.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

    trainable_params, all_params = get_parameter_counts(model)
    pct = 100.0 * trainable_params / max(all_params, 1)
    logger.info(
        "Applied LoRA (r=%d, alpha=%d) | Trainable params: %s / %s (%.2f%%)",
        config.lora_r,
        config.lora_alpha,
        f"{trainable_params:,}",
        f"{all_params:,}",
        pct,
    )
    return model


def get_parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    """Returns `(trainable_params, total_params)` for a PyTorch module."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def log_gpu_memory(prefix: str = "") -> None:
    """Logs current and peak CUDA memory usage via standard `logging`."""
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        max_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.info(
            "%sGPU VRAM | Allocated: %.1f MB | Reserved: %.1f MB | Peak Allocated: %.1f MB",
            f"[{prefix}] " if prefix else "",
            allocated_mb,
            reserved_mb,
            max_alloc_mb,
        )
