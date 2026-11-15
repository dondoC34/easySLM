"""Configuration schema and Hugging Face `TrainingArguments` builder for SLM fine-tuning."""

import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch
from transformers import TrainingArguments

from src.slm import PoolingStrategy
from src.utils.config import load_yaml

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """
    Configuration for fine-tuning Small Language Models with Hugging Face `Trainer`.

    Defaults follow best practices for 8 GB GPU VRAM (LoRA, Gradient Checkpointing,
    Mixed Precision bf16/fp16, Gradient Accumulation, Cosine LR Schedule with Warmup,
    Gradient Clipping, and MLflow tracking).
    """

    output_dir: str = "./models/finetuned"
    mode: Literal["json", "cls"] = "cls"
    pooling_strategy: PoolingStrategy = "eos"

    # Batching & VRAM optimization (tuned for 8 GB VRAM)
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True

    # LoRA / PEFT parameters
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"

    # Adaptive Learning Rate & Optimization best practices
    learning_rate: float | None = None  # Auto-selected: 2e-4 for LoRA, 2e-5 for full FT
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0

    # Prompt configuration
    prompts_config_path: str = "configs/prompts.yml"

    # Evaluation & Checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 10
    save_strategy: str = "no"
    save_total_limit: int = 2
    logging_steps: int = 5
    val_split_ratio: float = 0.15
    oversample_strategy: str | None = None
    oversample_max_factor: float | None = None
    early_stopping_window: int | None = 4
    early_stopping_metric: str = "eval_loss"
    seed: int = 42

    # MLflow Integration
    use_mlflow: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_config_path: str | None = None
    run_name: str | None = None
    max_errors_per_class: int = 5

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path = "configs/training_config.yml",
        **overrides,
    ) -> "FineTuningConfig":
        """
        Loads configuration from a YAML file, with optional keyword overrides.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Optional parameter overrides to take precedence over the YAML values.
        """
        raw_dict = load_yaml(yaml_path)
        raw_dict.update(overrides)

        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw_dict.items() if k in valid_fields}
        return cls(**filtered)

    def to_training_arguments(
        self,
        num_train_samples: int,
        has_eval_dataset: bool = True,
        report_to: list[str] | None = None,
        run_name: str | None = None,
    ) -> TrainingArguments:
        """
        Converts configuration fields into a Hugging Face `TrainingArguments` instance,
        dynamically determining precision flags (bf16/fp16), warmup steps, and learning rate.
        """
        use_cuda = torch.cuda.is_available()
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
        use_fp16 = use_cuda and not use_bf16

        effective_lr = self.learning_rate
        if effective_lr is None:
            effective_lr = 2e-4 if self.use_lora else 2e-5

        effective_batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        steps_per_epoch = max(1, math.ceil(num_train_samples / effective_batch))
        total_steps = max(1, int(steps_per_epoch * self.num_train_epochs))

        warmup_steps = (
            max(1, int(round(total_steps * self.warmup_ratio))) if self.warmup_ratio > 0.0 else 0
        )

        effective_report_to = (
            report_to if report_to is not None else (["mlflow"] if self.use_mlflow else ["none"])
        )
        eval_strategy = self.eval_strategy if has_eval_dataset else "no"
        effective_run_name = run_name or self.run_name or f"easyslm-{self.mode}"

        logger.info(
            "Configuring TrainingArguments | lr=%.2e | scheduler=%s | warmup_steps=%d/%d | "
            "batch_size=%d | grad_accum=%d | bf16=%s | fp16=%s | epochs=%.2f",
            effective_lr,
            self.lr_scheduler_type,
            warmup_steps,
            total_steps,
            self.per_device_train_batch_size,
            self.gradient_accumulation_steps,
            use_bf16,
            use_fp16,
            self.num_train_epochs,
        )

        return TrainingArguments(
            output_dir=self.output_dir,
            run_name=effective_run_name,
            learning_rate=effective_lr,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_steps=warmup_steps,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            optim="adamw_torch",
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            bf16=use_bf16,
            fp16=use_fp16,
            eval_strategy=eval_strategy,
            eval_steps=self.eval_steps if eval_strategy == "steps" else None,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            logging_steps=self.logging_steps,
            report_to=effective_report_to,
            remove_unused_columns=False,
            label_names=["labels"],
            dataloader_pin_memory=use_cuda,
            seed=self.seed,
        )
