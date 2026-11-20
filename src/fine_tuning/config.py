"""Configuration schema and Hugging Face `TrainingArguments` builder for SLM fine-tuning."""

import logging
import math
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from transformers import TrainingArguments

from src.slm import PoolingStrategy
from src.utils.config import get_default_finetuned_dir, load_yaml
from src.utils.device import is_native_bf16_supported

logger = logging.getLogger(__name__)


class FineTuningConfig(BaseModel):
    """
    Pydantic configuration model for fine-tuning Small Language Models with Hugging Face `Trainer`.

    Serves as the single source of truth for all parameter defaults and validation.
    Defaults follow best practices for 8 GB GPU VRAM (LoRA, Gradient Checkpointing,
    Mixed Precision bf16/fp16, Gradient Accumulation, Cosine LR Schedule with Warmup,
    Gradient Clipping, and MLflow tracking).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    output_dir: str = Field(default_factory=lambda: str(get_default_finetuned_dir()))
    mode: Literal["json", "cls"] = "cls"
    pooling_strategy: PoolingStrategy = "eos"

    # Batching & VRAM optimization (tuned for 8 GB VRAM)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    gradient_checkpointing: bool = True

    # LoRA / PEFT parameters
    use_lora: bool = True
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: str | list[str] = "all-linear"

    # Adaptive Learning Rate & Optimization best practices
    learning_rate: float | None = Field(default=None, gt=0.0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    num_train_epochs: float = Field(default=3.0, gt=0.0)

    # Prompt configuration
    prompts_config_path: str = "configs/prompts.yml"

    # Evaluation & Checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = Field(default=10, ge=1)
    save_strategy: str = "no"
    save_total_limit: int = Field(default=2, ge=1)
    logging_steps: int = Field(default=5, ge=1)
    val_split_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    oversample_strategy: str | None = None
    oversample_max_factor: float | None = Field(default=None, gt=0.0)
    early_stopping_window: int | None = Field(default=4, ge=1)
    early_stopping_metric: str = "eval_loss"
    seed: int = 42

    # MLflow Integration
    use_mlflow: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_config_path: str | None = None
    run_name: str | None = None
    max_errors_per_class: int = Field(default=5, ge=0)

    @staticmethod
    def _format_field_default(field_info: FieldInfo) -> Any:
        if field_info.default_factory is not None:
            return field_info.default_factory()  # type: ignore[call-arg]
        return field_info.default

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            unknown = sorted(set(data.keys()) - set(cls.model_fields.keys()))
            if unknown:
                accepted = ", ".join(
                    f"{name}={cls._format_field_default(info)!r}"
                    for name, info in cls.model_fields.items()
                )
                raise ValueError(
                    f"Unrecognized configuration key(s): {', '.join(unknown)}. "
                    f"Accepted keys and their defaults: {accepted}"
                )
        return data

    @field_validator("output_dir", mode="before")
    @classmethod
    def _default_output_dir_if_none(cls, value: Any) -> Any:
        if value is None or value == "":
            return str(get_default_finetuned_dir())
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None, **overrides: Any) -> "FineTuningConfig":
        """Instantiates `FineTuningConfig` from a partial or empty dictionary with optional overrides."""
        merged: dict[str, Any] = dict(data) if data is not None else {}
        merged.update(overrides)
        return cls.model_validate(merged)

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path = "configs/training_config.yml",
        **overrides: Any,
    ) -> "FineTuningConfig":
        """
        Loads configuration from a YAML file, with optional keyword overrides.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Optional parameter overrides to take precedence over the YAML values.
        """
        raw_dict = load_yaml(yaml_path)
        return cls.from_dict(raw_dict, **overrides)

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
        use_bf16 = use_cuda and is_native_bf16_supported()
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
