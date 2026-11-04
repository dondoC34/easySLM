import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, random_split
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.data import SLMClsDataset, SLMDatasetBase, SLMJsonDataset
from src.evaluation import (
    build_cls_compute_metrics,
    build_json_compute_metrics,
    preprocess_logits_for_json_metrics,
)
from src.slm import ModelFactory, PoolingStrategy, SLMBase, SLMCls, SLMJson
from src.slm.serialization import save_finetuned_model
from src.utils.config import get_project_root, load_yaml
from src.utils.model_card import clean_slug

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """
    Configuration for fine-tuning Small Language Models with Hugging Face `Trainer`.

    Defaults follow best practices for 8 GB GPU VRAM (LoRA, Gradient Checkpointing,
    Mixed Precision bf16/fp16, Gradient Accumulation, Cosine LR Schedule with Warmup,
    Gradient Clipping, and MLflow tracking).
    """

    output_dir: str = "./checkpoints"
    mode: Literal["json", "cls"] = "cls"
    multi_label: bool = True
    pooling_strategy: PoolingStrategy = "eos"

    # Batching & VRAM optimization (tuned for 8 GB VRAM)
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    max_length: int | None = 512

    # LoRA / PEFT parameters
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"

    # Adaptive Learning Rate & Optimization best practices
    learning_rate: float | None = None  # Auto-selected: 2e-4 for LoRA, 3e-5 for full FT
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0

    # Prompt and template configurations
    prompts_config_path: str = "configs/prompts.yml"
    chat_template_config_path: str = "configs/chat_template.yml"

    # Evaluation & Checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 10
    save_strategy: str = "no"
    save_total_limit: int = 2
    logging_steps: int = 1
    val_split_ratio: float = 0.2
    seed: int = 42

    # MLflow Integration
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "easySLM_finetuning"
    run_name: str | None = None

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


def apply_lora_to_model(
    model: SLMBase,
    config: FineTuningConfig,
) -> SLMBase:
    """
    Wraps `model.backbone` with LoRA adapters using `peft` so that only ~0.5% of parameters
    require optimizer states, keeping GPU VRAM usage well within 8 GB.
    In `SLMCls` mode, keeps the linear `classifier` head fully trainable alongside LoRA.
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

    # Ensure the dense classification head in SLMCls remains trainable
    if isinstance(model, SLMCls):
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


class SLMFineTuner:
    """
    High-level orchestrator for fine-tuning `SLMJson` and `SLMCls` models using
    Hugging Face `transformers.Trainer`.

    Features:
    - Automatic dataset construction and train/validation splitting.
    - 8 GB GPU VRAM management (LoRA/PEFT, Gradient Checkpointing, bf16/fp16 Mixed Precision,
      Gradient Accumulation, and on-GPU logit argmax reduction for JSON evaluation).
    - Adaptive Learning Rate (Cosine scheduler with warmup, AdamW weight decay, gradient clipping).
    - MLflow experiment and metric tracking.
    - Built-in classification evaluation metrics (`f1_macro`, `f1_micro`, `exact_match_accuracy`, etc.).
    """

    def __init__(
        self,
        model_name: str,
        config: FineTuningConfig | str | Path | None = None,
        factory: ModelFactory | None = None,
    ):
        self.model_name = model_name
        if isinstance(config, str | Path):
            self.config = FineTuningConfig.from_yaml(config)
        elif config is None:
            default_yaml = get_project_root() / "configs/training_config.yml"
            if default_yaml.exists():
                self.config = FineTuningConfig.from_yaml(default_yaml)
            else:
                self.config = FineTuningConfig()
        else:
            self.config = config

        self.factory = factory or ModelFactory()

        self.model: SLMBase | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.trainer: Trainer | None = None
        self.classes: list[str] = []

    def _configure_mlflow(self) -> list[str]:
        """Configures MLflow environment variables and returns the `report_to` list for `Trainer`."""
        if not self.config.use_mlflow:
            return ["none"]

        tracking_uri = self.config.mlflow_tracking_uri
        # Convert relative local directory paths into clean file:// URIs or local paths for MLflow
        if not tracking_uri.startswith(("http://", "https://", "sqlite://", "file:")):
            tracking_path = Path(tracking_uri).resolve()
            tracking_path.mkdir(parents=True, exist_ok=True)
            tracking_uri = tracking_path.as_uri()

        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = self.config.mlflow_experiment_name
        os.environ["MLFLOW_FLATTEN_PARAMS"] = "True"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

        logger.info(
            "MLflow configured | URI: %s | Experiment: %s",
            tracking_uri,
            self.config.mlflow_experiment_name,
        )
        return ["mlflow"]

    def prepare_datasets(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path | None = None,
        classes: Sequence[str] | None = None,
    ) -> tuple[Dataset, Dataset | None, SLMDatasetBase]:
        """
        Loads the dataset(s) using `SLMJsonDataset` or `SLMClsDataset`.
        If `val_data_path` is None and `val_split_ratio > 0`, splits `train_data_path` into train/val.

        Returns:
            `(train_dataset, eval_dataset, base_dataset)`
        """
        if self.tokenizer is None:
            local_path = self.factory.download_model(self.model_name)
            self.tokenizer = self.factory._prepare_tokenizer(local_path, padding_side="right")

        dataset_cls = SLMJsonDataset if self.config.mode == "json" else SLMClsDataset

        full_dataset: SLMDatasetBase = dataset_cls(
            data_path=train_data_path,
            tokenizer=self.tokenizer,
            classes=classes,
            multi_label=self.config.multi_label,
            max_length=self.config.max_length,
            include_targets=True,
            chat_template_config_path=self.config.chat_template_config_path,
            prompts_config_path=self.config.prompts_config_path,
        )
        self.classes = full_dataset.classes
        logger.info(
            "Loaded dataset from '%s' | Mode: %s | Samples: %d | Classes (%d): %s",
            train_data_path,
            self.config.mode.upper(),
            len(full_dataset),
            full_dataset.num_classes,
            self.classes,
        )

        if val_data_path is not None:
            val_dataset: Dataset | None = dataset_cls(
                data_path=val_data_path,
                tokenizer=self.tokenizer,
                classes=self.classes,
                multi_label=self.config.multi_label,
                max_length=self.config.max_length,
                include_targets=True,
            )
            train_dataset: Dataset = full_dataset
            logger.info(
                "Loaded separate validation dataset from '%s' (%d samples).",
                val_data_path,
                len(val_dataset),  # type: ignore[arg-type]
            )
        elif self.config.val_split_ratio > 0.0 and len(full_dataset) >= 4:
            val_size = max(1, int(round(len(full_dataset) * self.config.val_split_ratio)))
            train_size = len(full_dataset) - val_size
            generator = torch.Generator().manual_seed(self.config.seed)
            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=generator,
            )
            logger.info(
                "Split dataset into %d training samples and %d validation samples (ratio=%.2f).",
                train_size,
                val_size,
                self.config.val_split_ratio,
            )
        else:
            train_dataset = full_dataset
            val_dataset = None

        return train_dataset, val_dataset, full_dataset

    def build_model(self, classes: Sequence[str]) -> SLMBase:
        """Instantiates `SLMJson` or `SLMCls` and applies LoRA + Gradient Checkpointing."""
        log_gpu_memory("Before model load")

        model: SLMBase
        if self.config.mode == "json":
            model, tokenizer = SLMJson.from_pretrained(
                model_name=self.model_name,
                classes=classes,
                multi_label=self.config.multi_label,
                factory=self.factory,
            )
        elif self.config.mode == "cls":
            model, tokenizer = SLMCls.from_pretrained(
                model_name=self.model_name,
                classes=classes,
                multi_label=self.config.multi_label,
                pooling_strategy=self.config.pooling_strategy,
                factory=self.factory,
            )
        else:
            raise ValueError(f"Unsupported mode '{self.config.mode}'. Choose 'json' or 'cls'.")

        self.tokenizer = tokenizer

        if self.config.use_lora:
            model = apply_lora_to_model(model, self.config)

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        log_gpu_memory("After model + LoRA setup")
        self.model = model
        return model

    def build_training_arguments(
        self,
        num_train_samples: int,
        has_eval_dataset: bool,
    ) -> TrainingArguments:
        """Constructs Hugging Face `TrainingArguments` by delegating to `FineTuningConfig`."""
        report_to = self._configure_mlflow()
        run_name = (
            self.config.run_name or f"easyslm-{self.config.mode}-{Path(self.model_name).name}"
        )
        return self.config.to_training_arguments(
            num_train_samples=num_train_samples,
            has_eval_dataset=has_eval_dataset,
            report_to=report_to,
            run_name=run_name,
        )

    def train(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path | None = None,
        classes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Runs the complete fine-tuning and evaluation pipeline using `transformers.Trainer`,
        then automatically saves the checkpoint, tokenizer, and Markdown Model Card.

        Returns:
            Dictionary with `train_metrics`, `eval_metrics`, trained `model`, and `checkpoint_dir`.
        """
        train_dataset, eval_dataset, base_dataset = self.prepare_datasets(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            classes=classes,
        )

        model = self.build_model(classes=self.classes)
        collator = base_dataset.get_collator(padding_side="right", device=None)
        training_args = self.build_training_arguments(
            num_train_samples=len(train_dataset),  # type: ignore[arg-type]
            has_eval_dataset=eval_dataset is not None,
        )

        assert self.tokenizer is not None
        if self.config.mode == "cls":
            compute_metrics_fn = build_cls_compute_metrics(multi_label=self.config.multi_label)
            preprocess_logits_fn = None
        else:
            compute_metrics_fn = build_json_compute_metrics(
                tokenizer=self.tokenizer,
                classes=self.classes,
                multi_label=self.config.multi_label,
            )
            preprocess_logits_fn = preprocess_logits_for_json_metrics

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics_fn,
            preprocess_logits_for_metrics=preprocess_logits_fn,
        )

        logger.info(
            "Starting fine-tuning via Hugging Face Trainer (mode=%s)...", self.config.mode.upper()
        )
        eval_metrics: dict[str, Any] = {}

        if self.config.use_mlflow:
            import mlflow

            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            with mlflow.start_run(run_name=training_args.run_name):
                train_result = self.trainer.train()
                train_metrics = train_result.metrics
                logger.info("Training completed | Train Metrics: %s", train_metrics)
                log_gpu_memory("Post-training")

                if eval_dataset is not None:
                    logger.info(
                        "Running evaluation on validation set (%d samples)...",
                        len(eval_dataset),  # type: ignore[arg-type]
                    )
                    eval_metrics = self.trainer.evaluate()
                    logger.info("Validation Evaluation Metrics: %s", eval_metrics)
        else:
            train_result = self.trainer.train()
            train_metrics = train_result.metrics
            logger.info("Training completed | Train Metrics: %s", train_metrics)
            log_gpu_memory("Post-training")

            if eval_dataset is not None:
                logger.info(
                    "Running evaluation on validation set (%d samples)...",
                    len(eval_dataset),  # type: ignore[arg-type]
                )
                eval_metrics = self.trainer.evaluate()
                logger.info("Validation Evaluation Metrics: %s", eval_metrics)

        # Save model checkpoint and Model Card
        as_of = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = clean_slug(self.model_name)
        run_output_dir = Path(self.config.output_dir) / f"{model_slug}_{self.config.mode}_{as_of}"

        training_params = {
            "num_train_epochs": self.config.num_train_epochs,
            "learning_rate": self.config.learning_rate or (2e-4 if self.config.use_lora else 2e-5),
            "batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "use_lora": self.config.use_lora,
            "lora_r": self.config.lora_r if self.config.use_lora else None,
            "lora_alpha": self.config.lora_alpha if self.config.use_lora else None,
            "pooling_strategy": self.config.pooling_strategy if self.config.mode == "cls" else None,
            "max_length": self.config.max_length,
            "dataset": str(train_data_path),
        }

        assert self.model is not None
        checkpoint_path = save_finetuned_model(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=run_output_dir,
            base_model_name=self.model_name,
            training_params=training_params,
            metrics=eval_metrics or train_metrics,
            as_of=as_of,
        )

        return {
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "checkpoint_dir": checkpoint_path,
        }
