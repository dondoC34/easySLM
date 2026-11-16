"""High-level fine-tuning orchestrator (`SLMFineTuner`) combining datasets, models, Trainer, and tracking."""

import contextlib
import logging
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from src.data import SLMClsDataset, SLMDatasetBase, SLMJsonDataset
from src.data.base_dataset import get_backbone_max_length
from src.fine_tuning.callbacks import EarlyStoppingWindowCallback
from src.fine_tuning.config import FineTuningConfig
from src.fine_tuning.evaluation import (
    build_cls_compute_metrics,
    build_json_compute_metrics,
    extract_validation_prediction_details,
    preprocess_logits_for_json_metrics,
)
from src.fine_tuning.lora import apply_lora_to_model, log_gpu_memory
from src.slm import ModelFactory, SLMBase, SLMCls, SLMJson
from src.tracking.artifacts import log_evaluation_artifacts
from src.tracking.config import configure_mlflow_env, get_mlflow_config
from src.tracking.tables import create_validation_errors_table
from src.utils.config import get_project_root
from src.utils.model_card import clean_slug

logger = logging.getLogger(__name__)


class SLMFineTuner:
    """
    High-level orchestrator for fine-tuning `SLMJson` and `SLMCls` models using
    Hugging Face `transformers.Trainer`.

    Features:
    - Automatic dataset construction, task-type inference, and train/validation splitting.
    - 8 GB GPU VRAM management (LoRA/PEFT, Gradient Checkpointing, bf16/fp16 Mixed Precision,
      Gradient Accumulation, and on-GPU logit argmax reduction for JSON evaluation).
    - Adaptive Learning Rate (Cosine scheduler with warmup, AdamW weight decay, gradient clipping).
    - Optional in-memory Early Stopping (`EarlyStoppingWindowCallback`).
    - MLflow experiment, metric, and visual/tabular artifact tracking.
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
        self.multi_label: bool = False
        self.max_length: int | None = None
        self.run_output_dir: Path | None = None

        self.mlflow_cfg = (
            get_mlflow_config(self.config.mlflow_config_path) if self.config.use_mlflow else {}
        )

    def prepare_datasets(
        self,
        train_data_path: str | Path,
        val_data_path: str | Path | None = None,
        classes: Sequence[str] | None = None,
    ) -> tuple[SLMDatasetBase, SLMDatasetBase | None, SLMDatasetBase]:
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

        backbone_config = getattr(self.model, "config", None)
        if backbone_config is None:
            try:
                model_path = self.factory.download_model(self.model_name)
                backbone_config = AutoConfig.from_pretrained(model_path)
            except Exception:
                backbone_config = None
        backbone_max_length = get_backbone_max_length(backbone_config)

        # If a directory is passed as train_data_path, auto-discover train.txt and optional val.txt
        resolved_train_path = Path(train_data_path)
        if resolved_train_path.is_dir():
            if (resolved_train_path / "train.txt").is_file():
                data_dir = resolved_train_path
                resolved_train_path = data_dir / "train.txt"
                if val_data_path is None:
                    for candidate in ("val.txt", "validation.txt"):
                        if (data_dir / candidate).is_file():
                            val_data_path = data_dir / candidate
                            break

        full_dataset: SLMDatasetBase = dataset_cls(
            data_path=resolved_train_path,
            tokenizer=self.tokenizer,
            classes=classes,
            include_targets=True,
            prompts_config_path=self.config.prompts_config_path,
            backbone_max_length=backbone_max_length,
        )
        self.classes = full_dataset.classes
        self.multi_label = full_dataset.multi_label
        self.max_length = full_dataset.max_length
        logger.info(
            "Inferred task type from dataset: %s",
            "MULTI-LABEL" if self.multi_label else "SINGLE-LABEL",
        )
        logger.info(
            "Loaded dataset from '%s' | Mode: %s | Samples: %d | Max tokens: %s | Classes (%d): %s",
            resolved_train_path,
            self.config.mode.upper(),
            len(full_dataset),
            self.max_length,
            full_dataset.num_classes,
            self.classes,
        )

        if val_data_path is not None:
            val_ds: SLMDatasetBase = dataset_cls(
                data_path=val_data_path,
                tokenizer=self.tokenizer,
                classes=self.classes,
                multi_label=self.multi_label,
                include_targets=True,
                prompts_config_path=self.config.prompts_config_path,
                backbone_max_length=backbone_max_length,
            )
            if self.max_length is not None and val_ds.max_length is not None:
                self.max_length = max(self.max_length, val_ds.max_length)
            train_dataset: SLMDatasetBase = full_dataset
            logger.info(
                "Loaded separate validation dataset from '%s' (%d samples).",
                val_data_path,
                len(val_ds),
            )
            val_dataset: SLMDatasetBase | None = val_ds
        elif self.config.val_split_ratio > 0.0 and len(full_dataset) >= 4:
            train_dataset, val_ds = full_dataset.split(
                val_ratio=self.config.val_split_ratio,
                seed=self.config.seed,
            )
            logger.info(
                "Split dataset into %d training samples and %d validation samples (ratio=%.2f).",
                len(train_dataset),
                len(val_ds),
                self.config.val_split_ratio,
            )
            val_dataset = val_ds
        else:
            train_dataset = full_dataset
            val_dataset = None

        if self.config.oversample_strategy and self.config.oversample_strategy != "none":
            before_len = len(train_dataset)
            train_dataset.oversample(
                strategy=self.config.oversample_strategy,  # type: ignore[arg-type]
                max_factor=self.config.oversample_max_factor,
                seed=self.config.seed,
            )
            logger.info(
                "Oversampled training dataset (%s, max_factor=%s): %d -> %d samples.",
                self.config.oversample_strategy,
                self.config.oversample_max_factor,
                before_len,
                len(train_dataset),
            )

        return train_dataset, val_dataset, full_dataset

    def build_model(self, classes: Sequence[str]) -> SLMBase:
        """Instantiates `SLMJson` or `SLMCls` and applies LoRA + Gradient Checkpointing."""
        log_gpu_memory("Before model load")

        model: SLMBase
        if self.config.mode == "json":
            model = SLMJson.from_pretrained(
                model_name=self.model_name,
                classes=classes,
                multi_label=self.multi_label,
                prompts_config_path=self.config.prompts_config_path,
                max_length=self.max_length,
                factory=self.factory,
            )
        elif self.config.mode == "cls":
            model = SLMCls.from_pretrained(
                model_name=self.model_name,
                classes=classes,
                multi_label=self.multi_label,
                pooling_strategy=self.config.pooling_strategy,
                prompts_config_path=self.config.prompts_config_path,
                max_length=self.max_length,
                factory=self.factory,
            )
        else:
            raise ValueError(f"Unsupported mode '{self.config.mode}'. Choose 'json' or 'cls'.")

        self.tokenizer = model.tokenizer

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
        report_to = (
            configure_mlflow_env(
                mlflow_cfg=self.mlflow_cfg,
                tracking_uri_override=self.config.mlflow_tracking_uri,
                experiment_name_override=self.config.mlflow_experiment_name,
            )
            if self.config.use_mlflow
            else ["none"]
        )
        run_name = (
            self.config.run_name or f"easyslm-{self.config.mode}-{Path(self.model_name).name}"
        )
        return self.config.to_training_arguments(
            num_train_samples=num_train_samples,
            has_eval_dataset=has_eval_dataset,
            report_to=report_to,
            run_name=run_name,
        )

    def evaluate(self, eval_dataset: Dataset | None = None) -> dict[str, Any]:
        """Runs a single-pass evaluation and prediction on the validation dataset

        via `Trainer.predict(..., metric_key_prefix="eval")`, logs final metrics,
        and generates visual and tabular evaluation artifacts for MLflow and local disk.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer has not been initialized. Call train() first.")

        dataset = eval_dataset if eval_dataset is not None else self.trainer.eval_dataset
        if dataset is None or not isinstance(dataset, Dataset):
            logger.info("No evaluation dataset available; skipping evaluation.")
            return {}

        logger.info(
            "Running final evaluation on validation set (%d samples)...",
            len(dataset),  # type: ignore[arg-type]
        )
        pred_output = self.trainer.predict(dataset, metric_key_prefix="eval")
        metrics: dict[str, Any] = dict(pred_output.metrics or {})
        self.trainer.log(metrics)
        logger.info("Validation Evaluation Metrics: %s", metrics)

        try:
            (
                y_true,
                y_pred,
                input_texts,
                raw_outputs,
                parsed_outputs,
                error_scores,
            ) = extract_validation_prediction_details(
                pred_output=pred_output,
                dataset=dataset,
                mode=self.config.mode,
                classes=self.classes,
                multi_label=self.multi_label,
                tokenizer=self.tokenizer,
            )

            validation_errors_tbl = create_validation_errors_table(
                input_texts=input_texts,
                y_true=y_true,
                y_pred=y_pred,
                classes=self.classes,
                raw_outputs=raw_outputs,
                parsed_outputs=parsed_outputs,
                error_scores=error_scores,
                max_errors_per_class=self.config.max_errors_per_class,
            )

            log_evaluation_artifacts(
                y_true=y_true,
                y_pred=y_pred,
                classes=self.classes,
                multi_label=self.multi_label,
                local_output_dir=self.run_output_dir,
                validation_errors_table=validation_errors_tbl,
            )
        except Exception as e:
            logger.warning("Failed generating visual/tabular evaluation artifacts: %s", e)

        return metrics

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

        set_seed(self.config.seed)
        model = self.build_model(classes=self.classes)
        collator = base_dataset.get_collator(padding_side="right", device=None)
        training_args = self.build_training_arguments(
            num_train_samples=len(train_dataset),
            has_eval_dataset=eval_dataset is not None,
        )

        as_of = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = clean_slug(self.model_name)
        run_output_dir = Path(self.config.output_dir) / f"{model_slug}_{self.config.mode}_{as_of}"
        self.run_output_dir = run_output_dir

        assert self.tokenizer is not None
        if self.config.mode == "cls":
            compute_metrics_fn = build_cls_compute_metrics(multi_label=self.multi_label)
            preprocess_logits_fn = None
        else:
            compute_metrics_fn = build_json_compute_metrics(
                tokenizer=self.tokenizer,
                classes=self.classes,
                multi_label=self.multi_label,
            )
            preprocess_logits_fn = preprocess_logits_for_json_metrics

        callbacks: list[TrainerCallback] = []
        if (
            self.config.early_stopping_window is not None
            and self.config.early_stopping_window > 0
            and eval_dataset is not None
        ):
            callbacks.append(
                EarlyStoppingWindowCallback(
                    window_steps=self.config.early_stopping_window,
                    metric_name=self.config.early_stopping_metric,
                )
            )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics_fn,
            preprocess_logits_for_metrics=preprocess_logits_fn,
            callbacks=callbacks,
        )

        logger.info(
            "Starting fine-tuning via Hugging Face Trainer (mode=%s)...", self.config.mode.upper()
        )

        eval_metrics: dict[str, Any] = {}

        run_ctx: contextlib.AbstractContextManager[Any] = contextlib.nullcontext()
        if self.config.use_mlflow:
            import mlflow

            sys_metrics_enabled = self.mlflow_cfg.get("tracking", {}).get("system_metrics", True)
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])
            run_ctx = mlflow.start_run(
                run_name=training_args.run_name,
                log_system_metrics=sys_metrics_enabled,
            )

        with run_ctx:
            train_result = self.trainer.train()
            train_metrics = train_result.metrics
            logger.info("Training completed | Train Metrics: %s", train_metrics)
            log_gpu_memory("Post-training")

            eval_metrics = self.evaluate(eval_dataset)

        training_params = {
            "num_train_epochs": self.config.num_train_epochs,
            "learning_rate": self.config.learning_rate or (2e-4 if self.config.use_lora else 2e-5),
            "batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "use_lora": self.config.use_lora,
            "lora_r": self.config.lora_r if self.config.use_lora else None,
            "lora_alpha": self.config.lora_alpha if self.config.use_lora else None,
            "pooling_strategy": self.config.pooling_strategy if self.config.mode == "cls" else None,
            "max_length": self.max_length,
            "dataset": str(train_data_path),
        }

        assert self.model is not None
        checkpoint_path = self.model.save(
            output_dir=run_output_dir,
            base_model_name=self.model_name,
            tokenizer=self.tokenizer,
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
