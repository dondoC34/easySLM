from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.data.base_dataset import SLMDatasetBase
from src.data.cls_dataset import SLMClsDataset
from src.data.json_dataset import SLMJsonDataset
from src.utils.model_card import find_best_model

if TYPE_CHECKING:
    from src.slm.slm_factory import ModelFactory

logger = logging.getLogger(__name__)


class SLMBase(nn.Module, ABC):
    """
    Abstract base class for Small Language Model (SLM) classifiers.

    Holds the shared `backbone` model (produced by `ModelFactory`), the associated `tokenizer`,
    and the target `classes` metadata.

    Provides:
    - Checkpoint loading (`from_checkpoint`, `from_best_model`) and saving (`save`).
    - High-level inference (`predict_texts`, `predict_lines`, `predict_file_or_dir`) that reuses
      `SLMClsDataset` / `SLMJsonDataset` so training and inference share identical prompt and
      tokenization logic.
    - Abstract tensor-level methods (`forward`, `compute_loss`, `predict`) implemented by
      subclasses (`SLMCls` and `SLMJson`).
    """

    TASK_MODE: Literal["cls", "json"] = "cls"

    def __init__(
        self,
        backbone: PreTrainedModel,
        classes: Sequence[str],
        multi_label: bool = True,
        tokenizer: PreTrainedTokenizerBase | None = None,
        prompts_config_path: str | Path = "configs/prompts.yml",
        max_length: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__()
        if not classes:
            raise ValueError("`classes` must be a non-empty sequence of class names.")

        self.backbone = backbone
        self.tokenizer = tokenizer
        self.classes: list[str] = sorted(
            list(dict.fromkeys(c.strip() for c in classes if c.strip()))
        )
        self.num_classes: int = len(self.classes)
        self.class_to_idx: dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }
        self.idx_to_class: dict[int, str] = {
            idx: cls_name for idx, cls_name in enumerate(self.classes)
        }
        self.multi_label: bool = multi_label
        self.prompts_config_path: str | Path = prompts_config_path
        self.max_length: int | None = max_length
        self.metadata: dict[str, Any] = metadata or {}
        self.config = getattr(self.backbone, "config", None)

    @property
    def mode(self) -> Literal["cls", "json"]:
        """Returns the architecture task mode ('cls' or 'json')."""
        return self.TASK_MODE

    @property
    def device(self) -> torch.device:
        """Returns the device where the backbone resides."""
        return next(self.backbone.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the floating-point dtype of the backbone parameters."""
        return next(self.backbone.parameters()).dtype

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        prompts_config_path: str | Path = "configs/prompts.yml",
        factory: ModelFactory | None = None,
    ) -> SLMBase:
        """
        Loads a fine-tuned `SLMCls` or `SLMJson` model (along with its tokenizer and metadata)
        directly from a checkpoint directory.
        """
        from src.slm.serialization import load_finetuned_model

        return load_finetuned_model(
            checkpoint_dir=checkpoint_dir,
            prompts_config_path=prompts_config_path,
            factory=factory,
        )

    @classmethod
    def from_best_model(
        cls,
        models_dir: str | Path = "./models/finetuned",
        model_type: str = "cls",
        metric: str = "eval_f1_macro",
        prompts_config_path: str | Path = "configs/prompts.yml",
        factory: ModelFactory | None = None,
    ) -> SLMBase:
        """
        Discovers the best model in `models_dir` using its model cards and returns the loaded
        `SLMBase` instance (`SLMCls` or `SLMJson`), storing `"checkpoint_dir"` and `"model_card"`
        inside `model.metadata`.
        """
        best_dir, best_card = find_best_model(models_dir, model_type=model_type, metric=metric)
        model = cls.from_checkpoint(
            checkpoint_dir=best_dir,
            prompts_config_path=prompts_config_path,
            factory=factory,
        )
        model.metadata["checkpoint_dir"] = str(best_dir)
        model.metadata["model_card"] = best_card
        return model

    def save(
        self,
        output_dir: str | Path,
        base_model_name: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        training_params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        as_of: str | None = None,
    ) -> Path:
        """Saves the fine-tuned model, tokenizer, metadata config, and Markdown Model Card."""
        from src.slm.serialization import save_finetuned_model

        return save_finetuned_model(
            model=self,
            tokenizer=tokenizer or self.tokenizer,
            output_dir=output_dir,
            base_model_name=base_model_name,
            training_params=training_params,
            metrics=metrics,
            as_of=as_of,
        )

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Enables gradient checkpointing on the underlying Hugging Face backbone to save GPU VRAM."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        if self.config is not None and hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        logger.info("Enabled gradient checkpointing on %s backbone.", type(self.backbone).__name__)

    def gradient_checkpointing_disable(self) -> None:
        """Disables gradient checkpointing on the underlying Hugging Face backbone."""
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()
        if self.config is not None and hasattr(self.config, "use_cache"):
            self.config.use_cache = True

    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the task-specific loss given model logits and target labels."""
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Executes the forward pass on tokenized tensors and optionally computes the loss."""
        pass

    @abstractmethod
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> dict[str, Any]:
        """Runs inference on tokenized `(input_ids, attention_mask)` tensors."""
        pass

    @torch.no_grad()
    def predict_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        **predict_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Executes batch inference on a sequence of raw texts by delegating prompt formatting,
        chat template rendering, and tokenization to `SLMClsDataset` / `SLMJsonDataset`.

        Returns:
            List of dictionaries `{"text": text, "predictions": list[str], ...}`.
        """
        results: list[dict[str, Any]] = []
        if not texts:
            return results

        if self.tokenizer is None:
            raise ValueError(
                "Model has no associated `tokenizer`. Pass `tokenizer` when initializing the model "
                "or load via `from_pretrained` / `from_checkpoint`."
            )

        dataset_cls: type[SLMDatasetBase] = SLMClsDataset if self.mode == "cls" else SLMJsonDataset
        dataset = dataset_cls(
            texts=texts,
            tokenizer=self.tokenizer,
            classes=self.classes,
            multi_label=self.multi_label,
            max_length=self.max_length,
            include_targets=False,
            prompts_config_path=self.prompts_config_path,
        )
        collator = dataset.get_collator(device=self.device)

        for i in range(0, len(dataset), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_samples = [dataset[j] for j in range(i, i + len(batch_texts))]
            batch = collator(batch_samples)

            pred_out = self.predict(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **predict_kwargs,
            )
            predictions = pred_out.get("predictions", [])

            if self.mode == "cls":
                probabilities = pred_out.get("probabilities", [])
                if isinstance(probabilities, torch.Tensor):
                    probabilities = probabilities.cpu().tolist()

                for j, t in enumerate(batch_texts):
                    results.append(
                        {
                            "text": t,
                            "predictions": predictions[j] if j < len(predictions) else [],
                            "probabilities": {
                                c: round(p, 4)
                                for c, p in zip(self.classes, probabilities[j], strict=False)
                            }
                            if len(probabilities) > j
                            else {},
                        }
                    )
            else:
                raw_outputs = pred_out.get("raw_outputs", [])
                for j, t in enumerate(batch_texts):
                    results.append(
                        {
                            "text": t,
                            "predictions": predictions[j] if j < len(predictions) else [],
                            "raw_output": raw_outputs[j] if j < len(raw_outputs) else "",
                        }
                    )

        return results

    def predict_lines(
        self,
        lines: Sequence[str],
        batch_size: int = 8,
        **predict_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Parses raw lines (extracting text before `'|||'` and optional ground-truth labels after `'|||'`)
        and runs batch inference via `predict_texts`.
        """
        extracted_data: list[tuple[str, str, list[str] | None]] = []
        for line in lines:
            parsed = SLMDatasetBase.parse_raw_line(line)
            if parsed is None:
                continue
            text, ground_truth = parsed
            extracted_data.append((line.strip(), text, ground_truth))

        clean_texts = [item[1] for item in extracted_data]
        preds = self.predict_texts(clean_texts, batch_size=batch_size, **predict_kwargs)

        final_records: list[dict[str, Any]] = []
        for (raw, _text, gt), pred in zip(extracted_data, preds, strict=False):
            record = dict(pred)
            record["raw_line"] = raw
            if gt is not None:
                record["ground_truth"] = gt
            final_records.append(record)

        return final_records

    def predict_file_or_dir(
        self,
        data_path: str | Path,
        output_dir: str | Path,
        batch_size: int = 8,
        **predict_kwargs: Any,
    ) -> list[Path]:
        """
        Processes a `.txt` file or all `.txt` files in a directory and writes predictions
        to `output_dir` in `.txt` format (`<text> ||| <comma_separated_labels>`) and `.json` format.

        Returns:
            List of generated output prediction files.
        """
        txt_files = SLMDatasetBase.resolve_txt_files(Path(data_path).resolve())
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[Path] = []

        for txt_file in txt_files:
            lines = txt_file.read_text(encoding="utf-8").splitlines()
            records = self.predict_lines(lines, batch_size=batch_size, **predict_kwargs)

            stem = txt_file.stem
            out_txt = out_dir / f"{stem}_predictions_{self.mode}.txt"
            out_json = out_dir / f"{stem}_predictions_{self.mode}.json"

            # 1. Write formatted .txt (<text> ||| <labels>)
            txt_content = []
            for r in records:
                preds_str = ", ".join(r["predictions"])
                txt_content.append(f"{r['text']} ||| {preds_str}")
            out_txt.write_text("\n".join(txt_content) + "\n", encoding="utf-8")

            # 2. Write structured .json
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

            logger.info(
                "Saved %d predictions to '%s' and '%s'", len(records), out_txt.name, out_json.name
            )
            saved_files.append(out_txt)
            saved_files.append(out_json)

        return saved_files
