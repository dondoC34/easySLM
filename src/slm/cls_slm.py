from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.slm.base_slm import SLMBase
from src.slm.slm_factory import ModelFactory

PoolingStrategy = Literal["eos", "mean", "max"]


class SLMCls(SLMBase):
    """
    Direct Classification SLM with Parametric Pooling and Dense Head (Mode 1.2).

    Architecture:
    - Backbone: AutoModel (Transformer backbone returning `last_hidden_state`).
    - Pooling: Parametric pooling over sequence hidden states:
        * `'eos'` (default): extracts the hidden state of the final valid (non-pad) prompt token.
        * `'mean'`: attention-mask-weighted mean pooling across valid tokens.
        * `'max'`: attention-mask-aware max pooling across valid tokens.
    - Normalization & Dense Layer: `nn.LayerNorm(hidden_size)` followed by `nn.Linear(hidden_size, num_classes)`
      (kept in `float32` for numerical stability under mixed-precision AdamW updates) projecting
      normalized pooled states to class logits.
    - Inputs (prepared by `SLMClsDataset` + `SLMCollator`):
        * `input_ids`: `(batch_size, seq_len)`
        * `attention_mask`: `(batch_size, seq_len)`
        * `labels`: `(batch_size, num_classes)` binary 0/1 tensor.
    - Loss & Probabilities:
        * `multi_label=False` (single-label): `nn.CrossEntropyLoss` + `softmax` at inference.
        * `multi_label=True` (multi-label): `nn.BCEWithLogitsLoss` + `sigmoid` at inference (no softmax).
    """

    TASK_MODE: Literal["cls", "json"] = "cls"
    VALID_POOLING_STRATEGIES = ("eos", "mean", "max")

    def __init__(
        self,
        backbone: PreTrainedModel,
        classes: Sequence[str],
        multi_label: bool = True,
        pooling_strategy: PoolingStrategy = "eos",
        dropout: float = 0.1,
        tokenizer: PreTrainedTokenizerBase | None = None,
        prompts_config_path: str | Path = "configs/prompts.yml",
        max_length: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if pooling_strategy not in self.VALID_POOLING_STRATEGIES:
            raise ValueError(
                f"Invalid pooling_strategy '{pooling_strategy}'. "
                f"Choose from {self.VALID_POOLING_STRATEGIES}."
            )

        self.pooling_strategy: PoolingStrategy = pooling_strategy
        super().__init__(
            backbone=backbone,
            classes=classes,
            multi_label=multi_label,
            tokenizer=tokenizer,
            prompts_config_path=prompts_config_path,
            max_length=max_length,
            metadata=metadata,
        )

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not determine `hidden_size` from `backbone.config`.")

        self.norm = nn.LayerNorm(
            hidden_size,
            device=self.device,
            dtype=torch.float32,
        )
        nn.init.constant_(self.norm.weight, 0.5)
        nn.init.zeros_(self.norm.bias)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            hidden_size,
            self.num_classes,
            device=self.device,
            dtype=torch.float32,
        )
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fn: nn.Module = (
            nn.BCEWithLogitsLoss() if self.multi_label else nn.CrossEntropyLoss()
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        classes: Sequence[str],
        multi_label: bool = True,
        pooling_strategy: PoolingStrategy = "eos",
        dropout: float = 0.1,
        prompts_config_path: str | Path = "configs/prompts.yml",
        max_length: int | None = None,
        factory: ModelFactory | None = None,
        **model_kwargs,
    ) -> "SLMCls":
        """
        Instantiates `SLMCls` (with `model.tokenizer` attached) using `ModelFactory`.
        """
        factory = factory or ModelFactory()
        backbone, tokenizer = factory.get_backbone(model_name, **model_kwargs)
        return cls(
            backbone=backbone,
            classes=classes,
            multi_label=multi_label,
            pooling_strategy=pooling_strategy,
            dropout=dropout,
            tokenizer=tokenizer,
            prompts_config_path=prompts_config_path,
            max_length=max_length,
            metadata={"base_model_name": model_name, "mode": "cls"},
        )

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pools sequence hidden states `(batch_size, seq_len, hidden_size)` into `(batch_size, hidden_size)`
        according to `self.pooling_strategy`:
        - `'eos'`: extracts the hidden state at the last valid (non-pad) token index.
        - `'mean'`: attention-mask-weighted average pooling.
        - `'max'`: attention-mask-aware max pooling.
        """
        batch_size, seq_len, _ = hidden_states.shape

        if self.pooling_strategy == "eos":
            positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
            last_token_indices = (positions * attention_mask).argmax(dim=1)
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            return hidden_states[batch_indices, last_token_indices]

        elif self.pooling_strategy == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling_strategy == "max":
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            min_val = torch.finfo(hidden_states.dtype).min
            masked_hidden = hidden_states.masked_fill(~mask_expanded, min_val)
            return masked_hidden.max(dim=1).values

        raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes classification loss in float32 for numerical stability:
        - `multi_label=True`: `BCEWithLogitsLoss(logits, labels.float())` where `labels` is a 0/1 tensor `(B, C)`.
        - `multi_label=False`: `CrossEntropyLoss(logits, target_indices)` where 0/1 one-hot `labels`
          are converted to class indices via `argmax(dim=-1)`.
        """
        logits_fp32 = logits.float()
        if self.multi_label:
            return self.loss_fn(logits_fp32, labels.float())
        else:
            if labels.ndim > 1 and labels.shape[-1] == self.num_classes:
                labels = labels.argmax(dim=-1)
            return self.loss_fn(logits_fp32, labels.long())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_pooled: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass through the backbone, parametric pooling layer, LayerNorm, and dense classification head.
        Accepts exclusively tensor inputs (`input_ids`, `attention_mask`, and optional `labels`).
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]

        pooled = self.pool_hidden_states(hidden_states, attention_mask)
        normed = self.norm(pooled.float())
        logits = self.classifier(self.dropout(normed))

        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.compute_loss(logits, labels)

        result: dict[str, Any] = {
            "loss": loss,
            "logits": logits,
        }
        if return_pooled:
            result["pooled_states"] = pooled
        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
        fallback_top1: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Runs classification inference on tokenized `(input_ids, attention_mask)` tensors.

        - When `multi_label=False`: applies `softmax` over logits and selects `argmax` into a 0/1 vector.
        - When `multi_label=True`: applies `sigmoid` over logits (no softmax) and thresholds into a 0/1 vector
          (with optional `fallback_top1` if no class exceeds `threshold`).
        """
        was_training = self.training
        self.eval()

        forward_out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = forward_out["logits"].float()

        if self.multi_label:
            probabilities = torch.sigmoid(logits)
            binary_vectors = (probabilities >= threshold).long()
            if fallback_top1:
                empty_mask = binary_vectors.sum(dim=-1) == 0
                if empty_mask.any():
                    top_indices = probabilities.argmax(dim=-1, keepdim=True)
                    fallback_vecs = torch.zeros_like(binary_vectors)
                    fallback_vecs.scatter_(1, top_indices, 1)
                    binary_vectors = torch.where(
                        empty_mask.unsqueeze(-1), fallback_vecs, binary_vectors
                    )
        else:
            probabilities = torch.softmax(logits, dim=-1)
            top_indices = probabilities.argmax(dim=-1, keepdim=True)
            binary_vectors = torch.zeros_like(probabilities, dtype=torch.long)
            binary_vectors.scatter_(1, top_indices, 1)

        predictions: list[list[str]] = []
        for row in binary_vectors.cpu().tolist():
            pred_classes = [self.idx_to_class[i] for i, val in enumerate(row) if val == 1]
            predictions.append(sorted(pred_classes))

        if was_training:
            self.train()

        return {
            "logits": logits,
            "probabilities": probabilities,
            "binary_vectors": binary_vectors,
            "predictions": predictions,
        }
