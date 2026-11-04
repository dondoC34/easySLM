import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class SLMBase(nn.Module, ABC):
    """
    Abstract base class for Small Language Model (SLM) classifiers.

    Holds the shared `backbone` model (produced by `ModelFactory`) and the target `classes` metadata.
    All text tokenization, chat template rendering, and label encoding happen inside `easyslm.data` datasets.
    Models receive exclusively tensor inputs: `input_ids`, `attention_mask`, and `labels`.

    Subclasses must implement:
    - `forward`: executes the model forward pass on `(input_ids, attention_mask, labels)`.
    - `compute_loss`: computes the task-specific loss function.
    - `predict`: runs inference on `(input_ids, attention_mask)`.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        classes: Sequence[str],
        multi_label: bool = True,
    ):
        super().__init__()
        if not classes:
            raise ValueError("`classes` must be a non-empty sequence of class names.")

        self.backbone = backbone
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
        self.config = getattr(self.backbone, "config", None)

    @property
    def device(self) -> torch.device:
        """Returns the device where the backbone resides."""
        return next(self.backbone.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the floating-point dtype of the backbone parameters."""
        return next(self.backbone.parameters()).dtype

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
