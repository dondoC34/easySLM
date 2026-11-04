from collections.abc import Sequence
from typing import Literal

import torch


class SLMCollator:
    """
    Collate function for `SLMJsonDataset` and `SLMClsDataset`.

    Pads variable-length `input_ids` and `attention_mask` tensors across a batch:
    - `input_ids` is padded with `pad_token_id`.
    - `attention_mask` is padded with `0`.
    - `labels`:
        * When `pad_labels=True` (JSON CausalLM mode): variable-length 1D token label sequences
          are padded with `label_pad_token_id` (`-100`) so padded positions are ignored by CrossEntropyLoss.
        * When `pad_labels=False` (CLS mode): fixed-size 0/1 binary label vectors of shape `(num_classes,)`
          are stacked into `(batch_size, num_classes)`.
    """

    def __init__(
        self,
        pad_token_id: int,
        pad_labels: bool = False,
        label_pad_token_id: int = -100,
        padding_side: Literal["right", "left"] = "right",
        device: str | torch.device | None = None,
    ):
        if padding_side not in ("right", "left"):
            raise ValueError(f"padding_side must be 'right' or 'left', got '{padding_side}'.")
        self.pad_token_id = pad_token_id
        self.pad_labels = pad_labels
        self.label_pad_token_id = label_pad_token_id
        self.padding_side = padding_side
        self.device = torch.device(device) if device is not None else None

    def _pad_1d_sequence(
        self,
        tensors: Sequence[torch.Tensor],
        pad_value: int,
        max_len: int,
    ) -> torch.Tensor:
        batch_size = len(tensors)
        dtype = tensors[0].dtype
        out = torch.full((batch_size, max_len), fill_value=pad_value, dtype=dtype)
        for i, t in enumerate(tensors):
            seq_len = t.size(0)
            if self.padding_side == "right":
                out[i, :seq_len] = t
            else:
                out[i, max_len - seq_len :] = t
        return out

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not batch:
            return {}

        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        max_len = max(t.size(0) for t in input_ids_list)

        padded_input_ids = self._pad_1d_sequence(input_ids_list, self.pad_token_id, max_len)
        padded_attention_mask = self._pad_1d_sequence(attention_mask_list, 0, max_len)

        collated: dict[str, torch.Tensor] = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }

        if "labels" in batch[0] and batch[0]["labels"] is not None:
            labels_list = [item["labels"] for item in batch]
            if self.pad_labels:
                collated["labels"] = self._pad_1d_sequence(
                    labels_list,
                    self.label_pad_token_id,
                    max_len,
                )
            else:
                collated["labels"] = torch.stack(labels_list, dim=0)

        if self.device is not None:
            collated = {k: v.to(self.device) for k, v in collated.items()}

        return collated
