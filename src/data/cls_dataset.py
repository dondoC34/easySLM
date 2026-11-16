from collections.abc import Sequence
from typing import Literal

import torch

from src.data.base_dataset import SLMDatasetBase
from src.data.collator import SLMCollator


class SLMClsDataset(SLMDatasetBase):
    """
    Dataset for Direct Classification (`SLMCls`).

    For each datapoint `(text, labels)`:
    - Builds the plain-text prompt from `configs/prompts.yml`.
    - Tokenizes the prompt without appending a document-reset EOS token at the end,
      preserving the causal LM representation at the final prompt token (`Label:`).
    - Encodes the target `labels` as a binary 0/1 vector of shape `(num_classes,)` (`torch.float32`).
    """

    TASK_MODE = "cls"
    DEFAULT_PROMPT_TEMPLATE = "{system_prompt}\n\nText: {user_prompt}\nLabel:"

    def _build_default_system_prompt(self) -> str:
        classes_str = ", ".join(f"'{c}'" for c in self.classes)
        mode_descriptions = getattr(self, "prompts_config", {}).get("mode_descriptions", {})
        default_mode_desc = (
            mode_descriptions.get("multi_label", "one or more applicable classes (multi-label)")
            if self.multi_label
            else mode_descriptions.get(
                "single_label", "the single most appropriate class (single-label)"
            )
        )
        cls_cfg = getattr(self, "prompts_config", {}).get("cls", {})
        template = cls_cfg.get(
            "system_prompt",
            "You are an expert text classification assistant. "
            "Classify the user text into {mode_desc} from the following valid classes: [{classes}].",
        )
        return template.format(classes=classes_str, mode_desc=default_mode_desc).strip()

    def encode_labels_to_binary_vector(self, labels: Sequence[str]) -> torch.Tensor:
        """Encodes a list of class names into a 0/1 float tensor of shape `(num_classes,)`."""
        vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for lbl in labels:
            vec[self.class_to_idx[lbl]] = 1.0
        return vec

    def _encode_sample(
        self,
        text: str,
        labels: list[str],
    ) -> dict[str, torch.Tensor]:
        limit = self._effective_max_length()
        prompt_str = self.build_prompt(text)
        token_ids = self.encode_text_ids(prompt_str, add_special_tokens=True)

        if limit is not None and len(token_ids) > limit:
            truncated_text = self._truncate_text_to_fit(text, max_tokens=limit)
            prompt_str = self.build_prompt(truncated_text)
            token_ids = self.encode_text_ids(prompt_str, add_special_tokens=True)[:limit]

        attention_mask = [1] * len(token_ids)
        sample_dict: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

        if self.include_targets:
            sample_dict["labels"] = self.encode_labels_to_binary_vector(labels)

        return sample_dict

    def get_collator(
        self,
        padding_side: Literal["right", "left"] | None = None,
        device: str | torch.device | None = None,
    ) -> SLMCollator:
        effective_side: Literal["right", "left"] = padding_side or "right"
        return SLMCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            pad_labels=False,
            padding_side=effective_side,
            device=device,
        )
