import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from transformers import PreTrainedTokenizerBase

from src.data.base_dataset import SLMDatasetBase
from src.data.collator import SLMCollator


class SLMJsonDataset(SLMDatasetBase):
    """
    Dataset for Generative JSON Classification (`SLMJson`).

    For each datapoint `(text, labels)`:
    - Formats the target `labels` into a sorted JSON-style list string, e.g. `"['economy', 'politics']"`.
    - Builds the plain-text prompt prefix from `configs/prompts.yml` and appends the target JSON list
      followed immediately by the tokenizer's native `eos_token_id` (when `include_targets=True`).
    - Masks all prompt tokens with `-100` in `labels` so the CausalLM CrossEntropyLoss optimizes
      exclusively on the target JSON list + native `eos_token_id`.
    """

    TASK_MODE = "json"
    DEFAULT_PROMPT_TEMPLATE = "{system_prompt}\n\nText: {user_prompt}\nLabels:\n"

    def __init__(
        self,
        data_path: str | Path | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        classes: Sequence[str] | None = None,
        multi_label: bool | None = None,
        system_prompt: str | None = None,
        max_length: int | None = None,
        include_targets: bool = True,
        single_quotes: bool = True,
        prompts_config_path: str | Path = "configs/prompts.yml",
        user_prompt_template: str | None = None,
        prompt_template: str | None = None,
        texts: Sequence[str] | None = None,
    ):
        self.single_quotes = single_quotes
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            classes=classes,
            multi_label=multi_label,
            system_prompt=system_prompt,
            max_length=max_length,
            include_targets=include_targets,
            prompts_config_path=prompts_config_path,
            user_prompt_template=user_prompt_template,
            prompt_template=prompt_template,
            texts=texts,
        )

    def format_labels_as_string(self, labels: Sequence[str]) -> str:
        """
        Formats sorted labels into the exact string output expected from the model,
        e.g. `"['economy', 'politics']"`.
        """
        sorted_labels = sorted(list(dict.fromkeys(labels)))
        if self.single_quotes:
            inner = ", ".join(f"'{lbl}'" for lbl in sorted_labels)
            return f"[{inner}]"
        return json.dumps(sorted_labels)

    def _build_default_system_prompt(self) -> str:
        classes_str = ", ".join(f"'{c}'" for c in self.classes)
        example_labels = self.classes[: min(2, len(self.classes))]
        if not self.multi_label:
            example_labels = self.classes[:1]
        example_str = self.format_labels_as_string(example_labels)

        mode_descriptions = getattr(self, "prompts_config", {}).get("mode_descriptions", {})
        default_mode_desc = (
            mode_descriptions.get("multi_label", "one or more applicable classes")
            if self.multi_label
            else mode_descriptions.get("single_label", "the single most appropriate class")
        )
        json_cfg = getattr(self, "prompts_config", {}).get("json", {})
        template = json_cfg.get(
            "system_prompt",
            "You are a text classification model. "
            "Classify the user text into {mode_desc} from the following valid classes: [{classes}]. "
            "Respond ONLY with a valid list of predicted class names sorted in alphabetical order, "
            "for example: {example}. Do not output any additional text or explanation.",
        )
        return template.format(
            classes=classes_str,
            mode_desc=default_mode_desc,
            example=example_str,
        ).strip()

    def _encode_sample(
        self,
        text: str,
        labels: list[str],
    ) -> dict[str, torch.Tensor]:
        prompt_str = self.build_prompt(text)
        prompt_ids = self.encode_text_ids(prompt_str, add_special_tokens=True)

        if self.include_targets:
            assistant_str = self.format_labels_as_string(labels)
            target_ids = self.encode_text_ids(assistant_str, add_special_tokens=False)
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None and (not target_ids or target_ids[-1] != eos_token_id):
                target_ids = target_ids + [eos_token_id]

            full_ids = prompt_ids + target_ids
            # Set -100 on all prompt tokens so loss is computed strictly on target_ids (JSON list + eos_token)
            label_ids = [-100] * len(prompt_ids) + target_ids

            if self.max_length is not None:
                full_ids = full_ids[: self.max_length]
                label_ids = label_ids[: self.max_length]

            attention_mask = [1] * len(full_ids)
            return {
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(label_ids, dtype=torch.long),
            }
        else:
            if self.max_length is not None:
                prompt_ids = prompt_ids[: self.max_length]
            attention_mask = [1] * len(prompt_ids)
            return {
                "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    def get_collator(
        self,
        padding_side: Literal["right", "left"] | None = None,
        device: str | torch.device | None = None,
    ) -> SLMCollator:
        effective_side: Literal["right", "left"] = padding_side or (
            "right" if self.include_targets else "left"
        )
        return SLMCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            pad_labels=True,
            label_pad_token_id=-100,
            padding_side=effective_side,
            device=device,
        )
