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
    - Builds `[system, user, assistant]` chat messages (or `[system, user]` when `include_targets=False`).
    - Applies the tokenizer's chat template to produce `input_ids` and `attention_mask`.
    - Masks all `system`, `user`, and assistant header tokens with `-100` in `labels` so the
      CausalLM CrossEntropyLoss optimizes exclusively on the assistant's response tokens.
    """

    TASK_MODE = "json"

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        classes: Sequence[str] | None = None,
        multi_label: bool = True,
        system_prompt: str | None = None,
        max_length: int | None = None,
        include_targets: bool = True,
        single_quotes: bool = True,
        chat_template_config_path: str | Path = "configs/chat_template.yml",
        chat_template_name: str | None = None,
        prompts_config_path: str | Path = "configs/prompts.yml",
        user_prompt_template: str | None = None,
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
            chat_template_config_path=chat_template_config_path,
            chat_template_name=chat_template_name,
            prompts_config_path=prompts_config_path,
            user_prompt_template=user_prompt_template,
        )

    def format_labels_as_string(self, labels: Sequence[str]) -> str:
        """
        Formats sorted labels into the exact string output expected from the assistant,
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
        # Build [system, user] prompt and tokenize with add_generation_prompt=True
        prompt_messages = self.build_chat_messages(user_text=text, assistant_text=None)
        prompt_ids = self.render_chat_ids(prompt_messages, add_generation_prompt=True)

        if self.include_targets:
            assistant_str = self.format_labels_as_string(labels)
            full_messages = self.build_chat_messages(user_text=text, assistant_text=assistant_str)
            full_ids = self.render_chat_ids(full_messages, add_generation_prompt=False)

            prompt_len = len(prompt_ids)
            # Set -100 on all system, user, and assistant header tokens
            label_ids = [-100] * prompt_len + full_ids[prompt_len:]

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
