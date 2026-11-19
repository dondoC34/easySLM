import ast
import json
import re
from collections.abc import Container, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.slm.base_slm import SLMBase
from src.slm.slm_factory import ModelFactory


class SLMJson(SLMBase):
    """
    Generative JSON Classification SLM (Mode 1.1).

    Architecture:
    - Backbone: AutoModelForCausalLM (Transformer backbone + vocabulary linear layer `lm_head`).
    - Inputs (prepared by `SLMJsonDataset` + `SLMCollator`):
        * `input_ids`: `(batch_size, seq_len)`
        * `attention_mask`: `(batch_size, seq_len)`
        * `labels`: `(batch_size, seq_len)` with `-100` on all tokens that should be ignored
          by the loss (system prompt, user message, assistant turn header, and padding).
    - Loss function: Next-token `nn.CrossEntropyLoss(ignore_index=-100)` computed strictly on
      the unmasked assistant response tokens.
    """

    TASK_MODE: Literal["cls", "json"] = "json"

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
        super().__init__(
            backbone=backbone,
            classes=classes,
            multi_label=multi_label,
            tokenizer=tokenizer,
            prompts_config_path=prompts_config_path,
            max_length=max_length,
            metadata=metadata,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        classes: Sequence[str],
        multi_label: bool = True,
        prompts_config_path: str | Path = "configs/prompts.yml",
        max_length: int | None = None,
        factory: ModelFactory | None = None,
        **model_kwargs,
    ) -> "SLMJson":
        """
        Instantiates `SLMJson` (with `model.tokenizer` attached) using `ModelFactory`.
        """
        factory = factory or ModelFactory()
        backbone, tokenizer = factory.get_causal_lm(model_name, **model_kwargs)
        return cls(
            backbone=backbone,
            classes=classes,
            multi_label=multi_label,
            tokenizer=tokenizer,
            prompts_config_path=prompts_config_path,
            max_length=max_length,
            metadata={"base_model_name": model_name, "mode": "json"},
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes causal autoregressive CrossEntropyLoss strictly on the unmasked assistant tokens (`labels != -100`).
        Shifts logits left by 1 and labels right by 1 so token `t` predicts token `t + 1`.
        Filters to valid target positions before casting to float32 to avoid materializing a full
        `(batch_size, seq_len, vocab_size)` float32 tensor in GPU memory.
        """
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        valid_mask = shift_labels != -100
        if not valid_mask.any():
            return shift_logits.sum() * 0.0
        return self.loss_fn(
            shift_logits[valid_mask].float(),
            shift_labels[valid_mask],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_token_ids: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass through the CausalLM backbone.
        Accepts exclusively tensor inputs (`input_ids`, `attention_mask`, and optional `labels`).
        When `return_token_ids=True` (or `self._return_token_ids_on_eval` is active during evaluation),
        returns `logits.argmax(dim=-1)` inside `"logits"` before Accelerate's `convert_outputs_to_fp32`
        wrapper can upcast the full `(batch_size, seq_len, vocab_size)` tensor to float32.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **kwargs,
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.compute_loss(logits, labels)

        if return_token_ids or (
            not self.training and getattr(self, "_return_token_ids_on_eval", False)
        ):
            logits = logits.argmax(dim=-1)

        return {
            "loss": loss,
            "logits": logits,
        }

    @staticmethod
    def parse_class_list(
        raw_text: str,
        valid_classes: Container[str],
    ) -> tuple[list[str], bool]:
        """
        Extracts and parses a JSON/Python list of predicted class names from a generated string.
        Returns `(sorted_valid_classes, is_valid_json_list)`.
        """
        text = raw_text.strip()
        match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
        candidate = match.group(0) if match else text

        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                break
            except Exception:
                continue

        if isinstance(parsed, list):
            valid = sorted(
                list(dict.fromkeys(str(item) for item in parsed if str(item) in valid_classes))
            )
            return valid, True
        return [], False

    def parse_json_output(self, raw_text: str) -> list[str]:
        """
        Extracts and parses a JSON/Python list of predicted class names from a generated string.
        Filters and returns valid class names sorted alphabetically.
        """
        valid, _ = self.parse_class_list(raw_text, self.class_to_idx)
        return valid

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **generate_kwargs,
    ) -> dict[str, Any]:
        """
        Autoregressively generates the assistant tokens given `(input_ids, attention_mask)`.
        Uses `tokenizer` (or `self.tokenizer`) to decode the generated tokens into strings
        and parse the JSON class lists.
        """
        was_training = self.training
        self.eval()

        active_tokenizer = tokenizer or self.tokenizer

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_len = input_ids.shape[1]

        pad_token_id = getattr(self.backbone.config, "pad_token_id", None)
        if pad_token_id is None and active_tokenizer is not None:
            pad_token_id = active_tokenizer.pad_token_id

        stop_token_ids = []
        eos_id = getattr(self.backbone.config, "eos_token_id", None)
        if isinstance(eos_id, int):
            stop_token_ids.append(eos_id)
        elif isinstance(eos_id, list):
            stop_token_ids.extend(eos_id)

        if active_tokenizer is not None:
            if (
                active_tokenizer.eos_token_id is not None
                and active_tokenizer.eos_token_id not in stop_token_ids
            ):
                stop_token_ids.append(active_tokenizer.eos_token_id)

        gen_kwargs = dict(generate_kwargs)

        generated_ids = cast(Any, self.backbone).generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=stop_token_ids if stop_token_ids else None,
            **gen_kwargs,
        )

        assistant_token_ids = generated_ids[:, prompt_len:]
        result: dict[str, Any] = {
            "generated_token_ids": assistant_token_ids,
        }

        if active_tokenizer is not None:
            raw_outputs = active_tokenizer.batch_decode(
                assistant_token_ids, skip_special_tokens=True
            )
            result["raw_outputs"] = [r.strip() for r in raw_outputs]
            result["predictions"] = [self.parse_json_output(raw) for raw in raw_outputs]

        if was_training:
            self.train()

        return result
