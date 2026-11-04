import ast
import json
import re
from collections.abc import Sequence
from typing import Any, cast

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

    def __init__(
        self,
        backbone: PreTrainedModel,
        classes: Sequence[str],
        multi_label: bool = True,
    ):
        super().__init__(
            backbone=backbone,
            classes=classes,
            multi_label=multi_label,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        classes: Sequence[str],
        multi_label: bool = True,
        factory: ModelFactory | None = None,
        **model_kwargs,
    ) -> tuple["SLMJson", PreTrainedTokenizerBase]:
        """
        Instantiates `SLMJson` and returns `(model, tokenizer)` using `ModelFactory`.
        """
        factory = factory or ModelFactory()
        backbone, tokenizer = factory.get_causal_lm(model_name, **model_kwargs)
        model = cls(
            backbone=backbone,
            classes=classes,
            multi_label=multi_label,
        )
        return model, tokenizer

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes causal autoregressive CrossEntropyLoss strictly on the unmasked assistant tokens (`labels != -100`).
        Shifts logits left by 1 and labels right by 1 so token `t` predicts token `t + 1`.
        """
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        return self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Forward pass through the CausalLM backbone.
        Accepts exclusively tensor inputs (`input_ids`, `attention_mask`, and optional `labels`).
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            loss = self.compute_loss(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }

    def parse_json_output(self, raw_text: str) -> list[str]:
        """
        Extracts and parses a JSON/Python list of predicted class names from a generated string.
        Filters and returns valid class names sorted alphabetically.
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
            valid = [str(item) for item in parsed if str(item) in self.class_to_idx]
            return sorted(list(dict.fromkeys(valid)))
        return []

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
        If `tokenizer` is provided, also decodes the generated tokens into strings and parses the JSON lists.
        """
        was_training = self.training
        self.eval()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        prompt_len = input_ids.shape[1]

        pad_token_id = getattr(self.backbone.config, "pad_token_id", None)
        if pad_token_id is None and tokenizer is not None:
            pad_token_id = tokenizer.pad_token_id

        stop_token_ids = []
        eos_id = getattr(self.backbone.config, "eos_token_id", None)
        if isinstance(eos_id, int):
            stop_token_ids.append(eos_id)
        elif isinstance(eos_id, list):
            stop_token_ids.extend(eos_id)

        if tokenizer is not None:
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop_token_ids:
                stop_token_ids.append(tokenizer.eos_token_id)
            im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if (
                isinstance(im_end_id, int)
                and im_end_id != tokenizer.unk_token_id
                and im_end_id not in stop_token_ids
            ):
                stop_token_ids.append(im_end_id)

        generated_ids = cast(Any, self.backbone).generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=stop_token_ids if stop_token_ids else None,
            **generate_kwargs,
        )

        assistant_token_ids = generated_ids[:, prompt_len:]
        result: dict[str, Any] = {
            "generated_token_ids": assistant_token_ids,
        }

        if tokenizer is not None:
            raw_outputs = tokenizer.batch_decode(assistant_token_ids, skip_special_tokens=True)
            result["raw_outputs"] = [r.strip() for r in raw_outputs]
            result["predictions"] = [self.parse_json_output(raw) for raw in raw_outputs]

        if was_training:
            self.train()

        return result
