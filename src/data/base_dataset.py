from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.collator import SLMCollator
from src.utils.config import load_chat_template, load_prompts_config


class SLMDatasetBase(Dataset, ABC):
    """
    Abstract base Dataset for Small Language Model classification tasks.

    Responsibilities:
    1. Resolves `data_path`:
       - If a `.txt` file is provided, loads only that file.
       - If a directory is provided, loads all `.txt` files inside that directory.
    2. Parses each line using the `'|||'` separator into `(text, comma_separated_labels)`,
       then splits individual labels on `','` and sorts them alphabetically.
    3. Discovers (or validates) the global sorted `classes` vocabulary and builds `class_to_idx` / `idx_to_class`.
    4. Builds structured chat messages (`system`, `user`, and optional `assistant`) and applies
       the tokenizer's chat template (`apply_chat_template`).
    5. Tokenizes all samples into `input_ids`, `attention_mask`, and task-specific `labels`.
    """

    SEPARATOR = "|||"

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        classes: Sequence[str] | None = None,
        multi_label: bool = True,
        system_prompt: str | None = None,
        max_length: int | None = None,
        include_targets: bool = True,
        chat_template_config_path: str | Path = "configs/chat_template.yml",
        chat_template_name: str | None = None,
        prompts_config_path: str | Path = "configs/prompts.yml",
        user_prompt_template: str | None = None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.multi_label = multi_label
        self.max_length = max_length
        self.include_targets = include_targets
        self.chat_template_config_path = chat_template_config_path
        self.chat_template_name = chat_template_name
        self.prompts_config_path = prompts_config_path

        self._ensure_tokenizer_ready()

        # Load prompts configuration
        self.prompts_config: dict[str, Any] = load_prompts_config(prompts_config_path)
        if user_prompt_template is not None:
            self.user_prompt_template = user_prompt_template
        else:
            mode_key = getattr(self, "TASK_MODE", "cls")
            self.user_prompt_template = self.prompts_config.get(mode_key, {}).get(
                "user_prompt", "{text}"
            )

        # 1. Load and parse raw datapoints: List[(text, sorted_label_list)]
        self.raw_samples: list[tuple[str, list[str]]] = self._load_and_parse_files(self.data_path)
        if not self.raw_samples:
            raise ValueError(f"No valid datapoints found in '{self.data_path}'.")

        # 2. Build sorted class vocabulary
        if classes is None:
            discovered = {lbl for _, lbls in self.raw_samples for lbl in lbls}
            if not discovered:
                raise ValueError(
                    "Could not discover any classes from dataset; please pass `classes` explicitly."
                )
            self.classes: list[str] = sorted(discovered)
        else:
            self.classes = sorted(list(dict.fromkeys(c.strip() for c in classes if c.strip())))

        self.num_classes: int = len(self.classes)
        self.class_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class: dict[int, str] = {i: c for i, c in enumerate(self.classes)}

        # Validate that all sample labels exist in self.classes
        for line_idx, (_, sample_labels) in enumerate(self.raw_samples):
            for lbl in sample_labels:
                if lbl not in self.class_to_idx:
                    raise ValueError(
                        f"Sample {line_idx} contains unknown label '{lbl}'. Valid classes: {self.classes}"
                    )
            if not self.multi_label and self.include_targets and len(sample_labels) != 1:
                raise ValueError(
                    f"Single-label dataset expects 1 label per row, but sample {line_idx} has {sample_labels}."
                )

        # 3. Configure system prompt
        self.system_prompt: str = system_prompt or self._build_default_system_prompt()

        # 4. Tokenize all datapoints
        self.encoded_samples: list[dict[str, torch.Tensor]] = [
            self._encode_sample(text, labels) for text, labels in self.raw_samples
        ]

    def _ensure_tokenizer_ready(self) -> None:
        """Ensures the tokenizer has both a `pad_token` and a `chat_template`."""
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = load_chat_template(
                config_path=self.chat_template_config_path,
                template_name=self.chat_template_name,
            )

    def _resolve_txt_files(self, path: Path) -> list[Path]:
        """Returns the single `.txt` file or all `.txt` files inside a folder."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: '{path}'")

        if path.is_file():
            if path.suffix.lower() != ".txt":
                raise ValueError(f"Expected a '.txt' file, got '{path}'.")
            return [path]

        if path.is_dir():
            txt_files = sorted(path.glob("*.txt"))
            if not txt_files:
                raise ValueError(f"No '.txt' files found inside directory '{path}'.")
            return txt_files

        raise ValueError(f"Unsupported dataset path: '{path}'")

    def _load_and_parse_files(self, path: Path) -> list[tuple[str, list[str]]]:
        """
        Reads all lines from resolved `.txt` file(s), splits each line on `'|||'` into
        `(text, labels_str)`, and splits `labels_str` on `','` into alphabetically sorted labels.
        """
        txt_files = self._resolve_txt_files(path)
        parsed_rows: list[tuple[str, list[str]]] = []

        for file_path in txt_files:
            with open(file_path, encoding="utf-8") as f:
                for line_num, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    if self.SEPARATOR in line:
                        text_part, labels_part = line.split(self.SEPARATOR, maxsplit=1)
                        text = text_part.strip()
                        labels = sorted(
                            list(
                                dict.fromkeys(
                                    lbl.strip() for lbl in labels_part.split(",") if lbl.strip()
                                )
                            )
                        )
                    else:
                        if self.include_targets:
                            raise ValueError(
                                f"Malformed line in '{file_path}' (line {line_num}): "
                                f"missing separator '{self.SEPARATOR}' in: {line!r}"
                            )
                        text = line.strip()
                        labels = []

                    if not text:
                        raise ValueError(f"Empty text in '{file_path}' at line {line_num}.")

                    if self.include_targets and not labels:
                        raise ValueError(f"No labels found in '{file_path}' at line {line_num}.")

                    if not self.multi_label and self.include_targets and len(labels) != 1:
                        raise ValueError(
                            f"Single-label classification error in '{file_path}' at line {line_num}: "
                            f"Expected exactly 1 label per sample, but found {len(labels)}: {labels}. "
                            f"Sample text: '{text}'. "
                            f"To use multiple labels per sample, run in multi-label mode (multi_label=True)."
                        )

                    parsed_rows.append((text, labels))

        return parsed_rows

    @abstractmethod
    def _build_default_system_prompt(self) -> str:
        """Builds the default system prompt describing the task and classes."""
        pass

    def format_user_prompt(self, user_text: str) -> str:
        """Formats the raw user text using `user_prompt_template`."""
        if hasattr(self, "user_prompt_template") and "{text}" in self.user_prompt_template:
            return self.user_prompt_template.format(text=user_text)
        return user_text

    def build_chat_messages(
        self,
        user_text: str,
        assistant_text: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Creates the chat structure:
        - `system`: task instructions and valid classes.
        - `user`: formatted text to classify.
        - `assistant`: target response string (only when `assistant_text` is not None).
        """
        formatted_user = self.format_user_prompt(user_text)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_user},
        ]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def render_chat_ids(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Applies the tokenizer's chat template and returns a list of token IDs."""
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered, list):
            if hasattr(rendered, "input_ids"):
                return list(rendered.input_ids)
            if isinstance(rendered, dict) and "input_ids" in rendered:
                return list(rendered["input_ids"])
        return [int(x) for x in rendered]  # type: ignore[union-attr, arg-type]

    @abstractmethod
    def _encode_sample(
        self,
        text: str,
        labels: list[str],
    ) -> dict[str, torch.Tensor]:
        """Tokenizes a single `(text, labels)` pair into model-ready tensors."""
        pass

    @abstractmethod
    def get_collator(
        self,
        padding_side: Literal["right", "left"] | None = None,
        device: str | torch.device | None = None,
    ) -> SLMCollator:
        """Returns a configured `SLMCollator` instance for batching variable-length sequences."""
        pass

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encoded_samples[idx]
