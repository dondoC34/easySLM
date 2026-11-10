from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.data.collator import SLMCollator
from src.utils.config import load_prompts_config


class SLMDatasetBase(Dataset, ABC):
    """
    Abstract base Dataset for Small Language Model classification tasks.

    Responsibilities:
    1. Resolves `data_path` (or accepts in-memory `texts` directly):
       - If a `.txt` file is provided, loads only that file.
       - If a directory is provided, loads all `.txt` files inside that directory.
    2. Parses each line using the `'|||'` separator into `(text, comma_separated_labels)`,
       then splits individual labels on `','` and sorts them alphabetically.
    3. Discovers (or validates) the global sorted `classes` vocabulary and builds `class_to_idx` / `idx_to_class`.
    4. Builds plain-text prompts from `configs/prompts.yml` and appends the model's native `eos_token`.
    5. Tokenizes all samples into `input_ids`, `attention_mask`, and task-specific `labels`.
    """

    SEPARATOR = "|||"
    DEFAULT_PROMPT_TEMPLATE = "{system_prompt}\n\nText: {user_prompt}"

    def __init__(
        self,
        data_path: str | Path | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        classes: Sequence[str] | None = None,
        multi_label: bool = True,
        system_prompt: str | None = None,
        max_length: int | None = None,
        include_targets: bool = True,
        prompts_config_path: str | Path = "configs/prompts.yml",
        user_prompt_template: str | None = None,
        prompt_template: str | None = None,
        texts: Sequence[str] | None = None,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("`tokenizer` must be provided to initialize SLMDatasetBase.")
        if data_path is None and texts is None:
            raise ValueError("Either `data_path` or `texts` must be provided.")

        self.data_path = Path(data_path) if data_path is not None else None
        self.tokenizer = tokenizer
        self.multi_label = multi_label
        self.max_length = max_length
        self.include_targets = include_targets
        self.prompts_config_path = prompts_config_path

        self._ensure_tokenizer_ready()

        # Load prompts configuration
        self.prompts_config: dict[str, Any] = load_prompts_config(prompts_config_path)
        mode_key = getattr(self, "TASK_MODE", "cls")
        mode_cfg: dict[str, Any] = self.prompts_config.get(mode_key, {})

        if user_prompt_template is not None:
            self.user_prompt_template = user_prompt_template
        else:
            self.user_prompt_template = mode_cfg.get("user_prompt", "{text}")

        if prompt_template is not None:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = mode_cfg.get("prompt_template", self.DEFAULT_PROMPT_TEMPLATE)

        # 1. Load and parse raw datapoints: List[(text, sorted_label_list)]
        if texts is not None:
            self.raw_samples: list[tuple[str, list[str]]] = [(t, []) for t in texts]
        else:
            assert self.data_path is not None
            self.raw_samples = self._load_and_parse_files(self.data_path)
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

        # Validate that all sample labels exist in self.classes when targets are included
        if self.include_targets:
            for line_idx, (_, sample_labels) in enumerate(self.raw_samples):
                for lbl in sample_labels:
                    if lbl not in self.class_to_idx:
                        raise ValueError(
                            f"Sample {line_idx} contains unknown label '{lbl}'. Valid classes: {self.classes}"
                        )
                if not self.multi_label and len(sample_labels) != 1:
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
        """Ensures the tokenizer has a valid `pad_token`."""
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    @staticmethod
    def resolve_txt_files(path: str | Path) -> list[Path]:
        """Returns the single `.txt` file or all `.txt` files inside a folder."""
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset path does not exist: '{resolved}'")

        if resolved.is_file():
            if resolved.suffix.lower() != ".txt":
                raise ValueError(f"Expected a '.txt' file, got '{resolved}'.")
            return [resolved]

        if resolved.is_dir():
            txt_files = sorted(resolved.glob("*.txt"))
            if not txt_files:
                raise ValueError(f"No '.txt' files found inside directory '{resolved}'.")
            return txt_files

        raise ValueError(f"Unsupported dataset path: '{resolved}'")

    def _resolve_txt_files(self, path: Path) -> list[Path]:
        """Returns the single `.txt` file or all `.txt` files inside a folder."""
        return self.resolve_txt_files(path)

    @classmethod
    def parse_raw_line(cls, raw_line: str) -> tuple[str, list[str] | None] | None:
        """
        Parses a single raw line from a `.txt` file.
        Returns `None` if the line is empty; otherwise returns `(text, labels_or_none)` where
        `labels_or_none` is a list of label strings if `'|||'` is present, or `None` if omitted.
        """
        line = raw_line.strip()
        if not line:
            return None
        if cls.SEPARATOR in line:
            text_part, labels_part = line.split(cls.SEPARATOR, maxsplit=1)
            text = text_part.strip()
            labels = [lbl.strip() for lbl in labels_part.split(",") if lbl.strip()]
            return text, labels
        return line, None

    def _load_and_parse_files(self, path: Path) -> list[tuple[str, list[str]]]:
        """
        Reads all lines from resolved `.txt` file(s), splits each line on `'|||'` into
        `(text, labels_str)`, and splits `labels_str` on `','` into alphabetically sorted labels.
        """
        txt_files = self.resolve_txt_files(path)
        parsed_rows: list[tuple[str, list[str]]] = []

        for file_path in txt_files:
            with open(file_path, encoding="utf-8") as f:
                for line_num, raw_line in enumerate(f, start=1):
                    parsed = self.parse_raw_line(raw_line)
                    if parsed is None:
                        continue
                    text, raw_labels = parsed

                    if raw_labels is not None:
                        labels = sorted(list(dict.fromkeys(raw_labels)))
                    else:
                        if self.include_targets:
                            raise ValueError(
                                f"Malformed line in '{file_path}' (line {line_num}): "
                                f"missing separator '{self.SEPARATOR}' in: {raw_line.strip()!r}"
                            )
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

    def build_prompt(self, user_text: str) -> str:
        """Formats the complete plain-text prompt combining `system_prompt` and `user_prompt`."""
        formatted_user = self.format_user_prompt(user_text)
        return self.prompt_template.format(
            system_prompt=self.system_prompt,
            user_prompt=formatted_user,
        )

    def encode_text_ids(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Tokenizes a text string into a list of integer token IDs."""
        return [int(x) for x in self.tokenizer.encode(text, add_special_tokens=add_special_tokens)]

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
