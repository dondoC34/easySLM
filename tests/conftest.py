"""Shared pytest fixtures and lightweight CPU-only stubs for easySLM tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn


class FakeTokenizer:
    """Lightweight deterministic tokenizer stub avoiding HuggingFace Hub downloads."""

    def __init__(
        self,
        pad_token: str | None = "<pad>",
        eos_token: str | None = "<eos>",
        pad_token_id: int | None = 0,
        eos_token_id: int | None = 2,
    ) -> None:
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.padding_side = "right"
        self.saved_dirs: list[Path] = []
        self._vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self._id_to_token: dict[int, str] = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self._next_id = 3

    def add_special_tokens(self, special_tokens_dict: dict[str, str]) -> int:
        added = 0
        for key, val in special_tokens_dict.items():
            if key == "pad_token":
                self.pad_token = val
                if val not in self._vocab:
                    self._vocab[val] = self._next_id
                    self._id_to_token[self._next_id] = val
                    self.pad_token_id = self._next_id
                    self._next_id += 1
                else:
                    self.pad_token_id = self._vocab[val]
                added += 1
        return added

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = text.split()
        ids: list[int] = []
        for tok in tokens:
            if tok not in self._vocab:
                self._vocab[tok] = self._next_id
                self._id_to_token[self._next_id] = tok
                self._next_id += 1
            ids.append(self._vocab[tok])
        if not ids:
            ids = [1]
        return ids

    def decode(self, token_ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        special = {self.pad_token_id, self.eos_token_id} if skip_special_tokens else set()
        parts = [
            self._id_to_token.get(int(tid), "<unk>") for tid in token_ids if int(tid) not in special
        ]
        return " ".join(parts)

    def batch_decode(
        self,
        sequences: list[list[int]] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        self.saved_dirs.append(path)


class TinyBackbone(nn.Module):
    """Minimal Transformer-like encoder/decoder backbone returning `last_hidden_state`."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            pad_token_id=0,
            eos_token_id=2,
            use_cache=True,
        )
        self.gc_enabled = False
        self.input_require_grads = False
        self.saved_dirs: list[Path] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> SimpleNamespace:
        safe_ids = input_ids.clamp(min=0, max=self.embed.num_embeddings - 1)
        hidden = self.proj(self.embed(safe_ids))
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden,) if output_hidden_states else None,
        )

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        self.gc_enabled = True

    def gradient_checkpointing_disable(self) -> None:
        self.gc_enabled = False

    def enable_input_require_grads(self) -> None:
        self.input_require_grads = True

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        self.saved_dirs.append(path)


class TinyCausalLM(nn.Module):
    """Minimal CausalLM backbone returning `logits` and supporting `generate()`."""

    def __init__(self, vocab_size: int = 512, hidden_size: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            pad_token_id=0,
            eos_token_id=2,
            use_cache=True,
        )
        self.gc_enabled = False
        self.saved_dirs: list[Path] = []
        self._generated_suffix: list[int] = [3, 4]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        safe_ids = input_ids.clamp(min=0, max=self.embed.num_embeddings - 1)
        hidden = self.embed(safe_ids)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 8,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        suffix = torch.tensor(
            self._generated_suffix[:max_new_tokens],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).unsqueeze(0)
        suffix = suffix.expand(batch_size, -1)
        return torch.cat([input_ids, suffix], dim=1)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        self.gc_enabled = True

    def gradient_checkpointing_disable(self) -> None:
        self.gc_enabled = False

    def enable_input_require_grads(self) -> None:
        pass

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        self.saved_dirs.append(path)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def tiny_backbone() -> TinyBackbone:
    return TinyBackbone()


@pytest.fixture
def tiny_causal_lm() -> TinyCausalLM:
    return TinyCausalLM()


@pytest.fixture
def single_label_file(tmp_path: Path) -> Path:
    p = tmp_path / "single_label.txt"
    p.write_text(
        "\n".join(
            [
                "Stock markets rallied today ||| economy",
                "Football team wins championship ||| sports",
                "New AI chip announced ||| tech",
                "Central bank raises interest rates ||| economy",
                "Olympic swimmer breaks record ||| sports",
                "Quantum computer milestone reached ||| tech",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def multi_label_file(tmp_path: Path) -> Path:
    p = tmp_path / "multi_label.txt"
    p.write_text(
        "\n".join(
            [
                "Tech giant acquires startup ||| economy, tech",
                "Parliament debates tech regulation ||| politics, tech",
                "Election boosts stock index ||| economy, politics",
                "Sports league signs streaming deal ||| sports, tech",
                "Government funds stadium ||| politics, sports",
                "Market reacts to election ||| economy, politics",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p
