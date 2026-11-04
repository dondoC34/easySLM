import logging
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.utils.device import get_default_device, get_default_dtype

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory helper to manage, download, and instantiate Hugging Face Small Language Models (SLMs).

    If the model exists in the local directory, it loads directly from disk.
    Otherwise, it downloads the model snapshot to the local directory first,
    allowing efficient offline reuse and fine-tuning.
    """

    def __init__(
        self,
        cache_dir: str | Path = "./models",
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        """
        Args:
            cache_dir: Directory where models are stored locally.
            device: Target device ('cuda', 'cpu', or torch.device). Defaults to 'cuda' if available.
            torch_dtype: Model weights dtype (e.g., torch.float16, torch.bfloat16). Defaults to optimal GPU dtype.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = get_default_device()
        else:
            self.device = torch.device(device)

        if torch_dtype is None:
            self.torch_dtype = get_default_dtype(self.device)
        else:
            self.torch_dtype = torch_dtype

    def get_local_path(self, model_name: str) -> Path:
        """Converts a model repo ID (e.g. 'Qwen/Qwen2.5-0.5B') to a local folder path."""
        direct_path = Path(model_name)
        if direct_path.is_dir() and (direct_path / "config.json").exists():
            return direct_path

        sanitized_name = model_name.replace("/", "--")
        return self.cache_dir / sanitized_name

    def is_cached_locally(self, model_name: str) -> bool:
        """Checks whether the model files already exist locally."""
        local_path = self.get_local_path(model_name)
        return local_path.is_dir() and (local_path / "config.json").exists()

    def download_model(self, model_name: str, force_download: bool = False) -> Path:
        """
        Downloads the full model repository to the local storage path if not already present.

        Args:
            model_name: Hugging Face hub model ID (e.g. 'Qwen/Qwen2.5-0.5B').
            force_download: Re-download even if files already exist.

        Returns:
            Path to the local model folder.
        """
        local_path = self.get_local_path(model_name)

        if self.is_cached_locally(model_name) and not force_download:
            logger.info("Model '%s' already cached locally at %s", model_name, local_path)
            return local_path

        logger.info("Downloading '%s' to local directory: %s ...", model_name, local_path)
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_path),
            force_download=force_download,
        )
        logger.info("Successfully downloaded '%s' to %s", model_name, local_path)
        return local_path

    def _prepare_tokenizer(
        self,
        local_path: Path,
        padding_side: str = "right",
    ) -> PreTrainedTokenizerBase:
        """Loads and configures the tokenizer, ensuring pad token is properly defined."""
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_path),
            padding_side=padding_side,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def get_backbone(
        self,
        model_name: str,
        padding_side: str = "right",
        device_map: str | dict | None = "auto",
        **model_kwargs,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads the base transformer backbone (AutoModel, without task head) and its tokenizer.
        Used by SLMCls to attach custom pooling and classification layers on top of hidden states.
        """
        local_path = self.download_model(model_name)
        tokenizer = self._prepare_tokenizer(local_path, padding_side=padding_side)

        resolved_device_map = device_map if self.device.type == "cuda" else None

        logger.info(
            "Loading AutoModel backbone from %s (dtype=%s)...", local_path, self.torch_dtype
        )
        backbone = AutoModel.from_pretrained(
            str(local_path),
            dtype=self.torch_dtype,
            device_map=resolved_device_map,
            trust_remote_code=True,
            **model_kwargs,
        )

        if resolved_device_map is None:
            backbone = backbone.to(self.device)

        if tokenizer.pad_token_id is not None and backbone.config.pad_token_id is None:
            backbone.config.pad_token_id = tokenizer.pad_token_id

        return backbone, tokenizer

    def get_causal_lm(
        self,
        model_name: str,
        padding_side: str = "right",
        device_map: str | dict | None = "auto",
        **model_kwargs,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads an AutoModelForCausalLM model (backbone + vocab linear head) and its tokenizer.
        Used by SLMJson for generative JSON list classification.
        """
        local_path = self.download_model(model_name)
        tokenizer = self._prepare_tokenizer(local_path, padding_side=padding_side)

        resolved_device_map = device_map if self.device.type == "cuda" else None

        logger.info(
            "Loading AutoModelForCausalLM from %s (dtype=%s)...", local_path, self.torch_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(local_path),
            dtype=self.torch_dtype,
            device_map=resolved_device_map,
            trust_remote_code=True,
            **model_kwargs,
        )

        if resolved_device_map is None:
            model = model.to(self.device)  # type: ignore[arg-type]

        if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def get_sequence_classifier(
        self,
        model_name: str,
        num_labels: int = 2,
        id2label: dict[int, str] | None = None,
        label2id: dict[str, int] | None = None,
        problem_type: str | None = None,
        device_map: str | dict | None = "auto",
        **model_kwargs,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Loads an AutoModelForSequenceClassification model and its tokenizer.
        """
        local_path = self.download_model(model_name)
        tokenizer = self._prepare_tokenizer(local_path, padding_side="right")

        resolved_device_map = device_map if self.device.type == "cuda" else None

        config_kwargs: dict[str, Any] = {"num_labels": num_labels}
        if id2label is not None:
            config_kwargs["id2label"] = id2label
        if label2id is not None:
            config_kwargs["label2id"] = label2id
        if problem_type is not None:
            config_kwargs["problem_type"] = problem_type

        model = AutoModelForSequenceClassification.from_pretrained(
            str(local_path),
            dtype=self.torch_dtype,
            device_map=resolved_device_map,
            trust_remote_code=True,
            **config_kwargs,
            **model_kwargs,
        )

        if resolved_device_map is None:
            model = model.to(self.device)  # type: ignore[arg-type]

        if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer
