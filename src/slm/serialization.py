import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from src.slm.base_slm import SLMBase
from src.slm.cls_slm import SLMCls
from src.slm.json_slm import SLMJson
from src.slm.slm_factory import ModelFactory
from src.utils.model_card import clean_slug, generate_model_card_markdown

logger = logging.getLogger(__name__)


def save_finetuned_model(
    model: SLMBase,
    output_dir: str | Path,
    base_model_name: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
    training_params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    as_of: str | None = None,
) -> Path:
    """
    Saves a fine-tuned `SLMCls` or `SLMJson` model, its tokenizer, metadata configuration,
    and a Markdown Model Card with YAML frontmatter.

    Args:
        model: Trained `SLMCls` or `SLMJson` model.
        output_dir: Directory where the checkpoint will be saved.
        base_model_name: Hugging Face base model identifier.
        tokenizer: Associated tokenizer (defaults to `model.tokenizer`).
        training_params: Dictionary of training hyperparameters to document.
        metrics: Dictionary of evaluation metrics to document.
        as_of: Timestamp string (defaults to current UTC timestamp).

    Returns:
        Path to the saved checkpoint directory.
    """
    active_tokenizer = tokenizer or model.tokenizer
    if active_tokenizer is None:
        raise ValueError("Cannot save model checkpoint without a `tokenizer`.")

    save_path = Path(output_dir).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = as_of or datetime.now().strftime("%Y%m%d_%H%M%S")

    is_peft = isinstance(model.backbone, PeftModel)
    mode = model.mode

    # 1. Save backbone (LoRA adapters or full model)
    model.backbone.save_pretrained(save_path)

    # 2. Save tokenizer
    active_tokenizer.save_pretrained(save_path)

    # 3. For SLMCls, save normalization and classification head
    if isinstance(model, SLMCls):
        torch.save(
            {
                "norm": model.norm.state_dict(),
                "classifier": model.classifier.state_dict(),
            },
            save_path / "classifier.pt",
        )

    # 4. Save easySLM metadata
    max_length = model.max_length or (training_params or {}).get("max_length")
    easyslm_config: dict[str, Any] = {
        "mode": mode,
        "model_type": type(model).__name__,
        "base_model_name": base_model_name,
        "classes": model.classes,
        "num_classes": model.num_classes,
        "multi_label": model.multi_label,
        "is_peft": is_peft,
        "timestamp": timestamp,
    }
    if isinstance(model, SLMCls):
        easyslm_config["pooling_strategy"] = model.pooling_strategy
    if max_length is not None:
        easyslm_config["max_length"] = max_length

    config_file = save_path / "easyslm_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(easyslm_config, f, indent=2)

    # 5. Generate and save Model Card
    card_name = f"model_card_{clean_slug(base_model_name)}_{mode}_{timestamp}.md"
    card_content = generate_model_card_markdown(
        model_name=base_model_name,
        model_type=mode,
        as_of=timestamp,
        metrics=metrics or {},
        training_params=training_params or {},
        classes=model.classes,
        checkpoint_dir=save_path,
    )

    # Write inside the checkpoint directory
    (save_path / card_name).write_text(card_content, encoding="utf-8")

    # Also write into parent directory if save_path is inside an experiment folder
    parent_dir = save_path.parent
    if parent_dir != save_path and parent_dir.exists():
        (parent_dir / card_name).write_text(card_content, encoding="utf-8")

    logger.info("Saved fine-tuned %s model checkpoint to '%s'", mode.upper(), save_path)
    logger.info("Generated Model Card: '%s'", card_name)
    return save_path


def load_finetuned_model(
    checkpoint_dir: str | Path,
    prompts_config_path: str | Path = "configs/prompts.yml",
    factory: ModelFactory | None = None,
) -> SLMBase:
    """
    Loads a saved fine-tuned `SLMCls` or `SLMJson` model with its `tokenizer` and `metadata`
    attached directly to the returned `SLMBase` instance.

    Args:
        checkpoint_dir: Directory containing the saved model checkpoint.
        prompts_config_path: Path to prompts configuration YAML file.
        factory: Optional ModelFactory instance for caching and device management.

    Returns:
        Loaded `SLMBase` instance (`SLMCls` or `SLMJson`).
    """
    path = Path(checkpoint_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: '{checkpoint_dir}'")

    config_file = path / "easyslm_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Missing 'easyslm_config.json' inside checkpoint: '{path}'")

    with open(config_file, encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)

    cfg["checkpoint_dir"] = str(path)
    mode = cfg.get("mode", "cls").lower()
    base_model = cfg.get("base_model_name", "Qwen/Qwen2.5-0.5B")
    classes = cfg.get("classes", [])
    multi_label = cfg.get("multi_label", True)
    is_peft = cfg.get("is_peft", True)
    max_length = cfg.get("max_length", None)

    factory = factory or ModelFactory()

    # Load tokenizer: prefer checkpoint dir if config exists, fallback to base model
    tokenizer: PreTrainedTokenizerBase
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
    except Exception:
        tokenizer = factory._prepare_tokenizer(factory.download_model(base_model))

    backbone: Any
    if mode == "cls":
        # Load backbone: LoRA adapters on top of base model, or full fine-tuned model from checkpoint dir
        if is_peft and (path / "adapter_config.json").exists():
            base_backbone, _ = factory.get_backbone(base_model)
            backbone = PeftModel.from_pretrained(base_backbone, path)
            logger.info("Loaded fine-tuned LoRA adapters from '%s'", path)
        else:
            try:
                backbone = AutoModel.from_pretrained(path, dtype=factory.torch_dtype)
                logger.info("Loaded full fine-tuned backbone weights from '%s'", path)
            except Exception:
                backbone, _ = factory.get_backbone(base_model)
                logger.warning(
                    "Could not load weights from '%s'; loaded base model '%s'", path, base_model
                )

        pooling_strategy = cfg.get("pooling_strategy", "eos")
        cls_model = SLMCls(
            backbone=backbone,
            classes=classes,
            multi_label=multi_label,
            pooling_strategy=pooling_strategy,
            tokenizer=tokenizer,
            prompts_config_path=prompts_config_path,
            max_length=max_length,
            metadata=cfg,
        )

        classifier_file = path / "classifier.pt"
        if classifier_file.exists():
            classifier_state = torch.load(classifier_file, map_location=factory.device)
            if isinstance(classifier_state, dict) and "classifier" in classifier_state:
                if "norm" in classifier_state:
                    cls_model.norm.load_state_dict(classifier_state["norm"])
                cls_model.classifier.load_state_dict(classifier_state["classifier"])
            else:
                cls_model.classifier.load_state_dict(classifier_state)
            logger.info("Loaded trained classification head from '%s'.", classifier_file.name)
        else:
            logger.warning("No classifier.pt found in '%s'; using initialized head weights.", path)

        cls_model.to(factory.device)
        cls_model.eval()
        return cls_model

    elif mode == "json":
        # Load backbone: LoRA adapters on top of base model, or full fine-tuned model from checkpoint dir
        if is_peft and (path / "adapter_config.json").exists():
            base_causal, _ = factory.get_causal_lm(base_model)
            backbone = PeftModel.from_pretrained(base_causal, path)
            logger.info("Loaded fine-tuned LoRA adapters from '%s'", path)
        else:
            try:
                backbone = AutoModelForCausalLM.from_pretrained(path, dtype=factory.torch_dtype)
                logger.info("Loaded full fine-tuned CausalLM weights from '%s'", path)
            except Exception:
                backbone, _ = factory.get_causal_lm(base_model)
                logger.warning(
                    "Could not load weights from '%s'; loaded base model '%s'", path, base_model
                )

        json_model = SLMJson(
            backbone=backbone,
            classes=classes,
            multi_label=multi_label,
            tokenizer=tokenizer,
            prompts_config_path=prompts_config_path,
            max_length=max_length,
            metadata=cfg,
        )
        json_model.to(factory.device)
        json_model.eval()
        return json_model

    else:
        raise ValueError(f"Unknown model mode in easyslm_config.json: '{mode}'")
