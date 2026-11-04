import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Fallback default ChatML template in case config YAML is missing
DEFAULT_FALLBACK_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # src/utils/config.py -> parent of src/ is project root
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Loads a YAML file and returns its dictionary content."""
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_chat_template(
    config_path: str | Path = "configs/chat_template.yml",
    template_name: str | None = None,
) -> str:
    """
    Loads a Jinja2 chat template from the specified YAML configuration file.

    Args:
        config_path: Path to the chat template YAML file (relative to project root or absolute).
        template_name: Name of the template key under 'templates'. If None, uses 'default_template_name'.

    Returns:
        The Jinja2 template string.
    """
    try:
        data = load_yaml(config_path)
        selected_name = template_name or data.get("default_template_name", "chatml")
        templates = data.get("templates", {})
        template = templates.get(selected_name)
        if template:
            return template.strip()
        logger.warning(
            "Template '%s' not found in %s. Using default fallback template.",
            selected_name,
            config_path,
        )
    except Exception as e:
        logger.warning(
            "Could not load chat template from '%s' (%s). Using default fallback template.",
            config_path,
            e,
        )

    return DEFAULT_FALLBACK_CHATML_TEMPLATE


DEFAULT_PROMPTS_CONFIG = {
    "mode_descriptions": {
        "multi_label": "one or more applicable classes (multi-label)",
        "single_label": "the single most appropriate class (single-label)",
    },
    "cls": {
        "system_prompt": (
            "You are an expert text classification assistant. "
            "Classify the user text into {mode_desc} from the following valid classes: [{classes}]."
        ),
        "user_prompt": "{text}",
    },
    "json": {
        "system_prompt": (
            "You are a text classification model. "
            "Classify the user text into {mode_desc} from the following valid classes: [{classes}]. "
            "Respond ONLY with a valid list of predicted class names sorted in alphabetical order, "
            "for example: {example}. Do not output any additional text or explanation."
        ),
        "user_prompt": "{text}",
    },
}


def load_prompts_config(
    config_path: str | Path = "configs/prompts.yml",
) -> dict[str, Any]:
    """
    Loads prompt templates from the specified YAML configuration file.

    Args:
        config_path: Path to the prompts YAML file (relative to project root or absolute).

    Returns:
        Dictionary containing prompt configurations and mode descriptions.
    """
    try:
        data = load_yaml(config_path)
        if isinstance(data, dict) and data:
            return data
    except Exception as e:
        logger.warning(
            "Could not load prompts config from '%s' (%s). Using built-in default prompts.",
            config_path,
            e,
        )

    return DEFAULT_PROMPTS_CONFIG
