import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

EASYSLM_HOME_ENV_VAR = "EASYSLM_HOME"


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # src/utils/config.py -> parent of src/ is project root
    return Path(__file__).resolve().parent.parent.parent


def get_easyslm_home() -> Path:
    """Returns the base directory for easySLM caches, checkpoints, and tracking data.

    Defaults to ``~/.easyslm`` and can be overridden via the ``EASYSLM_HOME``
    environment variable.
    """
    env_home = os.environ.get(EASYSLM_HOME_ENV_VAR)
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".easyslm").resolve()


def get_default_models_cache_dir() -> Path:
    """Returns the default base-model cache directory (``<EASYSLM_HOME>/models``)."""
    return get_easyslm_home() / "models"


def get_default_finetuned_dir() -> Path:
    """Returns the default directory for fine-tuned checkpoints (``<EASYSLM_HOME>/finetuned``)."""
    return get_easyslm_home() / "finetuned"


def get_default_predictions_dir() -> Path:
    """Returns the default directory for prediction outputs (``<EASYSLM_HOME>/predictions``)."""
    return get_easyslm_home() / "predictions"


def get_default_mlflow_db_path() -> Path:
    """Returns the default SQLite database path for MLflow (``<EASYSLM_HOME>/mlflow.db``)."""
    return get_easyslm_home() / "mlflow.db"


def get_default_mlflow_tracking_uri() -> str:
    """Returns the default SQLite MLflow tracking URI under ``<EASYSLM_HOME>``."""
    return f"sqlite:///{get_default_mlflow_db_path().as_posix()}"


def get_default_mlflow_artifact_root() -> Path:
    """Returns the default MLflow artifact root directory (``<EASYSLM_HOME>/mlruns``)."""
    return get_easyslm_home() / "mlruns"


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Loads a YAML file and returns its dictionary content."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


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
        "prompt_template": "{system_prompt}\n\nText: {user_prompt}\nLabel:",
    },
    "json": {
        "system_prompt": (
            "You are a text classification model. "
            "Classify the user text into {mode_desc} from the following valid classes: [{classes}]. "
            "Respond ONLY with a valid list of predicted class names sorted in alphabetical order, "
            "for example: {example}. Do not output any additional text or explanation."
        ),
        "user_prompt": "{text}",
        "prompt_template": "{system_prompt}\n\nText: {user_prompt}\nLabels:\n",
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
