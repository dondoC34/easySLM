# Utilities & Model Cards

Path/home resolution (`EASYSLM_HOME`), YAML & prompt loaders, CUDA/device helpers, and Markdown Model Card generation & discovery.

---

## Home Paths & Config Loaders

::: src.utils.config.get_easyslm_home

::: src.utils.config.get_default_models_cache_dir

::: src.utils.config.get_default_finetuned_dir

::: src.utils.config.get_default_predictions_dir

::: src.utils.config.get_default_mlflow_db_path

::: src.utils.config.get_default_mlflow_tracking_uri

::: src.utils.config.get_default_mlflow_artifact_root

::: src.utils.config.load_yaml

::: src.utils.config.load_prompts_config

---

## Device & Precision Helpers

::: src.utils.device.get_default_device

::: src.utils.device.get_default_dtype

::: src.utils.device.is_native_bf16_supported

::: src.utils.device.print_cuda_info

---

## Model Cards

::: src.utils.model_card.generate_model_card_markdown

::: src.utils.model_card.parse_model_card

::: src.utils.model_card.find_best_model
