# Customized Fine-Tuning

This tutorial shows how to customize every stage of the pipeline: `ModelFactory`, classification architecture (`cls` vs `json`), LoRA & VRAM tuning, class-imbalance oversampling, early stopping, and custom prompt templates.

!!! tip "Interactive Notebook"
    A runnable version exposing all fields is available in [`notebooks/customize_fine_tuning.ipynb`](https://github.com/dondoC34/easySLM/blob/main/notebooks/customize_fine_tuning.ipynb).

---

## 1. Custom `ModelFactory` (Cache, Device, Precision)

Use `ModelFactory` when you want to override the local cache folder, pin a specific device, or force a precision dtype:

```python
import torch
from src import ModelFactory

factory = ModelFactory(
    cache_dir="/data/hf_models",  # defaults to ~/.easyslm/models
    device="cuda",  # auto-detected if None
    torch_dtype=torch.bfloat16,  # auto-selects bf16/fp16/fp32 if None
)
```

---

## 2. Direct Classification (`cls`) vs. Generative Mode (`json`)

=== "CLS Mode (Linear Head + Pooling)"

    Fast, calibrated probabilities (`Softmax` for single-label, `Sigmoid` for multi-label). Choose from `"eos"`, `"mean"`, or `"max"` sequence pooling:

    ```python
    from src import FineTuningConfig, SLMFineTuner

    config = FineTuningConfig(
        mode="cls",
        pooling_strategy="mean",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        lora_r=16,
        lora_alpha=32,
        num_train_epochs=3.0,
        early_stopping_window=4,
        early_stopping_metric="eval_f1_macro",
    )

    tuner = SLMFineTuner(model_name="Qwen/Qwen2.5-0.5B", config=config, factory=factory)
    results = tuner.train(train_data_path="data/agnews_tiny")
    ```

=== "JSON Mode (Generative CausalLM)"

    Fine-tunes `AutoModelForCausalLM` to generate alphabetically sorted Python/JSON label lists (e.g., `['business', 'sci_tech']`). Prompt tokens are masked with `-100` so loss is computed only on target tokens:

    ```python
    from src import FineTuningConfig, SLMFineTuner

    config = FineTuningConfig(
        mode="json",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3.0,
        early_stopping_metric="eval_loss",
    )

    tuner = SLMFineTuner(model_name="Qwen/Qwen2.5-0.5B", config=config, factory=factory)
    results = tuner.train(train_data_path="data/agnews_tiny")
    ```

---

## 3. Handling Class Imbalance with Oversampling

When minority classes are under-represented in your training split, enable oversampling on the training dataset (validation is never oversampled):

```python
config = FineTuningConfig(
    mode="cls",
    oversample_strategy="sqrt",  # "sqrt" (moderate) or "max" (full balance)
    oversample_max_factor=3.0,  # cap replication at 3x per class
    seed=42,
)
```

---

## 4. Custom Prompt Templates

By default, easySLM uses the built-in `DEFAULT_PROMPTS_CONFIG` (see [Prompt Templates](../getting-started/configuration.md#2-prompt-templates-configspromptsyml) for the full default template and supported placeholders). To customize it, create a domain-specific YAML file (e.g., `custom_prompts.yml`):

```yaml
mode_descriptions:
  single_label: "the single most relevant news category"
  multi_label: "all applicable news categories"

cls:
  system_prompt: >-
    Classify the headline into {mode_desc}. Valid categories: [{classes}].
  user_prompt: "Headline: {text}"
  prompt_template: "{system_prompt}\n\n{user_prompt}"
```

Pass it via `prompts_config_path`:

```python
config = FineTuningConfig(prompts_config_path="custom_prompts.yml")
```

The exact prompt snapshot is saved as `prompt_config.yml` inside the checkpoint directory so inference automatically reproduces the exact training prompt format.

---

## 5. Loading from YAML or Dict with Strict Validation

```python
# Load from YAML and override selected fields
config = FineTuningConfig.from_yaml(
    "configs/training_config.yml",
    num_train_epochs=2.0,
    run_name="agnews-experiment-v2",
)

# Or from a plain dictionary
config = FineTuningConfig.from_dict({"mode": "cls", "lora_r": 16})
```

If a typo slips into a YAML file or dictionary, Pydantic immediately raises a `ValidationError` listing all accepted keys and their defaults.
