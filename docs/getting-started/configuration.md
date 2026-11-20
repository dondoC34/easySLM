# Configuration

## 1. `FineTuningConfig` (Pydantic v2)

`FineTuningConfig` is the **single source of truth** for all fine-tuning defaults and validation. It is a Pydantic v2 `BaseModel` configured with:

- `extra="forbid"`: unrecognized keys immediately raise a `ValidationError` listing **all accepted keys and their defaults**.
- `validate_assignment=True`: mutating fields after instantiation is validated as well.
- Numeric and literal bounds via `Field(...)` (e.g., `per_device_train_batch_size >= 1`, `val_split_ratio` in `[0.0, 1.0)`, `mode` in `{"cls", "json"}`).

### Instantiation Methods

=== "Built-in Defaults & Keyword Overrides"

    ```python
    from src import FineTuningConfig

    # All built-in defaults (no YAML file needed)
    default_cfg = FineTuningConfig()

    # Override only specific parameters
    cfg = FineTuningConfig(
        mode="cls",
        num_train_epochs=2.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        use_mlflow=False,
    )
    ```

=== "From Dictionary (`from_dict`)"

    ```python
    from src import FineTuningConfig

    cfg = FineTuningConfig.from_dict(
        {"mode": "json", "lora_r": 16},
        num_train_epochs=3.0,
    )
    ```

=== "From YAML (`from_yaml`)"

    ```python
    from src import FineTuningConfig

    # Optional YAML template with keyword overrides
    cfg = FineTuningConfig.from_yaml(
        "configs/training_config.yml",
        num_train_epochs=2.0,
        use_mlflow=False,
    )
    ```

!!! note "Passing `config=None` to `SLMFineTuner`"
    `SLMFineTuner(model_name="Qwen/Qwen2.5-0.5B", config=None)` directly instantiates `FineTuningConfig()`, making `easySLM` completely independent of any external YAML file when installed as a library.

### Parameter Reference

| Group | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **Output & Mode** | `output_dir` | `~/.easyslm/finetuned` | Directory where checkpoints & Model Cards are saved |
| | `mode` | `"cls"` | `"cls"` (linear head) or `"json"` (generative list) |
| | `pooling_strategy` | `"eos"` | Sequence pooling for `cls`: `"eos"`, `"mean"`, or `"max"` |
| | `prompts_config_path` | `"configs/prompts.yml"` | Prompt templates YAML (falls back to built-in defaults) |
| **Batching & VRAM** | `per_device_train_batch_size` | `1` | Training micro-batch size per device (`>= 1`) |
| | `per_device_eval_batch_size` | `4` | Validation batch size per device (`>= 1`) |
| | `gradient_accumulation_steps` | `8` | Accumulation steps before optimizer update (`>= 1`) |
| | `gradient_checkpointing` | `True` | Trades compute for lower activation VRAM |
| **LoRA / PEFT** | `use_lora` | `True` | Use LoRA adapters (`False` for full fine-tuning) |
| | `lora_r` / `lora_alpha` | `8` / `16` | Adapter rank and scaling factor |
| | `lora_dropout` | `0.05` | Dropout probability inside LoRA layers (`[0.0, 1.0]`) |
| | `lora_target_modules` | `"all-linear"` | Target linear modules (`str` or `list[str]`) |
| **Optimization** | `learning_rate` | `None` | Auto-selects `2e-4` (LoRA) or `2e-5` (full FT) |
| | `lr_scheduler_type` | `"cosine"` | Hugging Face LR scheduler |
| | `warmup_ratio` | `0.1` | Fraction of total steps for LR warmup |
| | `weight_decay` / `max_grad_norm` | `0.01` / `1.0` | Weight decay and gradient clipping norm |
| | `num_train_epochs` | `3.0` | Total training epochs (`> 0.0`) |
| **Eval & Stopping** | `eval_strategy` / `eval_steps` | `"steps"` / `10` | Validation schedule (`"steps"`, `"epoch"`, `"no"`) |
| | `save_strategy` / `save_total_limit` | `"no"` / `2` | Trainer intermediate checkpointing settings |
| | `logging_steps` | `5` | Training log frequency |
| | `val_split_ratio` | `0.15` | Auto-split ratio when given a single `.txt` file |
| | `oversample_strategy` | `None` | Minority-class oversampling: `None`, `"max"`, `"sqrt"` |
| | `oversample_max_factor` | `None` | Optional cap multiplier for oversampling |
| | `early_stopping_window` | `4` | Evaluations without improvement before stopping (`None` disables) |
| | `early_stopping_metric` | `"eval_loss"` | Monitored validation metric |
| | `seed` | `42` | Random seed |
| **MLflow** | `use_mlflow` | `True` | Log run metadata, metrics, plots, and tables |
| | `mlflow_tracking_uri` | `None` | Override `MLFLOW_TRACKING_URI` |
| | `mlflow_experiment_name` | `None` | Override MLflow experiment name |
| | `mlflow_config_path` | `None` | Custom `mlflow-config.yml` path |
| | `run_name` | `None` | Custom run name |
| | `max_errors_per_class` | `5` | Max false-positive samples per class in `validation_errors.json` |

---

## 2. Prompt Templates (`configs/prompts.yml`)

By default—when no custom prompt YAML is provided via `prompts_config_path` (or `--prompts-config`)—easySLM uses the built-in `DEFAULT_PROMPTS_CONFIG`:

```yaml
mode_descriptions:
  single_label: "the single most appropriate class (single-label)"
  multi_label: "one or more applicable classes (multi-label)"

cls:
  system_prompt: >-
    You are an expert text classification assistant.
    Classify the user text into {mode_desc} from the following valid classes: [{classes}].
  user_prompt: "{text}"
  prompt_template: "{system_prompt}\n\nText: {user_prompt}\nLabel:"

json:
  system_prompt: >-
    You are a text classification model.
    Classify the user text into {mode_desc} from the following valid classes: [{classes}].
    Respond ONLY with a valid list of predicted class names sorted in alphabetical order,
    for example: {example}. Do not output any additional text or explanation.
  user_prompt: "{text}"
  prompt_template: "{system_prompt}\n\nText: {user_prompt}\nLabels:\n"
```

Supported placeholders:

- `{classes}`: Alphabetically sorted, single-quoted class list (e.g., `'business', 'sci_tech', 'sports', 'world'`).
- `{mode_desc}`: Task description from `mode_descriptions` (`single_label` vs. `multi_label`).
- `{example}`: Formatted example output for `json` mode.
- `{text}`, `{system_prompt}`, `{user_prompt}`: Prompt composition placeholders.

For example, with `mode="cls"` on `agnews_tiny`, the rendered prompt passed to the model is:

```text
You are an expert text classification assistant. Classify the user text into the single most appropriate class (single-label) from the following valid classes: ['business', 'sci_tech', 'sports', 'world'].

Text: Central bank holds interest rates steady as quarterly corporate earnings beat forecasts.
Label:
```

---

## 3. MLflow Settings (`mlflow-config.yml`)

Leaving `backend_store_uri` and `default_artifact_root` set to `null` resolves them automatically under `~/.easyslm/` (`EASYSLM_HOME`):

```yaml
server:
  host: "127.0.0.1"
  port: 5000
  backend_store_uri: null
  default_artifact_root: null

tracking:
  experiment_prefix: "easySLM"
  autolog_figures: true
  autolog_model: true
  system_metrics: true
  system_metrics_sampling_interval: 10
```
