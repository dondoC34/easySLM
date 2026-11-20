# easySLM

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=materialformkdocs&logoColor=white)](https://dondoc34.github.io/easySLM/)
[![CI](https://github.com/dondoC34/easySLM/actions/workflows/ci.yml/badge.svg)](https://github.com/dondoC34/easySLM/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20%26%20PEFT-orange)](https://huggingface.co/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-E92063.svg)](https://docs.pydantic.dev/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Full Documentation (Tutorials, CLI & API Reference):** [https://dondoc34.github.io/easySLM/](https://dondoc34.github.io/easySLM/)

**easySLM** is a lightweight, generic, and production-ready framework for fine-tuning and running inference with **Small Language Models (SLMs)** (e.g., Qwen 2.5, LLaMA 3.2, Gemma 2, SmolLM2) on single-label and multi-label text classification tasks.

Designed for developer ergonomics and constrained hardware, easySLM runs comfortably on consumer GPUs with **8 GB of VRAM** (or less) via built-in support for **PEFT (LoRA)**, **Gradient Checkpointing**, **Mixed Precision (`bf16`/`fp16`)**, **Gradient Accumulation**, and **In-Memory Early Stopping**.

---

## Key Features

- **Dual Classification Architectures**:
  - **Direct Classification (`cls`)**: Extracts contextual sequence embeddings from the SLM backbone using configurable parametric pooling (`eos`, `mean`, `max`), normalizes them via `float32` `LayerNorm`, and projects them through a dense linear classification head (`nn.Linear`). Fast, deterministic, and calibrated.
  - **Generative Classification (`json`)**: Fine-tunes a Causal Language Model (`AutoModelForCausalLM`) with masked prompt tokens (`ignore_index=-100`) to autoregressively output alphabetically sorted JSON/Python lists of predicted classes (e.g., `['economy', 'technology']`) terminated by the model's native EOS token.
- **Automatic Task Detection (Single-Label & Multi-Label)**:
  - Automatically infers whether a dataset is **single-label** (1 label per sample $\rightarrow$ `CrossEntropyLoss` + Softmax) or **multi-label** ($>1$ label on any sample $\rightarrow$ `BCEWithLogitsLoss` + Sigmoid with top-1 fallback).
  - Supports minority-class **oversampling** (`max` or `sqrt` strategies with optional cap factor) and automatic train/validation splitting.
- **Pydantic-Validated Configuration (Single Source of Truth)**:
  - `FineTuningConfig` is a strict Pydantic v2 `BaseModel` (`extra="forbid"`, `validate_assignment=True`) with built-in defaults for every parameter—no external YAML file is required to run training, while custom YAML files or dictionaries remain fully supported.
- **Centralized, CWD-Independent Storage (`~/.easyslm/`)**:
  - Base-model caches, fine-tuned checkpoints, batch predictions, and MLflow tracking data default to `~/.easyslm/` (overridable globally via `EASYSLM_HOME` or per-call via parameters/configs), keeping repositories clean and notebooks working from any directory.
- **8 GB VRAM Optimization & Early Stopping**:
  - Combines LoRA adapters, gradient checkpointing, automatic mixed precision (`bf16`/`fp16`), and on-GPU logit `argmax` reduction during generative evaluation.
  - Includes `EarlyStoppingWindowCallback`, which snapshots only trainable parameters in CPU memory whenever a validation metric improves and automatically restores the best weights at the end of training.
- **Self-Contained Checkpoints, Model Cards & Auto-Best Inference**:
  - Every fine-tuning run saves LoRA adapters (or full weights), `classifier.pt` (for `cls` mode), tokenizer files, `easyslm_config.json`, a prompt snapshot (`prompt_config.yml`), diagnostic `plots/` & `tables/`, and a **Markdown Model Card** (`model_card_*.md`).
  - `easyslm predict` and `SLMBase.from_best_model()` automatically rank saved Model Cards by any metric (e.g., `eval_f1_macro`, `eval_loss`) and load the best checkpoint.
- **Full MLflow Experiment Tracking & Diagnostics**:
  - Logs hyperparameters, learning curves, system/GPU metrics, confusion matrices, per-class metric heatmaps, label cardinality distributions, and validation error tables to `~/.easyslm/mlflow.db`, managed via `easyslm mlflow`.

---

## Repository Structure

```text
easySLM/
├── configs/
│   ├── prompts.yml                  # Optional prompt templates for CLS and JSON modes
│   └── training_config.yml          # Optional YAML template to customize FineTuningConfig defaults
├── data/
│   └── agnews_tiny/                 # Balanced toy AG News dataset (train.txt: 200, val.txt: 40)
├── docs/                            # MkDocs documentation source (tutorials, CLI, API reference)
├── notebooks/
│   ├── fine_tune.ipynb              # Minimal end-to-end fine-tuning & inference quickstart
│   └── customize_fine_tuning.ipynb  # Full configuration & customization walkthrough
├── src/
│   ├── cli.py                       # Main CLI entrypoint (`ftune`, `predict`, `mlflow`, `info`)
│   ├── data/                        # Dataset classes (SLMDatasetBase, SLMClsDataset, SLMJsonDataset) & collator
│   ├── fine_tuning/                 # Pydantic FineTuningConfig, SLMFineTuner, LoRA, callbacks, & metrics
│   ├── slm/                         # SLMBase, SLMCls, SLMJson, ModelFactory, & checkpoint serialization
│   ├── tracking/                    # MLflow server lifecycle, run manager, diagnostic plots, & tables
│   └── utils/                       # EASYSLM_HOME path helpers, YAML/prompt loaders, CUDA info, & Model Cards
├── tests/                           # Unit test suite (pytest)
├── mlflow-config.yml                # Optional MLflow server and tracking overrides
├── mkdocs.yml                       # MkDocs Material & mkdocstrings configuration
├── pyproject.toml                   # Project metadata, dependencies, and Ruff/MyPy/Pytest settings
└── README.md                        # Project documentation
```

---

## Installation

### Prerequisites
- **Python**: `>= 3.12`
- **Hardware**: NVIDIA GPU with CUDA support recommended for fine-tuning (CPU and Apple Silicon MPS are also supported for inference and lightweight runs).

### Install with `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/dondoC34/easySLM.git
cd easySLM

# Create virtual environment and install all dependencies
uv sync

# Verify CLI installation and inspect hardware/CUDA status
uv run easyslm info
```

Alternatively, using standard `pip`:

```bash
pip install -e .
```

---

## Default Storage (`~/.easyslm/` & `EASYSLM_HOME`)

To stay independent of the current working directory and work seamlessly when installed as a library, all runtime artifacts default to `~/.easyslm/`:

| Artifact | Default Path | Override Options |
| :--- | :--- | :--- |
| **Base HF Model Cache** | `~/.easyslm/models` | `ModelFactory(cache_dir=...)` |
| **Fine-Tuned Checkpoints** | `~/.easyslm/finetuned` | `FineTuningConfig(output_dir=...)`, YAML `output_dir`, `SLMBase.from_best_model(models_dir=...)`, `--models-dir` |
| **Batch Predictions** | `~/.easyslm/predictions` | `easyslm predict --output-dir ...`, `model.predict_file_or_dir(output_dir=...)` |
| **MLflow SQLite DB** | `~/.easyslm/mlflow.db` | `mlflow-config.yml` (`backend_store_uri`), `FineTuningConfig(mlflow_tracking_uri=...)`, `--backend-store-uri` |
| **MLflow Artifacts** | `~/.easyslm/mlruns` | `mlflow-config.yml` (`default_artifact_root`), `--default-artifact-root` |

To relocate **all** default directories at once (e.g., to a secondary drive or shared volume), set the `EASYSLM_HOME` environment variable:

```bash
# Linux / macOS
export EASYSLM_HOME=/data/easyslm

# Windows (PowerShell)
$env:EASYSLM_HOME = "D:\easyslm"
```

---

## Dataset Format

easySLM uses a simple, human-readable `.txt` format using `|||` as the separator between the input text and its comma-separated labels.

### Training & Validation Data
Each non-empty line in a training/validation `.txt` file must follow `<text> ||| <label_1>, <label_2>, ...`:

```text
The central bank cut benchmark interest rates by 25 basis points. ||| economy
New offshore wind farms will supply clean power to three coastal cities. ||| energy, environment
Tech giants unveiled quantum computing chips for artificial intelligence. ||| technology
```

- **Automatic Task Inference**:
  - If every sample in the dataset has **exactly 1 label**, easySLM automatically configures **Single-Label** classification (`multi_label=False`).
  - If any sample has **2 or more labels**, easySLM automatically configures **Multi-Label** classification (`multi_label=True`).
- **Flexible Input Paths (`--data`)**:
  - **Single `.txt` file**: Automatically split into training and validation sets according to `val_split_ratio` in `FineTuningConfig` (default `0.15`).
  - **Directory path**: If you pass a directory containing `train.txt` and `val.txt` (or `validation.txt`), easySLM automatically uses them as the dedicated training and validation splits.

### Inference Data
For `easyslm predict`, input `.txt` files can contain:
- **Unlabeled lines**: Plain text without `|||` (each line is classified directly).
- **Annotated lines with `|||`**: easySLM strips the labels before running inference and includes them as `"ground_truth"` in the generated `.json` output for easy inspection.

---

## Command Line Interface (CLI)

Run `easyslm` (or `easyslm --help`) at any time to display formatted ASCII reference tables for all commands:

```bash
uv run easyslm --help
uv run easyslm ftune --help
uv run easyslm predict --help
uv run easyslm mlflow
```

### 1. Fine-Tuning (`easyslm ftune`)

Fine-tune any compatible Hugging Face SLM using built-in `FineTuningConfig` defaults—no YAML file is required. Pass `--config` and/or `--prompts-config` whenever you want to customize training parameters or prompt templates from YAML files.

#### Recommended Base Models
| Hugging Face Model ID | Size / VRAM | Highlights |
| :--- | :--- | :--- |
| `Qwen/Qwen2.5-0.5B` | 0.5B (< 2 GB) | Default model, multilingual (29+ languages), ultra-fast training |
| `Qwen/Qwen2.5-1.5B` | 1.5B (< 4 GB) | Higher capacity multilingual SLM, strong accuracy/speed balance |
| `HuggingFaceTB/SmolLM2-1.7B` | 1.7B (< 4 GB) | High-efficiency reasoning SLM trained on curated data |
| `meta-llama/Llama-3.2-1B` | 1.2B (< 3 GB) | Compact generalist model with 128k context window |
| `google/gemma-2-2b` | 2.6B (< 6 GB) | Gemma 2 architecture with sliding-window attention |

#### CLI Options (`easyslm ftune`)
| Option | Default | Description |
| :--- | :--- | :--- |
| `--model` | `Qwen/Qwen2.5-0.5B` | Hugging Face model ID or local path |
| `--data` | `data/agnews_tiny` | Path to a `.txt` dataset file or a directory containing `train.txt` (and optional `val.txt`) |
| `--type` | *(built-in: `cls` / from config)* | Optional classification mode override: `cls` (linear head) or `json` (generative list) |
| `--config` | *(none — built-in defaults)* | Optional path to a YAML training configuration file to customize `FineTuningConfig` |
| `--prompts-config` | *(built-in / from config)* | Optional override path to a custom prompt templates YAML file |
| `--epochs` | *(built-in: `3.0` / from config)* | Optional override for `num_train_epochs` |

#### Examples
```bash
# 1. Direct Classification (CLS mode) using built-in FineTuningConfig defaults
uv run easyslm ftune \
  --model Qwen/Qwen2.5-0.5B \
  --type cls \
  --data data/agnews_tiny \
  --epochs 3

# 2. Generative Classification (JSON mode)
uv run easyslm ftune \
  --model Qwen/Qwen2.5-0.5B \
  --type json \
  --data data/agnews_tiny \
  --epochs 3

# 3. Customize fine-tuning with an optional YAML config and prompt template
uv run easyslm ftune \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --type cls \
  --data data/agnews_tiny \
  --config configs/training_config.yml \
  --prompts-config configs/prompts.yml
```

#### What Happens After Fine-Tuning
Each run creates a timestamped folder `<output_dir>/<model_slug>_<type>_<YYYYMMDD_HHMMSS>/` (defaulting to `~/.easyslm/finetuned/...`) containing:
- `adapter_model.safetensors` & `adapter_config.json` (or full backbone weights when `use_lora=False`)
- `classifier.pt` (trained `LayerNorm` + `Linear` classification head in `cls` mode)
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`)
- `easyslm_config.json` (task metadata, sorted class list, pooling strategy, `multi_label` flag)
- `prompt_config.yml` (exact snapshot of the prompts used during training)
- `model_card_<model>_<type>_<timestamp>.md` (structured YAML frontmatter + Markdown tables)
- `plots/` (`confusion_matrix.png` or `cross_label_confusion_matrix.png`, `per_class_metrics.png`, `label_cardinality.png`)
- `tables/` (`confusion_matrix.json`, `per_class_metrics.json`, `label_cardinality.json`, `validation_errors.json`)

---

### 2. Batch Inference (`easyslm predict`)

Run batch predictions on a single `.txt` file or an entire directory of `.txt` files. When `--model` is omitted, easySLM automatically inspects all Model Cards in `--models-dir` and selects the best checkpoint according to `--metric`.

#### CLI Options (`easyslm predict`)
| Option | Default | Description |
| :--- | :--- | :--- |
| `--model` | *(auto-best)* | Path to a specific checkpoint directory. If omitted, selects the best model from `--models-dir` |
| `--models-dir` | `~/.easyslm/finetuned` | Directory searched for `model_card_*.md` files when `--model` is omitted (respects `EASYSLM_HOME`) |
| `--type` | `cls` | Model architecture mode to search for (`cls` or `json`) |
| `--data` | `data/agnews_tiny/val.txt` | Path to `.txt` file or directory of `.txt` files to classify |
| `--output-dir` | `~/.easyslm/predictions` | Output directory where `.txt` and `.json` predictions are written (respects `EASYSLM_HOME`) |
| `--metric` | `eval_f1_macro` | Metric used to rank Model Cards when `--model` is omitted (e.g., `eval_f1_macro`, `eval_accuracy`, `eval_loss`) |
| `--batch-size` | `8` | Inference batch size |
| `--prompts-config` | *(checkpoint config)* | Optional override for prompt templates (defaults to the checkpoint's `prompt_config.yml`) |

#### Examples
```bash
# Automatic best-model discovery (ranked by eval_f1_macro, outputs to ~/.easyslm/predictions)
uv run easyslm predict \
  --type cls \
  --data data/agnews_tiny/val.txt

# Automatic best-model discovery ranked by lowest validation loss with custom output folder
uv run easyslm predict \
  --type cls \
  --metric eval_loss \
  --data data/agnews_tiny/val.txt \
  --output-dir ./predictions/cls_lowest_loss

# Inference using an explicit checkpoint directory
uv run easyslm predict \
  --model ~/.easyslm/finetuned/Qwen2.5-0.5B_cls_20261120_120000 \
  --data data/agnews_tiny/val.txt \
  --output-dir ./predictions/cls_explicit
```

#### Prediction Output Files
For each input `<stem>.txt`, two files are generated in `--output-dir`:
1. `<stem>_predictions_<type>.txt`: Formatted lines `<text> ||| <predicted_labels>`.
2. `<stem>_predictions_<type>.json`: Detailed JSON array containing `"text"`, `"predictions"`, `"probabilities"` (in `cls` mode) or `"raw_output"` (in `json` mode), and `"ground_truth"` (when input lines include `|||`).

---

### 3. Experiment Tracking (`easyslm mlflow`)

easySLM provides a built-in CLI suite to manage the local MLflow tracking server, inspect runs, export best artifacts, and clean storage. By default, MLflow stores data in `~/.easyslm/mlflow.db` and `~/.easyslm/mlruns` (overridable via `EASYSLM_HOME`, `mlflow-config.yml`, or CLI flags):

```bash
# Display MLflow subcommand reference table
uv run easyslm mlflow

# Launch the MLflow tracking server and Web UI dashboard (default: http://127.0.0.1:5000)
uv run easyslm mlflow start --port 5000

# Check server status, SQLite DB size, artifact storage, and top metrics per experiment
uv run easyslm mlflow status

# List recent training runs sorted by F1, loss, or date
uv run easyslm mlflow list --limit 10 --sort-by f1

# Inspect the best run by metric and optionally export its logged artifacts
uv run easyslm mlflow best --metric eval_f1_macro --export-to ./best_run_artifacts

# Clean up MLflow artifacts or reset the tracking database
uv run easyslm mlflow clear --artifacts-only
```

---

### 4. Hardware Diagnostics (`easyslm info`)

Check your PyTorch version, CUDA availability, GPU device name, and VRAM capacity:

```bash
uv run easyslm info
```

---

## Configuration Reference

### 1. Fine-Tuning Configuration (`FineTuningConfig`)

[`FineTuningConfig`](src/fine_tuning/config.py) is a **Pydantic v2 `BaseModel`** serving as the single source of truth for all fine-tuning defaults and validation:
- **Strict key validation (`extra="forbid"`)**: passing any unrecognized key raises a `ValidationError` listing all accepted keys and their defaults.
- **Range & type validation (`validate_assignment=True`)**: validates types and numeric bounds on creation and mutation.
- **Multiple instantiation methods**:
  - `FineTuningConfig()` or `FineTuningConfig(mode="cls", num_train_epochs=2.0)` — built-in defaults + keyword overrides.
  - `FineTuningConfig.from_dict(data, **overrides)` — load from a partial or empty dictionary with optional overrides.
  - `FineTuningConfig.from_yaml("configs/training_config.yml", **overrides)` — load from an optional YAML file with optional keyword overrides.
  - Passing `config=None` to `SLMFineTuner(...)` automatically instantiates `FineTuningConfig()` with built-in defaults.

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `output_dir` | `~/.easyslm/finetuned` | Base directory for saved checkpoints and Model Cards (respects `EASYSLM_HOME`) |
| `mode` | `"cls"` | Classification architecture: `"cls"` or `"json"` |
| `pooling_strategy` | `"eos"` | Sequence pooling for `cls` mode: `"eos"`, `"mean"`, or `"max"` |
| `prompts_config_path` | `"configs/prompts.yml"` | Path to prompt templates YAML (falls back to built-in `DEFAULT_PROMPTS_CONFIG`) |
| `per_device_train_batch_size` | `1` | Training batch size per GPU (`>= 1`) |
| `per_device_eval_batch_size` | `4` | Validation batch size per GPU (`>= 1`) |
| `gradient_accumulation_steps` | `8` | Steps to accumulate gradients before optimizer update (`>= 1`) |
| `gradient_checkpointing` | `true` | Recomputes activations during backward pass to save VRAM |
| `use_lora` | `true` | Enables PEFT LoRA adapters (`false` for full fine-tuning) |
| `lora_r` / `lora_alpha` / `lora_dropout` | `8` / `16` / `0.05` | LoRA rank, scaling factor, and dropout probability |
| `lora_target_modules` | `"all-linear"` | Modules targeted by LoRA adapters (`str` or `list[str]`) |
| `learning_rate` | `null` | Auto-selects `2e-4` when `use_lora=True`, or `2e-5` when `False` |
| `lr_scheduler_type` / `warmup_ratio` | `"cosine"` / `0.1` | Learning rate schedule and warmup fraction (`[0.0, 1.0]`) |
| `weight_decay` / `max_grad_norm` | `0.01` / `1.0` | AdamW weight decay and gradient clipping threshold |
| `num_train_epochs` | `3.0` | Total number of training epochs (`> 0.0`) |
| `eval_strategy` / `eval_steps` | `"steps"` / `10` | Validation frequency (`"steps"`, `"epoch"`, or `"no"`) |
| `save_strategy` / `save_total_limit` | `"no"` / `2` | Intermediate Trainer checkpointing strategy and retention limit |
| `logging_steps` | `5` | Steps between training progress logs |
| `val_split_ratio` | `0.15` | Validation split ratio when `train_data_path` is a single `.txt` file (`[0.0, 1.0)`) |
| `oversample_strategy` | `null` | Minority-class oversampling on train split: `"max"`, `"sqrt"`, or `null` |
| `oversample_max_factor` | `null` | Optional maximum multiplication factor per class when oversampling |
| `early_stopping_window` | `4` | Consecutive evaluations without improvement before stopping (`null` to disable) |
| `early_stopping_metric` | `"eval_loss"` | Metric monitored by `EarlyStoppingWindowCallback` |
| `seed` | `42` | Random seed for reproducibility |
| `use_mlflow` | `true` | Enables automatic MLflow run, metric, plot, and table logging |
| `mlflow_tracking_uri` | `null` | Optional override for `MLFLOW_TRACKING_URI` (defaults to `mlflow-config.yml` / `~/.easyslm/mlflow.db`) |
| `mlflow_experiment_name` | `null` | Optional override for MLflow experiment name |
| `mlflow_config_path` | `null` | Optional path to a custom MLflow YAML configuration file |
| `run_name` | `null` | Optional custom run name for Trainer and MLflow |
| `max_errors_per_class` | `5` | Maximum false-positive examples per class logged in `validation_errors.json` |

### 2. Prompt Templates (`configs/prompts.yml`)
Customizes system prompts, user prompts, and full prompt formatting for both `cls` and `json` modes. When no custom prompt YAML is provided, easySLM uses the built-in `DEFAULT_PROMPTS_CONFIG`. Supported placeholders:
- `{classes}`: Alphabetically sorted, single-quoted class list (e.g., `'economy', 'sports'`).
- `{mode_desc}`: Task description from `mode_descriptions` (`multi_label` vs. `single_label`).
- `{example}`: Formatted example output for `json` mode.
- `{text}`, `{system_prompt}`, `{user_prompt}`: Prompt composition placeholders.

### 3. MLflow Settings (`mlflow-config.yml`)
Controls the local tracking server (`host`, `port`, `backend_store_uri`, `default_artifact_root`) and experiment tracking behavior (`experiment_prefix`, `system_metrics`, `system_metrics_sampling_interval`). Leaving `backend_store_uri` and `default_artifact_root` as `null` resolves them automatically under `~/.easyslm/` (`EASYSLM_HOME`).

---

## Evaluation Metrics & Artifacts

During validation and post-training evaluation, easySLM computes and logs:

| Metric | Task Scope | Description |
| :--- | :--- | :--- |
| `eval_loss` | All | Validation loss (`CrossEntropyLoss` or `BCEWithLogitsLoss`) |
| `eval_f1_macro` / `eval_f1_micro` / `eval_f1_weighted` | All | Macro, micro, and support-weighted F1 scores |
| `eval_precision_*` / `eval_recall_*` | All | Macro, micro, and weighted precision and recall |
| `eval_exact_match_accuracy` | All | Subset accuracy (all predicted labels match ground truth) |
| `eval_hamming_accuracy` / `eval_hamming_loss` | All | Per-label binary assignment accuracy and loss |
| `eval_balanced_accuracy` | Single-label | Macro-averaged recall across classes (robust to imbalance) |
| `eval_matthews_corrcoef` | Single-label | Matthews correlation coefficient (MCC, range `-1` to `+1`) |
| `eval_cohen_kappa` | Single-label | Cohen's kappa chance-adjusted agreement |
| `eval_token_accuracy` | JSON mode | Next-token prediction accuracy on assistant response tokens |
| `eval_json_valid_parse_rate` | JSON mode | Fraction of generations parsed as valid JSON/Python lists |

In addition to scalar metrics, easySLM generates visual and tabular diagnostics in `<checkpoint>/plots/`, `<checkpoint>/tables/`, and MLflow:
- **Confusion Matrix**: Standard confusion matrix (`confusion_matrix.png`) for single-label tasks, or Cross-Label co-occurrence matrix (`cross_label_confusion_matrix.png`) for multi-label tasks.
- **Per-Class Metrics Heatmap & Table**: Precision, Recall, F1, and Support per class (`per_class_metrics.png` & `tables/per_class_metrics.json`).
- **Label Cardinality Distribution**: True vs. predicted label count distribution per sample (`label_cardinality.png` & `.json`).
- **Validation Errors Table**: Ranked table of misclassified validation samples with raw outputs and confidence scores (`validation_errors.json`).

---

## Python API & Notebooks

Interactive examples are available in `notebooks/`:
- [`notebooks/fine_tune.ipynb`](notebooks/fine_tune.ipynb): Minimal quickstart using built-in `FineTuningConfig` defaults.
- [`notebooks/customize_fine_tuning.ipynb`](notebooks/customize_fine_tuning.ipynb): Full walkthrough exposing every configurable option in `ModelFactory`, `FineTuningConfig`, `SLMFineTuner`, and `SLMBase`.

```python
from src import FineTuningConfig, SLMBase, SLMFineTuner

# 1. Configure fine-tuning programmatically (uses built-in defaults for all omitted fields)
#    Alternatively: FineTuningConfig.from_yaml("configs/training_config.yml", num_train_epochs=2.0)
config = FineTuningConfig(
    mode="cls",
    num_train_epochs=2.0,
    use_mlflow=False,
)

tuner = SLMFineTuner(model_name="Qwen/Qwen2.5-0.5B", config=config)
results = tuner.train(train_data_path="data/agnews_tiny")
print("Saved checkpoint to:", results["checkpoint_dir"])
print("Validation metrics:", results["eval_metrics"])

# 2. Load the best checkpoint from ~/.easyslm/finetuned (or use SLMBase.from_checkpoint(...))
model = SLMBase.from_best_model(
    model_type="cls",
    metric="eval_f1_macro",
)

# 3. Run batch inference on raw strings
predictions = model.predict_texts(
    [
        "The central bank lowered benchmark interest rates by 25 basis points.",
        "A star striker completed a record-breaking transfer to join the championship squad.",
    ]
)
for item in predictions:
    print(item["text"], "->", item["predictions"], item.get("probabilities"))
```

---

## Development & Code Quality

This repository uses **Pytest** for unit tests, **Ruff** for linting and formatting, and **MyPy** for static type checking:

```bash
# Run unit test suite
uv run pytest

# Run linter and formatter checks
uv run ruff check .
uv run ruff format --check .

# Run static type checking
uv run mypy src

# Preview documentation locally with live reload (opens http://127.0.0.1:8000 automatically)
uv sync --group docs
uv run easyslm docs start --port 8000
```

---

## License

This project is licensed under the MIT License.
