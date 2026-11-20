# easySLM

[![CI](https://github.com/dondoC34/easySLM/actions/workflows/ci.yml/badge.svg)](https://github.com/dondoC34/easySLM/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20%26%20PEFT-orange)](https://huggingface.co/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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
- **8 GB VRAM Optimization & Early Stopping**:
  - Combines LoRA adapters, gradient checkpointing, automatic mixed precision (`bf16`/`fp16`), and on-GPU logit `argmax` reduction during generative evaluation.
  - Includes `EarlyStoppingWindowCallback`, which snapshots only trainable parameters in CPU memory whenever a validation metric improves and automatically restores the best weights at the end of training.
- **Self-Contained Checkpoints & Automated Model Cards**:
  - Every fine-tuning run saves a self-contained checkpoint containing LoRA adapters (or full weights), `classifier.pt` (for `cls` mode), tokenizer files, `easyslm_config.json`, a snapshot of the prompt templates (`prompt_config.yml`), diagnostic plots (`plots/`), JSON tables (`tables/`), and a **Markdown Model Card** (`model_card_*.md`) with YAML frontmatter.
- **Automated "Best Model" Discovery for Inference**:
  - `easyslm predict` and `SLMBase.from_best_model()` automatically scan saved Model Cards, rank checkpoints by any target metric (e.g., `eval_f1_macro`, `eval_accuracy`, `eval_loss`), and load the best model for batch inference.
- **Full MLflow Experiment Tracking & Diagnostics**:
  - Tracks hyperparameters, learning curves, system/GPU metrics, confusion matrices (single-label and multi-label cross-label), per-class metric heatmaps, label cardinality distributions, and validation error tables in a local SQLite backend (`mlflow.db`), managed through `easyslm mlflow`.

---

## Repository Structure

```text
easySLM/
├── configs/
│   ├── prompts.yml                # System, user, and full prompt templates for CLS and JSON modes
│   └── training_config.yml        # Hyperparameters, LoRA, early stopping, oversampling, and Trainer settings
├── data/
│   └── agnews_tiny/               # Balanced toy AG News dataset (train.txt: 200, val.txt: 40)
├── notebooks/                     # Minimal interactive fine-tuning & prediction notebooks
├── models/
│   └── finetuned/                 # Saved checkpoints, Model Cards, plots, and evaluation tables
├── predictions/                   # Batch inference outputs (.txt and .json)
├── src/
│   ├── cli.py                     # Main CLI entrypoint (`ftune`, `predict`, `mlflow`, `info`)
│   ├── data/                      # Dataset classes (SLMDatasetBase, SLMClsDataset, SLMJsonDataset) & collator
│   ├── fine_tuning/               # FineTuningConfig, SLMFineTuner, LoRA setup, callbacks, & evaluation metrics
│   ├── slm/                       # SLMBase, SLMCls, SLMJson, ModelFactory, & checkpoint serialization
│   ├── tracking/                  # MLflow server lifecycle, run manager, diagnostic plots, & tables
│   └── utils/                     # Config loaders, CUDA diagnostics, CLI help tables, & Model Card parser
├── main.py                        # CLI entrypoint alias
├── mlflow-config.yml              # MLflow server and tracking configuration
├── pyproject.toml                 # Project metadata, dependencies, and linter/type-checker settings
└── README.md                      # Project documentation
```

---

## Installation

### Prerequisites
- **Python**: `>= 3.12`
- **Hardware**: NVIDIA GPU with CUDA support recommended for fine-tuning (CPU and Apple Silicon MPS are also supported for inference and lightweight runs).

### Install with `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/easySLM.git
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
  - **Single `.txt` file**: Automatically split into training and validation sets according to `val_split_ratio` in `configs/training_config.yml` (default `0.15`).
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

Fine-tune any compatible Hugging Face SLM using `configs/training_config.yml` and `configs/prompts.yml`.

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
| `--type` | *(from config)* | Optional classification mode override: `cls` (linear head) or `json` (generative list) |
| `--config` | `configs/training_config.yml` | Path to the training configuration YAML file |
| `--prompts-config` | *(from config)* | Optional override path to the prompt templates YAML file |
| `--epochs` | *(from config)* | Optional override for `num_train_epochs` |

#### Examples
```bash
# 1. Direct Classification (CLS mode) on the built-in agnews_tiny toy dataset
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

# 3. Fine-tuning with a custom config and prompt template
uv run easyslm ftune \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --type cls \
  --data data/agnews_tiny \
  --config configs/training_config.yml \
  --prompts-config configs/prompts.yml
```

#### What Happens After Fine-Tuning
Each run creates a timestamped folder `<output_dir>/<model_slug>_<type>_<YYYYMMDD_HHMMSS>/` containing:
- `adapter_model.safetensors` & `adapter_config.json` (or full backbone weights when `use_lora: false`)
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
| `--models-dir` | `./models/finetuned` | Directory searched for `model_card_*.md` files when `--model` is omitted |
| `--type` | `cls` | Model architecture mode to search for (`cls` or `json`) |
| `--data` | `data/agnews_tiny/val.txt` | Path to `.txt` file or directory of `.txt` files to classify |
| `--output-dir` | `predictions` | Output directory where `.txt` and `.json` predictions are written |
| `--metric` | `eval_f1_macro` | Metric used to rank Model Cards when `--model` is omitted (e.g., `eval_f1_macro`, `eval_accuracy`, `eval_loss`) |
| `--batch-size` | `8` | Inference batch size |
| `--prompts-config` | *(checkpoint config)* | Optional override for prompt templates (defaults to the checkpoint's `prompt_config.yml`) |

#### Examples
```bash
# Automatic best-model discovery (ranked by eval_f1_macro)
uv run easyslm predict \
  --type cls \
  --data data/agnews_tiny/val.txt \
  --output-dir predictions/cls_best

# Automatic best-model discovery ranked by lowest validation loss
uv run easyslm predict \
  --type cls \
  --metric eval_loss \
  --data data/agnews_tiny/val.txt \
  --output-dir predictions/cls_lowest_loss

# Inference using an explicit checkpoint directory
uv run easyslm predict \
  --model models/finetuned/Qwen2.5-0.5B_cls_20261114_190335 \
  --data data/agnews_tiny/val.txt \
  --output-dir predictions/cls_explicit
```

#### Prediction Output Files
For each input `<stem>.txt`, two files are generated in `--output-dir`:
1. `<stem>_predictions_<type>.txt`: Formatted lines `<text> ||| <predicted_labels>`.
2. `<stem>_predictions_<type>.json`: Detailed JSON array containing `"text"`, `"predictions"`, `"probabilities"` (in `cls` mode) or `"raw_output"` (in `json` mode), and `"ground_truth"` (when input lines include `|||`).

---

### 3. Experiment Tracking (`easyslm mlflow`)

easySLM provides a built-in CLI suite to manage the local MLflow tracking server, inspect runs, export best artifacts, and clean storage (configured via `mlflow-config.yml`):

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

## Configuration Files

### 1. Training Configuration (`configs/training_config.yml`)
All training hyperparameters and hardware settings are centralized in `configs/training_config.yml`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `mode` | `"cls"` | Classification architecture: `"cls"` or `"json"` |
| `pooling_strategy` | `"eos"` | Sequence pooling for `cls` mode: `"eos"`, `"mean"`, or `"max"` |
| `prompts_config_path` | `"configs/prompts.yml"` | Path to prompt templates YAML file |
| `output_dir` | `"./models/finetuned"` | Base directory for saved checkpoints and Model Cards |
| `per_device_train_batch_size` | `1` | Training batch size per GPU |
| `per_device_eval_batch_size` | `4` | Validation batch size per GPU |
| `gradient_accumulation_steps` | `8` | Steps to accumulate gradients before optimizer update |
| `gradient_checkpointing` | `true` | Recomputes activations during backward pass to save VRAM |
| `use_lora` | `true` | Enables PEFT LoRA adapters (`false` for full fine-tuning) |
| `lora_r` / `lora_alpha` / `lora_dropout` | `8` / `16` / `0.05` | LoRA rank, scaling factor, and dropout probability |
| `lora_target_modules` | `"all-linear"` | Modules targeted by LoRA adapters |
| `learning_rate` | `null` | Auto-selects `2e-4` when `use_lora: true`, or `2e-5` when `false` |
| `lr_scheduler_type` / `warmup_ratio` | `"cosine"` / `0.1` | Learning rate schedule and warmup fraction |
| `num_train_epochs` | `3.0` | Total number of training epochs |
| `eval_strategy` / `eval_steps` | `"steps"` / `10` | Validation frequency (`"steps"`, `"epoch"`, or `"no"`) |
| `val_split_ratio` | `0.15` | Validation split ratio when `--data` is a single `.txt` file |
| `oversample_strategy` | `null` | Minority-class oversampling on train split: `"max"`, `"sqrt"`, or `null` |
| `oversample_max_factor` | `null` | Optional maximum multiplication factor per class when oversampling |
| `early_stopping_window` | `4` | Consecutive evaluations without improvement before stopping (`null` to disable) |
| `early_stopping_metric` | `"eval_loss"` | Metric monitored by `EarlyStoppingWindowCallback` |
| `use_mlflow` | `true` | Enables automatic MLflow run, metric, plot, and table logging |
| `max_errors_per_class` | `5` | Maximum false-positive examples per class logged in `validation_errors.json` |

### 2. Prompt Templates (`configs/prompts.yml`)
Customizes system prompts, user prompts, and full prompt formatting for both `cls` and `json` modes. Supported placeholders:
- `{classes}`: Alphabetically sorted, single-quoted class list (e.g., `'economy', 'sports'`).
- `{mode_desc}`: Task description from `mode_descriptions` (`multi_label` vs. `single_label`).
- `{example}`: Formatted example output for `json` mode.
- `{text}`, `{system_prompt}`, `{user_prompt}`: Prompt composition placeholders.

### 3. MLflow Settings (`mlflow-config.yml`)
Controls the local tracking server (`host`, `port`, `backend_store_uri`, `default_artifact_root`) and experiment tracking behavior (`experiment_prefix`, `system_metrics`, `system_metrics_sampling_interval`).

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
- **Per-Class Metrics Heatmap**: Precision, Recall, F1, and Support per class (`per_class_metrics.png` & `.json`).
- **Label Cardinality Distribution**: True vs. predicted label count distribution per sample (`label_cardinality.png` & `.json`).
- **Validation Errors Table**: Ranked table of misclassified validation samples with raw outputs and confidence scores (`validation_errors.json`).

---

## Python API Usage

All components are exposed under `src` for use in custom Python scripts or notebooks:

```python
from src import FineTuningConfig, SLMBase, SLMFineTuner

# 1. Configure and run fine-tuning programmatically
config = FineTuningConfig.from_yaml(
    "configs/training_config.yml",
    mode="cls",
    num_train_epochs=3.0,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    early_stopping_window=4,
    early_stopping_metric="eval_loss",
)

tuner = SLMFineTuner(model_name="Qwen/Qwen2.5-0.5B", config=config)
results = tuner.train(train_data_path="data/agnews_tiny")
print("Saved checkpoint to:", results["checkpoint_dir"])
print("Validation metrics:", results["eval_metrics"])

# 2. Load the best discovered model (or use SLMBase.from_checkpoint(results["checkpoint_dir"]))
model = SLMBase.from_best_model(
    models_dir="./models/finetuned",
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

This repository uses **Ruff** for linting and formatting and **MyPy** for static type checking:

```bash
# Run linter and formatter checks
uv run ruff check .
uv run ruff format --check .

# Run static type checking
uv run mypy src
```

---

## License

This project is licensed under the MIT License.
