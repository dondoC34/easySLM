# easySLM

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%7C%20CUDA%2012.4-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20%26%20PEFT-orange)](https://huggingface.co/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**easySLM** is a lightweight, generic, and production-ready framework for fine-tuning **Small Language Models (SLMs)** (e.g. Qwen 2.5, LLaMA 3, Gemma, Mistral) on text classification tasks.

Designed specifically for developer efficiency and constrained hardware, easySLM runs comfortably on consumer GPUs with **8 GB of VRAM** (or less) through native support for **PEFT (LoRA)**, **Gradient Checkpointing**, **bfloat16/fp16**, and smart **Gradient Accumulation**.

---

## Key Features

- **Dual Architecture Modes**:
  - **Direct Classification (`cls`)**: Extracts sequence embeddings from the SLM backbone via customizable pooling (`eos`, `mean`, `max`) and feeds them into a custom classification head (`nn.Linear`). Fast, deterministic, and accurate.
  - **Generative Classification (`json`)**: Instructs a Causal Language Model via structured plain-text prompts to autoregressively output sorted JSON lists of predicted classes (e.g. `['economy', 'technology']`) terminated by the model's native EOS token.
- **Single-Label & Multi-Label Support**:
  - Full support for multi-label tagging (BCEWithLogitsLoss / Sigmoid) and single-label multiclass classification (CrossEntropyLoss / Softmax).
  - Strict eager validation: automatically validates dataset consistency when `--single-label` is enabled, pinpointing file and line errors immediately.
- **Config-Driven Architecture**:
  - Prompt templates (`configs/prompts.yml`) and training hyperparameters (`configs/training_config.yml`) are fully decoupled into documented YAML files.
- **Automated Model Cards**:
  - Every fine-tuning run automatically documents its results in a Markdown **Model Card** (`model_card_{model}_{type}_{timestamp}.md`) with structured YAML frontmatter, hyperparameter snapshots, and metrics tables with human-readable explanations.
- **Automated "Best Model" Inference**:
  - The CLI can automatically search your checkpoints, parse Model Cards according to any metric (e.g. `eval_f1_macro`, `eval_accuracy`), and run batch inference using the best discovered model.
- **Experiment Tracking**:
  - Built-in integration with **MLflow** (`sqlite:///mlflow.db`) tracking training loss, evaluation metrics, and run configurations.
- **Rich CLI with Help Tables**:
  - Global `easyslm` executable with formatted terminal reference tables for `ftune` and `predict`.

---

## Directory Structure

```text
easySLM/
├── configs/
│   ├── prompts.yml                # System, user, and full prompt templates for CLS and JSON
│   └── training_config.yml        # Hyperparameters and Hugging Face Trainer configuration
├── data/
│   ├── sample_news.txt            # Multi-label dataset (210 samples, 6 classes)
│   ├── sample_news_validation.txt # Unseen validation headlines for batch inference
│   ├── sample_news_single_label_train.txt # Single-label training dataset (72 samples)
│   └── sample_news_single_label_val.txt   # Single-label validation dataset (18 samples)
├── models/
│   └── finetuned/                 # Checkpoints, adapters, and Model Cards (model_card_*.md)
├── predictions/                   # Generated prediction files (.txt and .json)
├── src/
│   ├── cli.py                     # CLI commands (`ftune`, `predict`, `info`)
│   ├── data/                      # Dataset loaders, parsers (base_dataset.py), and collators
│   ├── slm/                       # SLMBase, SLMCls, SLMJson, ModelFactory, serialization & inference
│   ├── fine_tuning.py             # SLMFineTuner with HF Trainer & MLflow
│   ├── evaluation.py              # Metric computation (Accuracy, F1, MCC, Cohen's Kappa, etc.)
│   └── utils/                     # Model card generator, help tables, CUDA diagnostics
├── main.py                        # Entrypoint alias
├── mlflow.db                      # Local SQLite database for MLflow experiment tracking
├── pyproject.toml                 # Dependencies and CLI entry points
└── README.md                      # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA drivers (recommended for training; CPU inference also supported)

### Setup with `uv` (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/easySLM.git
cd easySLM

# Install dependencies into virtual environment
uv sync

# Or install in editable mode
uv pip install -e .
```

---

## Data Format

easySLM accepts plain `.txt` files or directories containing multiple `.txt` files.

### Training & Validation Data
Each line must follow the format `<text> ||| <comma_separated_labels>`:

```text
The central bank cut benchmark interest rates by 25 basis points. ||| economy
New offshore wind farms will supply clean power to three coastal cities. ||| energy, environment
Tech giants unveiled quantum computing chips for artificial intelligence. ||| technology
```

- In **multi-label mode** (`default`), each sample can have one or more classes separated by commas.
- In **single-label mode** (`--single-label`), each sample must contain exactly one class. If a row violates this rule, easySLM reports the exact file, line number, and offending content.

### Inference Data
For prediction (`easyslm predict`), the input can be:
- **Plain text lines** (no pipes needed): each line is treated as an unlabeled text to classify.
- **Annotated lines with `|||`**: easySLM extracts the text before `|||` for inference and preserves the annotations as `ground_truth` in the JSON output for evaluation.

---

## Command Line Interface (CLI)

easySLM provides the global `easyslm` CLI. Run `easyslm --help` to inspect all parameters:

```bash
uv run easyslm --help
uv run easyslm ftune --help
uv run easyslm predict --help
```

### 1. Fine-Tuning: `easyslm ftune`

Fine-tune any compatible Hugging Face Small Language Model (SLM) using PEFT LoRA and Hugging Face Trainer.

#### Recommended Base Models
| Hugging Face Model ID | Parameters / VRAM | Highlights |
| :--- | :--- | :--- |
| `Qwen/Qwen2.5-0.5B` | 0.5B (< 2 GB) | Default model, multilingual (29+ languages), ultra-fast training |
| `Qwen/Qwen2.5-1.5B` | 1.5B (< 4 GB) | High capacity multilingual SLM, great accuracy/speed balance |
| `meta-llama/Llama-3.2-1B` | 1.2B (< 3 GB) | Meta compact generalist with 128k context window |
| `google/gemma-2-2b` | 2.6B (< 6 GB) | Google Gemma 2 architecture with sliding-window attention |
| `HuggingFaceTB/SmolLM2-1.7B` | 1.7B (< 4 GB) | Compact reasoning SLM trained on curated synthetic data |

```bash
# Direct Classification (CLS mode) with LoRA on multi-label data
uv run easyslm ftune \
  --model Qwen/Qwen2.5-0.5B \
  --type cls \
  --data data/sample_news.txt \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum 2 \
  --output-dir ./models/finetuned

# Single-Label Classification
uv run easyslm ftune \
  --model Qwen/Qwen2.5-0.5B \
  --type cls \
  --single-label \
  --data data/sample_news_single_label_train.txt \
  --epochs 3 \
  --batch-size 4

# Generative Classification (JSON mode)
uv run easyslm ftune \
  --model Qwen/Qwen2.5-0.5B \
  --type json \
  --data data/sample_news.txt \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4
```

When fine-tuning completes, easySLM automatically:
1. Saves LoRA adapter weights (or full model if `--no-lora` is set).
2. For `cls` mode, saves the trained classification head (`classifier.pt`).
3. Saves the tokenizer and `easyslm_config.json`.
4. Writes a structured **Markdown Model Card** (`model_card_<model>_<type>_<timestamp>.md`) recording all validation metrics and training arguments.
5. Logs parameters, metrics, and models to MLflow.

### 2. Batch Inference: `easyslm predict`

Run predictions on a `.txt` file or an entire directory of `.txt` files:

```bash
# Automatic: Discovers and loads the BEST model saved in models/finetuned
uv run easyslm predict \
  --type cls \
  --data data/sample_news_validation.txt \
  --output-dir predictions/cls_best

# Automatic best selection ranked by a custom metric (e.g. eval_accuracy or eval_loss)
uv run easyslm predict \
  --type cls \
  --metric eval_accuracy \
  --data data/sample_news_single_label_val.txt \
  --output-dir predictions/cls_best_acc

# Explicit checkpoint path
uv run easyslm predict \
  --type cls \
  --model models/finetuned/Qwen2.5-0.5B_cls_20261104_121530 \
  --data data/sample_news_validation.txt \
  --output-dir predictions/cls_explicit
```

**Output Formats**:
Predictions are written to the target directory in two formats:
- `<filename>_predictions_<type>.txt`: Plain text with `<text> ||| <predicted_labels>`
- `<filename>_predictions_<type>.json`: Structured JSON containing predictions, class probability distributions, and raw generation outputs (for JSON mode).

### 3. MLflow Tracking: `easyslm mlflow`

easySLM includes a built-in suite to manage the MLflow tracking server, query runs, inspect storage diagnostics, and open the web dashboard:

```bash
# Display command reference and available subcommands
uv run easyslm mlflow

# Check server status, database size, and experiment summary
uv run easyslm mlflow status

# List recent training runs with metrics and status
uv run easyslm mlflow list --limit 10 --sort-by f1

# Display details of the best performing run (and optionally export artifacts)
uv run easyslm mlflow best --metric eval_f1_macro --export-to ./best_model_export

# Launch the official MLflow Web UI server
uv run easyslm mlflow start --port 5000

# Open the Web UI dashboard in your default browser
uv run easyslm mlflow ui

# Clean database records or artifacts
uv run easyslm mlflow clear --artifacts-only
```

### 4. Hardware Diagnostics: `easyslm info`

Inspect your local CUDA setup, GPU name, VRAM availability, and PyTorch configuration:

```bash
uv run easyslm info
```

---

## Evaluation Metrics

easySLM calculates and logs task-appropriate metrics:

| Metric | Mode | Description |
| :--- | :--- | :--- |
| `eval_loss` | All | Validation cross-entropy / BCE loss |
| `eval_f1_macro` | All | Unweighted macro-average F1 across all classes |
| `eval_f1_micro` | All | Global aggregate F1 across all classes |
| `eval_exact_match_accuracy` | Multi-label | Fraction of samples where all predicted labels match ground truth exactly |
| `eval_hamming_accuracy` | Multi-label | Fraction of individual binary label decisions that are correct |
| `eval_accuracy` | Single-label | Top-1 multiclass classification accuracy |
| `eval_balanced_accuracy` | Single-label | Macro-average of recall per class (robust to class imbalance) |
| `eval_matthews_corrcoef` | Single-label | Matthews correlation coefficient (MCC, range -1 to +1) |
| `eval_cohen_kappa` | Single-label | Chance-adjusted agreement between predictions and truth |
| `token_accuracy` | JSON mode | Next-token prediction accuracy on generated assistant tokens |
| `json_valid_parse_rate` | JSON mode | Fraction of generations successfully parsed as valid JSON lists |

---

## Experiment Tracking with MLflow

All training runs are automatically tracked in local MLflow SQLite store. To explore runs, compare learning curves, and view logged parameters:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Python API Usage

You can also use easySLM programmatically within your own scripts:

```python
from src import FineTuningConfig, SLMBase, SLMFineTuner

# 1. Fine-tuning programmatically
config = FineTuningConfig(
    output_dir="./models/finetuned",
    mode="cls",
    multi_label=False,
    num_train_epochs=3.0,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    use_lora=True,
    lora_r=8,
)
tuner = SLMFineTuner(model_name="Qwen/Qwen2.5-0.5B", config=config)
tuner.train(train_data_path="data/sample_news_single_label_train.txt")

# 2. Inference programmatically
model = SLMBase.from_best_model(
    models_dir="./models/finetuned",
    model_type="cls",
    metric="eval_accuracy",
)
results = model.predict_texts(
    [
        "The central bank lowered benchmark interest rates by 25 basis points.",
        "A star striker completed a record-breaking transfer to join the championship squad.",
    ]
)
for r in results:
    print(r["text"], "->", r["predictions"], r["probabilities"])
```

---

## Configuration Files

- `configs/training_config.yml`: Controls learning rate, scheduler (`cosine`, `linear`), warmup ratio, weight decay, batch sizes, LoRA rank/alpha, and gradient checkpointing.
- `configs/prompts.yml`: System prompts, user prompt templates, and full prompt structures for direct (`cls`) and generative (`json`) modes.

---

## License

This project is licensed under the MIT License.
