# CLI Reference

easySLM installs the `easyslm` console script with four subcommands: `ftune`, `predict`, `mlflow`, and `info`.

```bash
uv run easyslm --help
```

---

## `easyslm ftune`

Fine-tune an SLM on a `.txt` dataset file or directory. Uses built-in `FineTuningConfig` defaults unless `--config` is provided.

| Option | Default | Description |
| :--- | :--- | :--- |
| `--model` | `Qwen/Qwen2.5-0.5B` | Hugging Face model ID or local path |
| `--data` | `data/agnews_tiny` | `.txt` file or directory with `train.txt` (and optional `val.txt`) |
| `--type` | *(built-in: `cls`)* | Override classification mode: `cls` or `json` |
| `--config` | `None` | Optional path to a YAML training configuration file |
| `--prompts-config` | `None` | Optional override path to a prompt templates YAML file |
| `--epochs` | `None` | Optional override for `num_train_epochs` |

```bash
uv run easyslm ftune --model Qwen/Qwen2.5-0.5B --type cls --data data/agnews_tiny --epochs 2
```

---

## `easyslm predict`

Run batch inference on a `.txt` file or directory of `.txt` files.

| Option | Default | Description |
| :--- | :--- | :--- |
| `--model` | `None` *(auto-best)* | Explicit checkpoint directory; if omitted, selects best from `--models-dir` |
| `--models-dir` | `~/.easyslm/finetuned` | Directory scanned for Model Cards (respects `EASYSLM_HOME`) |
| `--type` | `cls` | Model architecture to search for (`cls` or `json`) |
| `--data` | `data/agnews_tiny/val.txt` | Input `.txt` file or directory |
| `--output-dir` | `~/.easyslm/predictions` | Output directory for `.txt` and `.json` predictions (respects `EASYSLM_HOME`) |
| `--metric` | `eval_f1_macro` | Ranking metric for best-model selection |
| `--batch-size` | `8` | Inference batch size |
| `--prompts-config` | `None` | Optional prompt YAML override (defaults to checkpoint's `prompt_config.yml`) |

```bash
uv run easyslm predict --type cls --data data/agnews_tiny/val.txt --metric eval_f1_macro
```

---

## `easyslm mlflow`

Manage the local MLflow tracking server and experiment storage.

| Subcommand | Description |
| :--- | :--- |
| `easyslm mlflow start` (`run`) | Launch the MLflow tracking server & Web UI (`--host`, `--port`, `--backend-store-uri`, `--default-artifact-root`) |
| `easyslm mlflow status` | Probe server health, DB size, artifact usage, and top metrics |
| `easyslm mlflow list` (`runs`) | Tabulate recent runs (`--experiment`, `--model`, `--sort-by`, `--limit`) |
| `easyslm mlflow best` | Show best run by metric and optionally export artifacts (`--metric`, `--export-to`) |
| `easyslm mlflow clear` (`clean`) | Delete MLflow artifacts and/or SQLite database (`--artifacts-only`, `-y`) |

---

## `easyslm docs`

Preview the MkDocs documentation locally with live reload and automatic browser launch.

```bash
# Serve docs on default port 8000 and open browser automatically
uv run easyslm docs start

# Custom port (and optionally --no-open to skip launching the browser)
uv run easyslm docs start --port 8080
```

---

## `easyslm info`

Display PyTorch version, CUDA availability, GPU model, VRAM, and native `bfloat16` support.

```bash
uv run easyslm info
```
