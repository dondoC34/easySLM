# Installation & Storage

## Prerequisites

- **Python**: `>= 3.12`
- **Hardware**: An NVIDIA GPU with CUDA support is recommended for fine-tuning (8 GB VRAM is sufficient for 0.5B–2.6B models with LoRA and gradient checkpointing). CPU and Apple Silicon MPS are also supported for inference and lightweight experiments.

---

## Install with `uv` (Recommended)

```bash
git clone https://github.com/dondoC34/easySLM.git
cd easySLM

# Create virtual environment and install all runtime dependencies
uv sync

# Verify CLI installation and inspect GPU / CUDA diagnostics
uv run easyslm info
```

To include development tools (`pytest`, `ruff`, `mypy`) or documentation tools (`mkdocs`):

```bash
uv sync --dev
uv sync --group docs
```

## Install with `pip`

```bash
pip install -e .
```

---

## Default Storage (`~/.easyslm/` & `EASYSLM_HOME`)

All runtime artifacts default to `~/.easyslm/` so that CLI commands, scripts, and Jupyter notebooks behave identically regardless of the current working directory:

| Artifact | Default Location | Explicit Override |
| :--- | :--- | :--- |
| Base HF Model Cache | `~/.easyslm/models` | `ModelFactory(cache_dir=...)` |
| Fine-Tuned Checkpoints | `~/.easyslm/finetuned` | `FineTuningConfig(output_dir=...)`, YAML `output_dir`, `SLMBase.from_best_model(models_dir=...)`, `--models-dir` |
| Batch Predictions | `~/.easyslm/predictions` | `easyslm predict --output-dir ...`, `model.predict_file_or_dir(output_dir=...)` |
| MLflow SQLite Database | `~/.easyslm/mlflow.db` | `mlflow-config.yml` (`backend_store_uri`), `FineTuningConfig(mlflow_tracking_uri=...)`, `--backend-store-uri` |
| MLflow Artifact Store | `~/.easyslm/mlruns` | `mlflow-config.yml` (`default_artifact_root`), `--default-artifact-root` |

### Relocating All Artifacts with `EASYSLM_HOME`

Set `EASYSLM_HOME` to move all default directories to a custom location (for instance, a secondary drive or shared cluster storage):

=== "Linux / macOS"

    ```bash
    export EASYSLM_HOME=/data/easyslm
    ```

=== "Windows (PowerShell)"

    ```powershell
    $env:EASYSLM_HOME = "D:\easyslm"
    ```

!!! tip "Precedence"
    Explicit arguments (`output_dir`, `cache_dir`, `models_dir`, CLI flags, or non-null YAML entries) always take precedence over `EASYSLM_HOME`.
