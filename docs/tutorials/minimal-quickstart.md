# Minimal Quickstart

This tutorial walks through the fastest way to fine-tune `Qwen/Qwen2.5-0.5B` on the built-in `data/agnews_tiny` dataset (200 training / 40 validation samples across 4 news categories: `business`, `sci_tech`, `sports`, `world`) and run inference on new text.

!!! tip "Interactive Notebook"
    You can run this exact workflow interactively in [`notebooks/fine_tune.ipynb`](https://github.com/dondoC34/easySLM/blob/main/notebooks/fine_tune.ipynb).

---

## Option A: Python API

### 1. Configure and Fine-Tune

No external YAML file is required—`FineTuningConfig` supplies sensible 8 GB VRAM defaults for every omitted parameter:

```python
from src import FineTuningConfig, SLMBase, SLMFineTuner
from src.utils import get_project_root

config = FineTuningConfig(
    mode="cls",
    num_train_epochs=2.0,
    use_mlflow=False,
)

tuner = SLMFineTuner(model_name="Qwen/Qwen2.5-0.5B", config=config)
results = tuner.train(train_data_path=get_project_root() / "data" / "agnews_tiny")

print("Saved checkpoint:", results["checkpoint_dir"])
```

Behind the scenes, `tuner.train(...)`:

1. Downloads (or reuses) `Qwen/Qwen2.5-0.5B` in `~/.easyslm/models/`.
2. Auto-discovers `train.txt` and `val.txt` inside `data/agnews_tiny/` and detects single-label classification.
3. Wraps the backbone with LoRA adapters (`r=8, alpha=16`), gradient checkpointing, and `bf16`/`fp16` mixed precision.
4. Evaluates every 10 steps, restores the best weights via `EarlyStoppingWindowCallback`, and saves a self-contained checkpoint + Model Card + plots/tables in `~/.easyslm/finetuned/`.

### 2. Inspect Per-Class Validation Metrics

`SLMFineTuner.evaluate()` automatically writes `tables/per_class_metrics.json` inside the checkpoint directory (even when `use_mlflow=False`):

```python
import json
from pathlib import Path

metrics_file = Path(results["checkpoint_dir"]) / "tables" / "per_class_metrics.json"
per_class = json.loads(metrics_file.read_text(encoding="utf-8"))

print(f"{'Class':<15} | {'Precision':>9} | {'Recall':>9} | {'F1':>9}")
print("-" * 51)
for cls_name, prec, rec, f1 in zip(
    per_class["class_name"],
    per_class["precision"],
    per_class["recall"],
    per_class["f1_score"],
    strict=True,
):
    print(f"{cls_name:<15} | {prec:>9.4f} | {rec:>9.4f} | {f1:>9.4f}")
```

### 3. Reload the Best Model & Predict

`SLMBase.from_best_model(...)` scans `~/.easyslm/finetuned` for Model Cards, picks the highest-scoring checkpoint for `eval_f1_macro`, and loads it ready for inference:

```python
model = SLMBase.from_best_model(model_type="cls", metric="eval_f1_macro")

predictions = model.predict_texts(
    [
        "Tech giant unveils next-generation quantum processor for cloud AI workloads.",
        "Central bank holds interest rates steady as quarterly corporate earnings beat forecasts.",
        "National football squad secures dramatic overtime victory in the championship final.",
    ]
)

for item in predictions:
    print(f"Text:       {item['text']}")
    print(f"Prediction: {item['predictions']}")
    print(f"Confidence: {item['probabilities']}\n")
```

---

## Option B: CLI Equivalent

The exact same workflow takes two shell commands:

```bash
# 1. Fine-tune using built-in defaults
uv run easyslm ftune \
  --model Qwen/Qwen2.5-0.5B \
  --type cls \
  --data data/agnews_tiny \
  --epochs 2

# 2. Run batch inference with automatic best-model selection
uv run easyslm predict \
  --type cls \
  --data data/agnews_tiny/val.txt
```

Predictions are written as `.txt` and `.json` files in `~/.easyslm/predictions` (or any folder passed via `--output-dir`).
