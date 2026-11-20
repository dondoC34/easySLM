# Inference & Model Cards

Every fine-tuning run produces a **self-contained checkpoint** containing everything needed for reproducible inference—no need to remember class lists, pooling strategies, or prompt templates.

---

## Checkpoint Layout

```text
~/.easyslm/finetuned/Qwen2.5-0.5B_cls_20261120_120000/
├── adapter_config.json          # LoRA config (when use_lora=True)
├── adapter_model.safetensors    # LoRA weights
├── classifier.pt                # Trained LayerNorm + Linear head (cls mode)
├── easyslm_config.json          # Classes, mode, multi_label, pooling_strategy, max_length
├── prompt_config.yml            # Snapshot of prompts used during training
├── tokenizer.json               # Tokenizer files
├── model_card_*.md              # Markdown Model Card with YAML frontmatter
├── plots/                       # Confusion matrix, per-class metrics, cardinality PNGs
└── tables/                      # Structured JSON evaluation tables
```

---

## 1. Automatic Best-Model Discovery (`from_best_model`)

`SLMBase.from_best_model(...)` scans all Model Cards in `models_dir` (defaulting to `~/.easyslm/finetuned`), ranks checkpoints by any metric (minimizing `*_loss` / `*_error` metrics, maximizing all others), and loads the winner:

```python
from src import SLMBase

# Highest validation macro-F1 (default)
model = SLMBase.from_best_model(model_type="cls", metric="eval_f1_macro")
print("Loaded checkpoint:", model.metadata["checkpoint_dir"])

# Lowest validation loss
best_loss_model = SLMBase.from_best_model(model_type="cls", metric="eval_loss")
```

## 2. Loading a Specific Checkpoint (`from_checkpoint`)

```python
from src import SLMBase

model = SLMBase.from_checkpoint(
    checkpoint_dir="~/.easyslm/finetuned/Qwen2.5-0.5B_cls_20261120_120000"
)
```

---

## 3. Running Predictions

`SLMBase` provides three inference methods sharing the exact same prompt and tokenization logic as training:

=== "Raw Strings (`predict_texts`)"

    ```python
    results = model.predict_texts(
        [
            "Quarterly earnings surged after record cloud revenue.",
            "Olympic swimming finals set two new world records.",
        ],
        batch_size=8,
    )
    for item in results:
        print(item["predictions"], item.get("probabilities"))
    ```

=== "Formatted Lines (`predict_lines`)"

    Handles both plain lines and annotated `<text> ||| <labels>` lines:

    ```python
    records = model.predict_lines(
        [
            "Central bank holds rates steady. ||| business",
            "New spacecraft launches toward Mars.",
        ]
    )
    ```

=== "Files or Directories (`predict_file_or_dir`)"

    Writes `<stem>_predictions_<mode>.txt` and `<stem>_predictions_<mode>.json`:

    ```python
    saved_files = model.predict_file_or_dir(
        data_path="data/agnews_tiny/val.txt",
        output_dir="./my_predictions",
        batch_size=8,
    )
    print("Generated files:", saved_files)
    ```
