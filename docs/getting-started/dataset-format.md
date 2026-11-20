# Dataset Format

easySLM uses a lightweight, human-readable `.txt` format with `|||` separating the input text from its comma-separated labels.

---

## Training & Validation Format

Each non-empty line must follow:

```text
<text> ||| <label_1>, <label_2>, ...
```

### Example (Single-Label)

```text
The central bank cut benchmark interest rates by 25 basis points. ||| business
Tech giants unveiled quantum computing chips for artificial intelligence. ||| sci_tech
National football squad secures overtime victory in the championship final. ||| sports
```

### Example (Multi-Label)

```text
New offshore wind farms will supply clean power to three coastal cities. ||| energy, environment
Central bank unveils green bond framework for renewable infrastructure. ||| economy, energy, environment
```

---

## Automatic Task Detection

`SLMDatasetBase` inspects all labels in the dataset and configures the task automatically:

- **Single-Label (`multi_label=False`)**: Every sample has exactly **1** label $\rightarrow$ uses `CrossEntropyLoss` and Softmax probabilities.
- **Multi-Label (`multi_label=True`)**: At least one sample has **2 or more** labels $\rightarrow$ uses `BCEWithLogitsLoss` and Sigmoid probabilities (with top-1 fallback if no class exceeds threshold).

---

## Flexible Input Paths

When passing `train_data_path` (Python API) or `--data` (CLI):

1. **Directory path** (e.g., `data/agnews_tiny`):
   - Automatically loads `train.txt` for training and `val.txt` (or `validation.txt`) for validation if present.
2. **Single `.txt` file**:
   - Automatically splits into training and validation subsets using `FineTuningConfig.val_split_ratio` (default `0.15`) and `seed` (default `42`).

---

## Inference Format

For `easyslm predict` and `model.predict_file_or_dir(...)`, input `.txt` files support both:

- **Unlabeled lines**: Plain text without `|||` (classified directly).
- **Labeled lines with `|||`**: Labels are stripped before inference and preserved under `"ground_truth"` in the generated `.json` output for side-by-side comparison.
