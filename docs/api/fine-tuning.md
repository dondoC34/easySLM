# Fine-Tuning

Pydantic configuration schema, Hugging Face `Trainer` orchestrator, LoRA injection, early stopping callback, and evaluation metrics.

---

## `FineTuningConfig`

::: src.fine_tuning.config.FineTuningConfig

---

## `SLMFineTuner`

::: src.fine_tuning.trainer.SLMFineTuner

---

## `EarlyStoppingWindowCallback`

::: src.fine_tuning.callbacks.EarlyStoppingWindowCallback

---

## LoRA & GPU Memory Utilities

::: src.fine_tuning.lora.apply_lora_to_model

::: src.fine_tuning.lora.get_parameter_counts

::: src.fine_tuning.lora.log_gpu_memory

---

## Evaluation Metrics

::: src.fine_tuning.evaluation.compute_classification_metrics

::: src.fine_tuning.evaluation.build_cls_compute_metrics

::: src.fine_tuning.evaluation.build_json_compute_metrics

::: src.fine_tuning.evaluation.extract_validation_prediction_details
