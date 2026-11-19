"""Custom Hugging Face `TrainerCallback` implementations for SLM fine-tuning."""

import logging
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EarlyStoppingWindowCallback(TrainerCallback):
    """Stops training when a validation metric fails to improve for `window_steps` consecutive evaluations.

    Snapshots only trainable (`requires_grad=True`) parameters on CPU whenever a new best
    metric is observed, and restores them onto `model` at the end of training.
    """

    def __init__(self, window_steps: int = 4, metric_name: str = "eval_loss") -> None:
        if window_steps <= 0:
            raise ValueError(f"`window_steps` must be > 0, got {window_steps}.")
        self.window_steps = window_steps
        self.metric_name = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
        # Losses improve by decreasing; score metrics (accuracy, f1, etc.) improve by increasing.
        self.greater_is_better = "loss" not in self.metric_name.lower()
        self.best_metric: float = float("-inf") if self.greater_is_better else float("inf")
        self.best_step: int = 0
        self.no_improvement_evals: int = 0
        self.best_trainable_state: dict[str, torch.Tensor] | None = None
        self._training_ended: bool = False

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        if self._training_ended:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not metrics or self.metric_name not in metrics:
            logger.warning(
                "EarlyStoppingWindowCallback: metric '%s' missing from eval metrics; skipping.",
                self.metric_name,
            )
            return

        value = float(metrics[self.metric_name])
        improved = value > self.best_metric if self.greater_is_better else value < self.best_metric

        if improved:
            self.best_metric = value
            self.best_step = state.global_step
            self.no_improvement_evals = 0
            if model is not None:
                # Snapshot only trainable weights on CPU to keep memory overhead minimal (~few MB for LoRA).
                self.best_trainable_state = {
                    name: param.detach().cpu().clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
            logger.info(
                "EarlyStopping: new best %s=%.4f at step %d.",
                self.metric_name,
                self.best_metric,
                self.best_step,
            )
        else:
            self.no_improvement_evals += 1
            if self.no_improvement_evals >= self.window_steps:
                logger.info(
                    "EarlyStopping triggered at step %d: '%s' did not improve for %d consecutive "
                    "evaluations (best=%.4f at step %d).",
                    state.global_step,
                    self.metric_name,
                    self.no_improvement_evals,
                    self.best_metric,
                    self.best_step,
                )
                control.should_training_stop = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        self._training_ended = True
        if model is None or self.best_trainable_state is None:
            return

        # Restore best weights in-place so final evaluation and model.save() use the best step.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.best_trainable_state:
                    saved = self.best_trainable_state[name]
                    param.data.copy_(saved.to(device=param.device, dtype=param.dtype))

        logger.info(
            "EarlyStopping: restored best trainable weights from step %d (%s=%.4f).",
            self.best_step,
            self.metric_name,
            self.best_metric,
        )
