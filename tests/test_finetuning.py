"""Unit tests for src/fine_tuning/* modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import Subset
from transformers import EvalPrediction, TrainerControl, TrainerState

from src.data.cls_dataset import SLMClsDataset
from src.data.json_dataset import SLMJsonDataset
from src.fine_tuning.callbacks import EarlyStoppingWindowCallback
from src.fine_tuning.config import FineTuningConfig
from src.fine_tuning.evaluation import (
    build_cls_compute_metrics,
    build_json_compute_metrics,
    compute_classification_metrics,
    extract_cls_binary_arrays,
    extract_json_binary_arrays,
    extract_validation_prediction_details,
    preprocess_logits_for_json_metrics,
)
from src.fine_tuning.lora import apply_lora_to_model, get_parameter_counts, log_gpu_memory
from src.fine_tuning.trainer import SLMFineTuner
from src.slm.cls_slm import SLMCls
from src.slm.json_slm import SLMJson
from tests.conftest import FakeTokenizer, TinyBackbone, TinyCausalLM


class TestFineTuningConfig:
    def test_from_yaml_and_overrides(self, tmp_path: Path) -> None:
        yml = tmp_path / "cfg.yml"
        yml.write_text("mode: json\nnum_train_epochs: 5\nunknown_key: 123\n", encoding="utf-8")
        cfg = FineTuningConfig.from_yaml(yml, num_train_epochs=2)
        assert cfg.mode == "json"
        assert cfg.num_train_epochs == 2

    def test_to_training_arguments_cpu(self, tmp_path: Path) -> None:
        cfg = FineTuningConfig(
            output_dir=str(tmp_path / "out"),
            use_lora=False,
            use_mlflow=False,
            num_train_epochs=2.0,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
        )
        with patch("torch.cuda.is_available", return_value=False):
            args = cfg.to_training_arguments(num_train_samples=10, has_eval_dataset=False)
        assert args.learning_rate == pytest.approx(2e-5)
        assert args.bf16 is False
        assert args.fp16 is False
        assert args.eval_strategy == "no"
        assert args.report_to == []


class TestLoraAndMemory:
    def test_apply_lora_and_counts(
        self,
        tiny_backbone: TinyBackbone,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        model = SLMCls(backbone=tiny_backbone, classes=["a", "b"], tokenizer=fake_tokenizer)
        cfg = FineTuningConfig(lora_r=4, lora_alpha=8, lora_target_modules=["proj"])

        with patch("src.fine_tuning.lora.get_peft_model", return_value=tiny_backbone) as mock_peft:
            wrapped = apply_lora_to_model(model, cfg)
        mock_peft.assert_called_once()
        assert all(p.requires_grad for p in wrapped.norm.parameters())
        assert all(p.requires_grad for p in wrapped.classifier.parameters())

        trainable, total = get_parameter_counts(wrapped)
        assert total > 0
        assert trainable <= total

    def test_log_gpu_memory(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.memory_allocated", return_value=1024**2),
            patch("torch.cuda.memory_reserved", return_value=2 * 1024**2),
            patch("torch.cuda.max_memory_allocated", return_value=3 * 1024**2),
        ):
            log_gpu_memory("test")


class TestEarlyStoppingWindowCallback:
    def test_invalid_window_steps(self) -> None:
        with pytest.raises(ValueError, match="window_steps"):
            EarlyStoppingWindowCallback(window_steps=0)

    def test_improvement_stop_and_restore(self, tiny_backbone: TinyBackbone) -> None:
        cb = EarlyStoppingWindowCallback(window_steps=2, metric_name="loss")
        assert cb.metric_name == "eval_loss"
        assert cb.greater_is_better is False

        args = MagicMock()
        state = TrainerState(global_step=1)
        control = TrainerControl()

        # Missing metric -> no crash, no stop
        cb.on_evaluate(args, state, control, metrics={}, model=tiny_backbone)
        assert control.should_training_stop is False

        # Step 1: improvement (0.5)
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.5}, model=tiny_backbone)
        assert cb.best_metric == pytest.approx(0.5)
        assert cb.best_step == 1

        # Mutate weights, then worsen twice -> should stop
        with torch.no_grad():
            for p in tiny_backbone.parameters():
                p.add_(10.0)

        state.global_step = 2
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.6}, model=tiny_backbone)
        assert control.should_training_stop is False

        state.global_step = 3
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.7}, model=tiny_backbone)
        assert control.should_training_stop is True

        # Restore best weights on train end
        before_restore = next(tiny_backbone.parameters()).clone()
        cb.on_train_end(args, state, control, model=tiny_backbone)
        after_restore = next(tiny_backbone.parameters())
        assert not torch.allclose(before_restore, after_restore)

        # Further on_evaluate after train_end is ignored
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.1}, model=tiny_backbone)
        assert cb.best_metric == pytest.approx(0.5)


class TestEvaluation:
    def test_compute_classification_metrics_single_and_multi(self) -> None:
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        m_single = compute_classification_metrics(y_true, y_pred, multi_label=False)
        assert m_single["accuracy"] == pytest.approx(2 / 3)
        assert "balanced_accuracy" in m_single
        assert "matthews_corrcoef" in m_single
        assert "cohen_kappa" in m_single

        y_true_m = np.array([[1, 1, 0], [0, 1, 1]])
        y_pred_m = np.array([[1, 0, 0], [0, 1, 1]])
        m_multi = compute_classification_metrics(y_true_m, y_pred_m, multi_label=True)
        assert m_multi["exact_match_accuracy"] == pytest.approx(0.5)

    def test_cls_extract_and_compute_metrics(self) -> None:
        logits = np.array([[3.0, -3.0], [-3.0, 3.0]])
        labels = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_true, y_pred = extract_cls_binary_arrays((logits,), labels, multi_label=True)
        assert np.array_equal(y_true, y_pred)

        # Single-label with integer class indices
        y_true_s, y_pred_s = extract_cls_binary_arrays(logits, np.array([0, 1]), multi_label=False)
        assert np.array_equal(y_true_s, y_pred_s)

        fn = build_cls_compute_metrics(multi_label=False)
        res = fn(EvalPrediction(predictions=logits, label_ids=np.array([[1, 0], [0, 1]])))
        assert res["accuracy"] == pytest.approx(1.0)

    def test_cls_extract_threshold_boundary_inclusive(self) -> None:
        logits = np.array([[0.0, -3.0]])
        labels = np.array([[1.0, 0.0]])
        _, y_pred = extract_cls_binary_arrays(logits, labels, multi_label=True, threshold=0.5)
        assert y_pred[0, 0] == 1
        assert y_pred[0, 1] == 0

    def test_json_preprocess_and_compute_metrics(self, fake_tokenizer: FakeTokenizer) -> None:
        raw_logits = torch.randn(2, 5, 10)
        reduced = preprocess_logits_for_json_metrics((raw_logits,), torch.zeros(2, 5))
        assert reduced.shape == (2, 5)

        ids_a = fake_tokenizer.encode("['economy']", add_special_tokens=False)
        ids_b = fake_tokenizer.encode("['tech']", add_special_tokens=False)
        seq_len = max(len(ids_a), len(ids_b)) + 2

        pred_ids = np.zeros((2, seq_len), dtype=np.int64)
        label_ids = np.full((2, seq_len), -100, dtype=np.int64)
        # Shift alignment: pred_ids[:, :-1] aligns with label_ids[:, 1:]
        for i, seq in enumerate([ids_a, ids_b]):
            for j, tok in enumerate(seq):
                pred_ids[i, j] = tok
                label_ids[i, j + 1] = tok

        y_true, y_pred, correct, total, valid_json = extract_json_binary_arrays(
            pred_ids, label_ids, fake_tokenizer, ["economy", "tech"]
        )
        assert correct == total
        assert valid_json == 2
        assert np.array_equal(y_true, y_pred)

        fn = build_json_compute_metrics(fake_tokenizer, ["economy", "tech"], multi_label=False)
        metrics = fn(EvalPrediction(predictions=pred_ids, label_ids=label_ids))
        assert metrics["token_accuracy"] == pytest.approx(1.0)
        assert metrics["json_valid_parse_rate"] == pytest.approx(1.0)

    def test_extract_validation_prediction_details(
        self,
        fake_tokenizer: FakeTokenizer,
        single_label_file: Path,
    ) -> None:
        ds = SLMClsDataset(data_path=single_label_file, tokenizer=fake_tokenizer)
        subset = Subset(ds, [0, 1])
        logits = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0]])
        labels = np.array([[1, 0, 0], [0, 1, 0]])
        pred_out = SimpleNamespace(predictions=logits, label_ids=labels)

        y_true, y_pred, texts, raw, parsed, scores = extract_validation_prediction_details(
            pred_out, subset, mode="cls", classes=ds.classes, multi_label=False
        )
        assert len(texts) == 2
        assert len(raw) == 2
        assert parsed == ["economy", "sports"]
        assert scores is not None and len(scores) == 2

        # Also test JSON branch with non-SLMDatasetBase
        ds_json = SLMJsonDataset(data_path=single_label_file, tokenizer=fake_tokenizer)
        toks = fake_tokenizer.encode("['economy']", add_special_tokens=False)
        pred_ids = np.zeros((1, len(toks) + 2), dtype=np.int64)
        label_ids = np.full((1, len(toks) + 2), -100, dtype=np.int64)
        for j, t in enumerate(toks):
            pred_ids[0, j] = t
            label_ids[0, j + 1] = t
        y_t, y_p, t_list, _, parsed_j, scores_j = extract_validation_prediction_details(
            SimpleNamespace(predictions=pred_ids, label_ids=label_ids),
            Subset(ds_json, [0]),
            mode="json",
            classes=ds_json.classes,
            multi_label=False,
            tokenizer=fake_tokenizer,
        )
        assert parsed_j == ["economy"]
        assert scores_j is None


class TestSLMFineTuner:
    def test_prepare_datasets_dir_and_oversample(
        self,
        tmp_path: Path,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        data_dir = tmp_path / "data_dir"
        data_dir.mkdir()
        (data_dir / "train.txt").write_text(
            "\n".join([f"Text {i} ||| a" for i in range(5)] + ["Minor ||| b"]),
            encoding="utf-8",
        )
        (data_dir / "val.txt").write_text("Val 1 ||| a\nVal 2 ||| b\n", encoding="utf-8")

        mock_factory = MagicMock()
        mock_factory.download_model.return_value = tmp_path
        mock_factory._prepare_tokenizer.return_value = fake_tokenizer

        cfg = FineTuningConfig(
            mode="cls",
            use_mlflow=False,
            oversample_strategy="max",
            oversample_max_factor=2.0,
        )
        tuner = SLMFineTuner(model_name="tiny", config=cfg, factory=mock_factory)
        train_ds, val_ds, _ = tuner.prepare_datasets(data_dir)
        assert val_ds is not None
        assert len(val_ds) == 2
        assert len(train_ds) > 6

    def test_prepare_datasets_takes_max_length_across_train_and_val(
        self,
        tmp_path: Path,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        data_dir = tmp_path / "data_len_dir"
        data_dir.mkdir()
        (data_dir / "train.txt").write_text("hi ||| a\n", encoding="utf-8")
        (data_dir / "val.txt").write_text(
            "this is a much longer validation sample with many words ||| a\n",
            encoding="utf-8",
        )

        mock_factory = MagicMock()
        mock_factory.download_model.return_value = tmp_path
        mock_factory._prepare_tokenizer.return_value = fake_tokenizer

        tuner = SLMFineTuner(
            model_name="tiny",
            config=FineTuningConfig(mode="cls", use_mlflow=False),
            factory=mock_factory,
        )
        train_ds, val_ds, _ = tuner.prepare_datasets(data_dir)
        assert val_ds is not None
        assert val_ds.max_length > train_ds.max_length
        assert tuner.max_length == val_ds.max_length

    def test_train_pipeline_cls_mocked_trainer(
        self,
        tmp_path: Path,
        single_label_file: Path,
        fake_tokenizer: FakeTokenizer,
        tiny_backbone: TinyBackbone,
    ) -> None:
        mock_factory = MagicMock()
        mock_factory.download_model.return_value = tmp_path
        mock_factory._prepare_tokenizer.return_value = fake_tokenizer
        mock_factory.get_backbone.return_value = (tiny_backbone, fake_tokenizer)

        cfg = FineTuningConfig(
            output_dir=str(tmp_path / "models"),
            mode="cls",
            use_lora=False,
            use_mlflow=False,
            gradient_checkpointing=True,
            early_stopping_window=2,
            num_train_epochs=1.0,
        )
        tuner = SLMFineTuner(model_name="tiny-bb", config=cfg, factory=mock_factory)

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = SimpleNamespace(metrics={"train_loss": 0.4})
        mock_trainer_instance.predict.return_value = SimpleNamespace(
            predictions=np.array([[2.0, -1.0, -1.0]]),
            label_ids=np.array([[1, 0, 0]]),
            metrics={"eval_loss": 0.3, "eval_f1_macro": 0.9},
        )
        mock_trainer_instance.eval_dataset = None

        with (
            patch("src.fine_tuning.trainer.Trainer", return_value=mock_trainer_instance),
            patch("src.fine_tuning.trainer.log_evaluation_artifacts"),
        ):
            res = tuner.train(single_label_file)

        assert res["train_metrics"]["train_loss"] == pytest.approx(0.4)
        assert res["eval_metrics"]["eval_f1_macro"] == pytest.approx(0.9)
        assert Path(res["checkpoint_dir"]).exists()
        assert (Path(res["checkpoint_dir"]) / "easyslm_config.json").exists()

    def test_evaluate_without_trainer_raises(self) -> None:
        cfg = FineTuningConfig(use_mlflow=False)
        tuner = SLMFineTuner(model_name="tiny", config=cfg)
        with pytest.raises(RuntimeError, match="Trainer has not been initialized"):
            tuner.evaluate()

    def test_build_model_json_and_invalid_mode(
        self,
        tiny_causal_lm: TinyCausalLM,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        mock_factory = MagicMock()
        mock_factory.get_causal_lm.return_value = (tiny_causal_lm, fake_tokenizer)
        cfg = FineTuningConfig(mode="json", use_lora=False, use_mlflow=False)
        tuner = SLMFineTuner(model_name="tiny-clm", config=cfg, factory=mock_factory)
        model = tuner.build_model(["a", "b"])
        assert isinstance(model, SLMJson)

        tuner.config.mode = "invalid"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unsupported mode"):
            tuner.build_model(["a"])
