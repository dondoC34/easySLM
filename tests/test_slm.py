"""Unit tests for src/slm/* modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.slm.base_slm import SLMBase
from src.slm.cls_slm import SLMCls
from src.slm.json_slm import SLMJson
from src.slm.serialization import load_finetuned_model, save_finetuned_model
from src.slm.slm_factory import ModelFactory
from tests.conftest import FakeTokenizer, TinyBackbone, TinyCausalLM


class TestModelFactory:
    def test_get_local_path_and_cache(self, tmp_path: Path) -> None:
        factory = ModelFactory(cache_dir=tmp_path, device="cpu", torch_dtype=torch.float32)
        assert factory.get_local_path("Qwen/Qwen2.5-0.5B") == tmp_path / "Qwen--Qwen2.5-0.5B"
        assert not factory.is_cached_locally("Qwen/Qwen2.5-0.5B")

        direct_dir = tmp_path / "local_model"
        direct_dir.mkdir()
        (direct_dir / "config.json").write_text("{}", encoding="utf-8")
        assert factory.get_local_path(str(direct_dir)) == direct_dir
        assert factory.is_cached_locally(str(direct_dir))

    def test_download_model_cached_and_hub(self, tmp_path: Path) -> None:
        factory = ModelFactory(cache_dir=tmp_path, device="cpu", torch_dtype=torch.float32)
        cached = tmp_path / "org--model"
        cached.mkdir()
        (cached / "config.json").write_text("{}", encoding="utf-8")

        with patch("src.slm.slm_factory.snapshot_download") as mock_dl:
            assert factory.download_model("org/model") == cached
            mock_dl.assert_not_called()

            factory.download_model("org/other")
            mock_dl.assert_called_once()

    def test_get_backbone_causal_lm_and_seq_cls(
        self,
        tmp_path: Path,
        fake_tokenizer: FakeTokenizer,
        tiny_backbone: TinyBackbone,
        tiny_causal_lm: TinyCausalLM,
    ) -> None:
        factory = ModelFactory(cache_dir=tmp_path, device="cpu", torch_dtype=torch.float32)
        local_dir = tmp_path / "org--model"
        local_dir.mkdir()
        (local_dir / "config.json").write_text("{}", encoding="utf-8")

        fake_tokenizer.pad_token = None
        fake_tokenizer.pad_token_id = None
        tiny_backbone.config.pad_token_id = None
        tiny_causal_lm.config.pad_token_id = None

        with (
            patch("src.slm.slm_factory.AutoTokenizer.from_pretrained", return_value=fake_tokenizer),
            patch("src.slm.slm_factory.AutoModel.from_pretrained", return_value=tiny_backbone),
            patch(
                "src.slm.slm_factory.AutoModelForCausalLM.from_pretrained",
                return_value=tiny_causal_lm,
            ),
            patch(
                "src.slm.slm_factory.AutoModelForSequenceClassification.from_pretrained",
                return_value=tiny_backbone,
            ),
        ):
            bb, tok = factory.get_backbone("org/model")
            assert bb is tiny_backbone
            assert tok.pad_token == "<eos>"

            clm, _ = factory.get_causal_lm("org/model")
            assert clm is tiny_causal_lm

            seq, _ = factory.get_sequence_classifier(
                "org/model",
                num_labels=3,
                id2label={0: "a"},
                label2id={"a": 0},
                problem_type="multi_label_classification",
            )
            assert seq is tiny_backbone


class TestSLMCls:
    @pytest.mark.parametrize("pooling", ["eos", "mean", "max"])
    def test_forward_and_predict_multilabel_and_single(
        self,
        pooling: str,
        tiny_backbone: TinyBackbone,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        model = SLMCls(
            backbone=tiny_backbone,
            classes=["sports", "economy", "tech"],
            multi_label=True,
            pooling_strategy=pooling,  # type: ignore[arg-type]
            tokenizer=fake_tokenizer,
        )
        assert model.mode == "cls"
        assert model.classes == ["economy", "sports", "tech"]
        assert model.device == torch.device("cpu")
        assert model.dtype == torch.float32

        input_ids = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
        labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

        out = model(input_ids, attention_mask, labels=labels, return_pooled=True)
        assert out["loss"] is not None
        assert out["logits"].shape == (2, 3)
        assert out["pooled_states"].shape == (2, 16)

        pred_multi = model.predict(input_ids, attention_mask, threshold=0.99, fallback_top1=True)
        assert len(pred_multi["predictions"]) == 2
        assert all(len(p) >= 1 for p in pred_multi["predictions"])

        # Exact threshold boundary: probability == threshold must be included (>=)
        exact_prob = float(pred_multi["probabilities"][0, 0].item())
        pred_boundary = model.predict(
            input_ids[:1], attention_mask[:1], threshold=exact_prob, fallback_top1=False
        )
        assert pred_boundary["binary_vectors"][0, 0].item() == 1
        assert model.idx_to_class[0] in pred_boundary["predictions"][0]

        # Single-label mode
        model_single = SLMCls(
            backbone=tiny_backbone,
            classes=["economy", "sports", "tech"],
            multi_label=False,
            pooling_strategy=pooling,  # type: ignore[arg-type]
            tokenizer=fake_tokenizer,
        )
        out_single = model_single(input_ids, attention_mask, labels=labels)
        assert out_single["loss"] is not None
        pred_single = model_single.predict(input_ids, attention_mask)
        assert all(len(p) == 1 for p in pred_single["predictions"])

    def test_eos_pooling_ignores_right_padding(
        self,
        tiny_backbone: TinyBackbone,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        model = SLMCls(
            backbone=tiny_backbone,
            classes=["a", "b"],
            pooling_strategy="eos",
            tokenizer=fake_tokenizer,
        )
        # (batch=2, seq_len=4, hidden_size=16) with distinct values per position
        hidden_states = torch.arange(2 * 4 * 16, dtype=torch.float32).reshape(2, 4, 16)
        attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.long)

        pooled = model.pool_hidden_states(hidden_states, attention_mask)
        assert torch.allclose(pooled[0], hidden_states[0, 1])
        assert torch.allclose(pooled[1], hidden_states[1, 2])

    def test_classes_are_deduplicated(
        self,
        tiny_backbone: TinyBackbone,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        model = SLMCls(
            backbone=tiny_backbone,
            classes=["sports", "tech", "sports", " tech "],
            tokenizer=fake_tokenizer,
        )
        assert model.classes == ["sports", "tech"]
        assert model.num_classes == 2

    def test_predict_texts_truncates_to_backbone_max_position_embeddings(
        self,
        tiny_backbone: TinyBackbone,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        tiny_backbone.config.max_position_embeddings = 55
        model = SLMCls(
            backbone=tiny_backbone,
            classes=["sports", "tech"],
            tokenizer=fake_tokenizer,
        )
        captured_lengths: list[int] = []
        orig_predict = model.predict

        def spy_predict(
            input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: object
        ) -> dict[str, object]:
            captured_lengths.append(int(input_ids.shape[1]))
            return orig_predict(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        model.predict = spy_predict  # type: ignore[method-assign]
        long_text = " ".join(f"tok{i}" for i in range(100))
        out = model.predict_texts([long_text])
        assert len(out) == 1
        assert len(captured_lengths) == 1
        assert captured_lengths[0] <= 55

    def test_invalid_init_and_gradient_checkpointing(
        self,
        tiny_backbone: TinyBackbone,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        with pytest.raises(ValueError, match="pooling_strategy"):
            SLMCls(backbone=tiny_backbone, classes=["a"], pooling_strategy="bad")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="classes"):
            SLMCls(backbone=tiny_backbone, classes=[])

        tiny_backbone.config.hidden_size = None
        with pytest.raises(ValueError, match="hidden_size"):
            SLMCls(backbone=tiny_backbone, classes=["a"])
        tiny_backbone.config.hidden_size = 16

        model = SLMCls(backbone=tiny_backbone, classes=["a", "b"], tokenizer=fake_tokenizer)
        model.gradient_checkpointing_enable()
        assert tiny_backbone.gc_enabled is True
        assert tiny_backbone.config.use_cache is False
        model.gradient_checkpointing_disable()
        assert tiny_backbone.gc_enabled is False
        assert tiny_backbone.config.use_cache is True

    def test_from_pretrained_and_predict_files(
        self,
        tiny_backbone: TinyBackbone,
        fake_tokenizer: FakeTokenizer,
        tmp_path: Path,
    ) -> None:
        mock_factory = MagicMock(spec=ModelFactory)
        mock_factory.get_backbone.return_value = (tiny_backbone, fake_tokenizer)

        model = SLMCls.from_pretrained(
            "tiny-model",
            classes=["economy", "sports"],
            multi_label=False,
            factory=mock_factory,
        )
        assert model.metadata["base_model_name"] == "tiny-model"

        data_file = tmp_path / "infer.txt"
        data_file.write_text("Market rises ||| economy\nGoal scored\n", encoding="utf-8")
        out_dir = tmp_path / "preds"
        saved = model.predict_file_or_dir(data_file, out_dir, batch_size=1)
        assert len(saved) == 2
        assert (out_dir / "infer_predictions_cls.txt").exists()
        assert (out_dir / "infer_predictions_cls.json").exists()

        assert model.predict_texts([]) == []


class TestSLMJson:
    def test_forward_parse_and_predict(
        self,
        tiny_causal_lm: TinyCausalLM,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        # Pre-seed tokenizer with JSON-like tokens so generate() decodes a valid list
        ids = fake_tokenizer.encode("['economy', 'tech']", add_special_tokens=False)
        tiny_causal_lm._generated_suffix = ids

        model = SLMJson(
            backbone=tiny_causal_lm,
            classes=["economy", "sports", "tech"],
            multi_label=True,
            tokenizer=fake_tokenizer,
        )
        assert model.mode == "json"

        input_ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        labels = torch.tensor([[-100, -100, 5, 6]], dtype=torch.long)

        out = model(input_ids, attention_mask, labels=labels)
        assert out["loss"] is not None
        assert out["logits"].shape[:2] == (1, 4)

        # All-masked labels returns zero loss safely
        all_masked = torch.full_like(labels, -100)
        out_masked = model(input_ids, attention_mask, labels=all_masked)
        assert out_masked["loss"] is not None
        assert float(out_masked["loss"].item()) == pytest.approx(0.0)

        # On-eval token reduction returns 2D token IDs instead of 3D vocab logits
        model.eval()
        model._return_token_ids_on_eval = True  # type: ignore[attr-defined]
        out_reduced = model(input_ids, attention_mask, labels=labels)
        assert out_reduced["logits"].shape == (1, 4)
        assert out_reduced["logits"].dtype == torch.int64
        model._return_token_ids_on_eval = False  # type: ignore[attr-defined]
        model.train()

        pred = model.predict(input_ids, attention_mask, max_new_tokens=len(ids))
        assert pred["predictions"] == [["economy", "tech"]]

        texts_out = model.predict_texts(["Hello world"])
        assert len(texts_out) == 1
        assert "raw_output" in texts_out[0]

    def test_parse_class_list_variants(self) -> None:
        valid = {"economy", "sports", "tech"}
        assert SLMJson.parse_class_list("['tech', 'economy', 'invalid']", valid) == (
            ["economy", "tech"],
            True,
        )
        assert SLMJson.parse_class_list('Some prefix ["sports"] trailing', valid) == (
            ["sports"],
            True,
        )
        assert SLMJson.parse_class_list("not a list", valid) == ([], False)

    def test_from_pretrained(
        self,
        tiny_causal_lm: TinyCausalLM,
        fake_tokenizer: FakeTokenizer,
    ) -> None:
        mock_factory = MagicMock(spec=ModelFactory)
        mock_factory.get_causal_lm.return_value = (tiny_causal_lm, fake_tokenizer)
        model = SLMJson.from_pretrained("tiny-causal", classes=["a", "b"], factory=mock_factory)
        assert model.mode == "json"


class TestSerialization:
    def test_save_preserves_non_default_pooling_strategy(
        self,
        tiny_backbone: TinyBackbone,
        fake_tokenizer: FakeTokenizer,
        tmp_path: Path,
    ) -> None:
        cls_model = SLMCls(
            backbone=tiny_backbone,
            classes=["a", "b"],
            pooling_strategy="mean",
            tokenizer=fake_tokenizer,
        )
        saved_cls = cls_model.save(
            output_dir=tmp_path / "cls_mean_ckpt",
            base_model_name="tiny-bb",
        )
        cfg = json.loads((saved_cls / "easyslm_config.json").read_text(encoding="utf-8"))
        assert cfg["pooling_strategy"] == "mean"

    def test_save_and_load_cls_and_json(
        self,
        tiny_backbone: TinyBackbone,
        tiny_causal_lm: TinyCausalLM,
        fake_tokenizer: FakeTokenizer,
        tmp_path: Path,
    ) -> None:
        # 1. Save SLMCls
        cls_model = SLMCls(
            backbone=tiny_backbone,
            classes=["a", "b"],
            multi_label=False,
            tokenizer=fake_tokenizer,
        )
        cls_dir = tmp_path / "exp" / "cls_ckpt"
        saved_cls = cls_model.save(
            output_dir=cls_dir,
            base_model_name="tiny-bb",
            training_params={"epochs": 1},
            metrics={"eval_f1_macro": 0.8},
            as_of="20260101_000000",
        )
        assert (saved_cls / "easyslm_config.json").exists()
        assert (saved_cls / "classifier.pt").exists()
        assert (saved_cls / "prompt_config.yml").exists()
        assert (saved_cls.parent / "model_card_tiny-bb_cls_20260101_000000.md").exists()

        mock_factory = MagicMock(spec=ModelFactory)
        mock_factory.device = torch.device("cpu")
        mock_factory.torch_dtype = torch.float32

        with (
            patch(
                "src.slm.serialization.AutoTokenizer.from_pretrained", return_value=fake_tokenizer
            ),
            patch("src.slm.serialization.AutoModel.from_pretrained", return_value=tiny_backbone),
        ):
            loaded_cls = SLMBase.from_checkpoint(saved_cls, factory=mock_factory)
            assert isinstance(loaded_cls, SLMCls)
            assert loaded_cls.classes == ["a", "b"]

            # Also test from_best_model
            best_cls = SLMBase.from_best_model(
                models_dir=tmp_path / "exp",
                model_type="cls",
                factory=mock_factory,
            )
            assert isinstance(best_cls, SLMCls)
            assert "model_card" in best_cls.metadata

        # 2. Save & Load SLMJson
        json_model = SLMJson(
            backbone=tiny_causal_lm,
            classes=["a", "b"],
            tokenizer=fake_tokenizer,
        )
        json_dir = tmp_path / "json_ckpt"
        save_finetuned_model(
            model=json_model,
            output_dir=json_dir,
            base_model_name="tiny-clm",
            as_of="20260101_000001",
        )

        with (
            patch(
                "src.slm.serialization.AutoTokenizer.from_pretrained", return_value=fake_tokenizer
            ),
            patch(
                "src.slm.serialization.AutoModelForCausalLM.from_pretrained",
                return_value=tiny_causal_lm,
            ),
        ):
            loaded_json = load_finetuned_model(json_dir, factory=mock_factory)
            assert isinstance(loaded_json, SLMJson)

    def test_serialization_errors(self, tiny_backbone: TinyBackbone, tmp_path: Path) -> None:
        cls_no_tok = SLMCls(backbone=tiny_backbone, classes=["a"])
        with pytest.raises(ValueError, match="without a `tokenizer`"):
            save_finetuned_model(cls_no_tok, tmp_path / "out", "tiny")

        with pytest.raises(FileNotFoundError):
            load_finetuned_model(tmp_path / "missing_dir")

        empty = tmp_path / "empty_ckpt"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="easyslm_config.json"):
            load_finetuned_model(empty)

        bad_mode = tmp_path / "bad_mode"
        bad_mode.mkdir()
        (bad_mode / "easyslm_config.json").write_text(
            json.dumps({"mode": "invalid", "classes": ["a"]}),
            encoding="utf-8",
        )
        mock_factory = MagicMock(spec=ModelFactory)
        mock_factory.device = torch.device("cpu")
        mock_factory.torch_dtype = torch.float32
        with patch(
            "src.slm.serialization.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ):
            with pytest.raises(ValueError, match="Unknown model mode"):
                load_finetuned_model(bad_mode, factory=mock_factory)
