"""Unit tests for src/data/* modules."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.base_dataset import SLMDatasetBase
from src.data.cls_dataset import SLMClsDataset
from src.data.collator import SLMCollator
from src.data.json_dataset import SLMJsonDataset
from tests.conftest import FakeTokenizer


class TestSLMCollator:
    def test_invalid_padding_side(self) -> None:
        with pytest.raises(ValueError, match="padding_side"):
            SLMCollator(pad_token_id=0, padding_side="center")  # type: ignore[arg-type]

    def test_empty_batch(self) -> None:
        collator = SLMCollator(pad_token_id=0)
        assert collator([]) == {}

    def test_right_and_left_padding_with_and_without_pad_labels(self) -> None:
        batch = [
            {
                "input_ids": torch.tensor([10, 11], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, 11], dtype=torch.long),
            },
            {
                "input_ids": torch.tensor([20, 21, 22, 23], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([-100, -100, 22, 23], dtype=torch.long),
            },
        ]

        right_collator = SLMCollator(pad_token_id=0, pad_labels=True, padding_side="right")
        out_r = right_collator(batch)
        assert out_r["input_ids"].shape == (2, 4)
        assert out_r["input_ids"][0].tolist() == [10, 11, 0, 0]
        assert out_r["attention_mask"][0].tolist() == [1, 1, 0, 0]
        assert out_r["labels"][0].tolist() == [-100, 11, -100, -100]

        left_collator = SLMCollator(
            pad_token_id=0, pad_labels=True, padding_side="left", device="cpu"
        )
        out_l = left_collator(batch)
        assert out_l["input_ids"][0].tolist() == [0, 0, 10, 11]
        assert out_l["attention_mask"][0].tolist() == [0, 0, 1, 1]
        assert out_l["labels"][0].tolist() == [-100, -100, -100, 11]

    def test_cls_labels_stacking(self) -> None:
        batch = [
            {
                "input_ids": torch.tensor([1, 2], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                "labels": torch.tensor([1.0, 0.0], dtype=torch.float32),
            },
            {
                "input_ids": torch.tensor([3, 4, 5], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
                "labels": torch.tensor([0.0, 1.0], dtype=torch.float32),
            },
        ]
        collator = SLMCollator(pad_token_id=0, pad_labels=False)
        out = collator(batch)
        assert out["labels"].shape == (2, 2)
        assert out["labels"].dtype == torch.float32


class TestSLMDatasets:
    def test_init_validation_errors(self, fake_tokenizer: FakeTokenizer, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="tokenizer"):
            SLMClsDataset(data_path=tmp_path / "x.txt", tokenizer=None)

        with pytest.raises(ValueError, match="Either `data_path` or `texts`"):
            SLMClsDataset(tokenizer=fake_tokenizer)

        with pytest.raises(FileNotFoundError):
            SLMClsDataset(data_path=tmp_path / "missing.txt", tokenizer=fake_tokenizer)

        bad_ext = tmp_path / "data.csv"
        bad_ext.write_text("a ||| b\n", encoding="utf-8")
        with pytest.raises(ValueError, match=".txt"):
            SLMClsDataset(data_path=bad_ext, tokenizer=fake_tokenizer)

        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No '.txt' files"):
            SLMClsDataset(data_path=empty_dir, tokenizer=fake_tokenizer)

    def test_tokenizer_pad_token_auto_setup(self, single_label_file: Path) -> None:
        tok_eos = FakeTokenizer(
            pad_token=None, pad_token_id=None, eos_token="<eos>", eos_token_id=2
        )
        SLMClsDataset(data_path=single_label_file, tokenizer=tok_eos)
        assert tok_eos.pad_token == "<eos>"

        tok_no_eos = FakeTokenizer(
            pad_token=None, pad_token_id=None, eos_token=None, eos_token_id=None
        )
        SLMClsDataset(data_path=single_label_file, tokenizer=tok_no_eos)
        assert tok_no_eos.pad_token == "[PAD]"

    def test_cls_dataset_single_and_multi_label(
        self,
        fake_tokenizer: FakeTokenizer,
        single_label_file: Path,
        multi_label_file: Path,
    ) -> None:
        ds_single = SLMClsDataset(
            data_path=single_label_file,
            tokenizer=fake_tokenizer,
            max_length=64,
        )
        assert ds_single.multi_label is False
        assert ds_single.classes == ["economy", "sports", "tech"]
        assert len(ds_single) == 6
        item = ds_single[0]
        assert item["labels"].sum().item() == 1.0
        assert ds_single.get_collator().padding_side == "right"

        ds_multi = SLMClsDataset(
            data_path=multi_label_file,
            tokenizer=fake_tokenizer,
        )
        assert ds_multi.multi_label is True
        assert ds_multi.classes == ["economy", "politics", "sports", "tech"]
        assert ds_multi[0]["labels"].sum().item() == 2.0

    def test_json_dataset_with_and_without_targets(
        self,
        fake_tokenizer: FakeTokenizer,
        multi_label_file: Path,
    ) -> None:
        ds_train = SLMJsonDataset(
            data_path=multi_label_file,
            tokenizer=fake_tokenizer,
            single_quotes=True,
            max_length=128,
        )
        assert ds_train.format_labels_as_string(["tech", "economy"]) == "['economy', 'tech']"
        sample = ds_train[0]
        assert "labels" in sample
        assert (sample["labels"] == -100).any()
        assert sample["input_ids"][-1].item() == fake_tokenizer.eos_token_id
        assert ds_train.get_collator().padding_side == "right"

        ds_double_quotes = SLMJsonDataset(
            texts=["Hello world"],
            tokenizer=fake_tokenizer,
            classes=["economy", "tech"],
            include_targets=False,
            single_quotes=False,
            max_length=32,
        )
        assert ds_double_quotes.format_labels_as_string(["tech"]) == '["tech"]'
        assert "labels" not in ds_double_quotes[0]
        assert ds_double_quotes.get_collator().padding_side == "left"

    def test_directory_loading_and_malformed_lines(
        self,
        fake_tokenizer: FakeTokenizer,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "dir_ds"
        data_dir.mkdir()
        (data_dir / "part1.txt").write_text("First text ||| a\n\n", encoding="utf-8")
        (data_dir / "part2.txt").write_text("Second text ||| b\n", encoding="utf-8")
        ds = SLMClsDataset(data_path=data_dir, tokenizer=fake_tokenizer)
        assert len(ds) == 2

        no_sep = tmp_path / "no_sep.txt"
        no_sep.write_text("Missing separator\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing separator"):
            SLMClsDataset(data_path=no_sep, tokenizer=fake_tokenizer, classes=["a"])

        # Without targets, missing separator is allowed
        ds_inf = SLMClsDataset(
            data_path=no_sep, tokenizer=fake_tokenizer, classes=["a"], include_targets=False
        )
        assert len(ds_inf) == 1

        empty_txt = tmp_path / "empty_text.txt"
        empty_txt.write_text("   ||| a\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Empty text"):
            SLMClsDataset(data_path=empty_txt, tokenizer=fake_tokenizer)

        no_lbl = tmp_path / "no_lbl.txt"
        no_lbl.write_text("Some text |||   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="No labels found"):
            SLMClsDataset(data_path=no_lbl, tokenizer=fake_tokenizer)

        unknown_lbl = tmp_path / "unknown_lbl.txt"
        unknown_lbl.write_text("Some text ||| unknown\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown label"):
            SLMClsDataset(data_path=unknown_lbl, tokenizer=fake_tokenizer, classes=["a"])

        multi_in_single = tmp_path / "multi_in_single.txt"
        multi_in_single.write_text("Some text ||| a, b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Single-label dataset expects 1 label"):
            SLMClsDataset(data_path=multi_in_single, tokenizer=fake_tokenizer, multi_label=False)

    def test_split_and_oversample(
        self,
        fake_tokenizer: FakeTokenizer,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "imbalanced.txt"
        lines = [f"Majority text {i} ||| maj" for i in range(6)] + ["Minority text ||| min"]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

        ds = SLMClsDataset(data_path=p, tokenizer=fake_tokenizer)
        with pytest.raises(ValueError, match="val_ratio"):
            ds.split(val_ratio=1.5)

        train_ds, val_ds = ds.split(val_ratio=0.3, seed=42)
        assert len(train_ds) + len(val_ds) == len(ds)

        with pytest.raises(ValueError, match="Unsupported oversample strategy"):
            ds.oversample(strategy="invalid")  # type: ignore[arg-type]

        ds.oversample(strategy="max", max_factor=3.0, seed=42)
        min_count = sum(1 for _, lbls in ds.raw_samples if "min" in lbls)
        assert min_count == 3

        ds_sqrt = SLMClsDataset(data_path=p, tokenizer=fake_tokenizer)
        ds_sqrt.oversample(strategy="sqrt", seed=42)
        assert len(ds_sqrt) > 7

    def test_parse_raw_line(self) -> None:
        assert SLMDatasetBase.parse_raw_line("   ") is None
        assert SLMDatasetBase.parse_raw_line("hello ||| a, b") == ("hello", ["a", "b"])
        assert SLMDatasetBase.parse_raw_line("hello world") == ("hello world", None)
