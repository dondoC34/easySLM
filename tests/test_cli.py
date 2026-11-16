"""Unit tests for src/cli.py and package exports."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src
from src.cli import build_parser, main, run_ftune, run_predict


class TestMainCLI:
    def test_package_exports(self) -> None:
        assert hasattr(src, "SLMCls")
        assert hasattr(src, "SLMJson")
        assert hasattr(src, "SLMFineTuner")
        assert src.__version__ == "0.1.0"

    def test_help_and_no_args_exit_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        for argv in (["easyslm"], ["easyslm", "--help"], ["easyslm", "ftune", "--help"]):
            with (
                patch.object(sys, "argv", argv),
                pytest.raises(SystemExit) as exc_info,
            ):
                main()
            assert exc_info.value.code == 0

        for argv in (["easyslm", "predict", "--help"], ["easyslm", "mlflow"]):
            with (
                patch.object(sys, "argv", argv),
                pytest.raises(SystemExit) as exc_info,
            ):
                main()
            assert exc_info.value.code == 0
        assert "easySLM" in capsys.readouterr().out

    def test_mlflow_delegation(self) -> None:
        with (
            patch.object(sys, "argv", ["easyslm", "mlflow", "status"]),
            patch("src.cli.handle_mlflow_cli", return_value=0) as mock_ml,
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0
        mock_ml.assert_called_once_with(["status"])

    def test_info_command(self) -> None:
        with (
            patch.object(sys, "argv", ["easyslm", "info"]),
            patch("src.cli.print_cuda_info") as mock_info,
        ):
            main()
        mock_info.assert_called_once()

    def test_run_ftune(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "train.yml"
        cfg_file.write_text("mode: cls\nuse_mlflow: false\n", encoding="utf-8")

        parser = build_parser()
        args = parser.parse_args(
            [
                "ftune",
                "--model",
                "tiny",
                "--data",
                "data.txt",
                "--type",
                "json",
                "--config",
                str(cfg_file),
                "--epochs",
                "2",
            ]
        )

        mock_tuner = MagicMock()
        mock_tuner.train.return_value = {"checkpoint_dir": tmp_path / "ckpt"}
        with patch("src.cli.SLMFineTuner", return_value=mock_tuner) as mock_cls:
            run_ftune(args)
        mock_cls.assert_called_once()
        mock_tuner.train.assert_called_once_with(train_data_path="data.txt")

    def test_run_predict_explicit_and_best(self, tmp_path: Path) -> None:
        parser = build_parser()
        mock_model = MagicMock()
        mock_model.metadata = {"checkpoint_dir": str(tmp_path), "model_card": {"as_of": "today"}}
        mock_model.predict_file_or_dir.return_value = [tmp_path / "out.txt"]

        args_explicit = parser.parse_args(["predict", "--model", str(tmp_path), "--data", "d.txt"])
        with patch("src.cli.SLMBase.from_checkpoint", return_value=mock_model) as mock_ckpt:
            run_predict(args_explicit)
        mock_ckpt.assert_called_once()
        mock_model.predict_file_or_dir.assert_called_once()

        args_best = parser.parse_args(["predict", "--models-dir", str(tmp_path), "--data", "d.txt"])
        with patch("src.cli.SLMBase.from_best_model", return_value=mock_model) as mock_best:
            run_predict(args_best)
        mock_best.assert_called_once()
