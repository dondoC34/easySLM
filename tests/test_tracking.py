"""Unit tests for src/tracking/* modules."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.tracking.artifacts import log_evaluation_artifacts
from src.tracking.cli import format_cli_table, handle_mlflow_cli
from src.tracking.config import (
    DEFAULT_CONFIG,
    configure_mlflow_env,
    get_mlflow_config,
    load_mlflow_config,
)
from src.tracking.manager import (
    artifact_uri_to_path,
    clear_mlflow_data,
    export_run_artifacts,
    get_best_mlflow_run,
    get_mlflow_status,
    list_mlflow_runs,
    resolve_storage_paths,
)
from src.tracking.plotting import (
    create_cross_label_confusion_matrix_figure,
    create_label_cardinality_figure,
    create_per_class_metrics_figure,
    create_single_label_confusion_matrix_figure,
)
from src.tracking.server import check_mlflow_server_online, start_mlflow_server
from src.tracking.tables import (
    create_cross_label_confusion_matrix_table,
    create_label_cardinality_table,
    create_per_class_metrics_table,
    create_single_label_confusion_matrix_table,
    create_validation_errors_table,
)


class TestTrackingConfig:
    def test_load_and_get_mlflow_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mlflow.yml"
        cfg_file.write_text(
            "server:\n  port: 5055\ntracking:\n  experiment_prefix: testExp\n",
            encoding="utf-8",
        )
        cfg = get_mlflow_config(cfg_file)
        assert cfg["server"]["port"] == 5055
        assert cfg["tracking"]["experiment_prefix"] == "testExp"

        missing = load_mlflow_config(tmp_path / "missing.yml")
        assert missing["server"]["port"] == DEFAULT_CONFIG["server"]["port"]

    def test_configure_mlflow_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        local_dir = tmp_path / "mlstore"
        cfg = {
            "server": {"backend_store_uri": str(local_dir)},
            "tracking": {
                "experiment_prefix": "demo",
                "system_metrics": True,
                "system_metrics_sampling_interval": 5,
            },
        }
        with patch("mlflow.enable_system_metrics_logging"):
            res = configure_mlflow_env(cfg)
        assert res == ["mlflow"]
        assert os.environ["MLFLOW_TRACKING_URI"].startswith("file:")
        assert os.environ["MLFLOW_EXPERIMENT_NAME"] == "demo_finetuning"


class TestTrackingServer:
    def test_check_mlflow_server_online(self) -> None:
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        with patch("socket.create_connection", return_value=mock_conn):
            assert check_mlflow_server_online("127.0.0.1", 5000) is True

        with patch("socket.create_connection", side_effect=OSError("refused")):
            assert check_mlflow_server_online("127.0.0.1", 5000) is False

    def test_start_mlflow_server(self) -> None:
        with patch("subprocess.call", return_value=0) as mock_call:
            assert start_mlflow_server(host="127.0.0.1", port=5001) == 0
            mock_call.assert_called_once()

        with patch("subprocess.call", side_effect=KeyboardInterrupt):
            assert start_mlflow_server() == 0


class TestTrackingTablesAndPlotting:
    def test_single_label_confusion_matrix_normalizes_by_true_row(self) -> None:
        # True: 1 economy, 1 tech. Both predicted as economy -> row-normalized recall is 1.0 for each.
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [1, 0]])
        t = create_single_label_confusion_matrix_table(y_true, y_pred, ["economy", "tech"])
        # Order: (economy, economy)=1.0, (economy, tech)=0.0, (tech, economy)=1.0, (tech, tech)=0.0
        assert t["normalized_rate"] == [1.0, 0.0, 1.0, 0.0]

    def test_validation_errors_selects_highest_error_scores(self) -> None:
        y_true = np.array([[1, 0], [1, 0], [1, 0]])
        y_pred = np.array([[0, 1], [0, 1], [0, 1]])
        t = create_validation_errors_table(
            input_texts=["low", "high", "mid"],
            y_true=y_true,
            y_pred=y_pred,
            classes=["economy", "tech"],
            raw_outputs=["r0", "r1", "r2"],
            parsed_outputs=["tech", "tech", "tech"],
            error_scores=[0.1, 0.9, 0.5],
            max_errors_per_class=1,
        )
        assert t["input_text"] == ["high"]

    def test_tables_and_plots_single_and_multi(self) -> None:
        y_true = np.array([[1, 0], [0, 1], [1, 1]])
        y_pred = np.array([[1, 0], [1, 0], [1, 1]])
        classes = ["economy", "tech"]

        t_single = create_single_label_confusion_matrix_table(y_true[:2], y_pred[:2], classes)
        assert len(t_single["count"]) == 4

        t_cross = create_cross_label_confusion_matrix_table(y_true, y_pred, classes)
        assert len(t_cross["co_occurrence_count"]) == 4

        t_metrics = create_per_class_metrics_table(y_true, y_pred, classes)
        assert t_metrics["class_name"] == classes

        t_card = create_label_cardinality_table(y_true, y_pred)
        assert "num_labels" in t_card

        t_err = create_validation_errors_table(
            input_texts=["t0", "t1", "t2"],
            y_true=y_true,
            y_pred=y_pred,
            classes=classes,
            raw_outputs=["r0", "r1", "r2"],
            parsed_outputs=[["economy"], ["economy"], ["economy", "tech"]],
            error_scores=[0.1, 0.9, 0.2],
            max_errors_per_class=2,
        )
        assert len(t_err["input_text"]) == 1
        assert t_err["input_text"][0] == "t1"

        t_no_err = create_validation_errors_table(
            input_texts=["t0"],
            y_true=y_true[:1],
            y_pred=y_pred[:1],
            classes=classes,
            raw_outputs=["r0"],
            parsed_outputs=[["economy"]],
        )
        assert t_no_err["input_text"] == []

        # Partial multi-label mismatch (differs on one class, matches on another)
        t_partial_err = create_validation_errors_table(
            input_texts=["partial"],
            y_true=np.array([[1, 1]]),
            y_pred=np.array([[1, 0]]),
            classes=classes,
            raw_outputs=["r_partial"],
            parsed_outputs=[["economy"]],
        )
        assert t_partial_err["input_text"] == ["partial"]

        for fig in (
            create_single_label_confusion_matrix_figure(y_true[:2], y_pred[:2], classes, epoch=1),
            create_cross_label_confusion_matrix_figure(y_true, y_pred, classes, epoch=1),
            create_per_class_metrics_figure(y_true, y_pred, classes, epoch=1),
            create_label_cardinality_figure(y_true, y_pred, epoch=1),
        ):
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_log_evaluation_artifacts_local_and_mlflow(self, tmp_path: Path) -> None:
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [1, 0]])
        classes = ["a", "b"]

        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.log_figure") as mock_log_fig,
            patch("mlflow.log_table") as mock_log_tbl,
        ):
            out_multi = log_evaluation_artifacts(
                y_true,
                y_pred,
                classes,
                multi_label=True,
                local_output_dir=tmp_path / "multi",
                validation_errors_table={"input_text": ["x"]},
            )
            assert "plot_cross_cm" in out_multi
            assert "table_errors" in out_multi
            assert mock_log_fig.call_count == 3
            assert mock_log_tbl.call_count == 4

            out_single = log_evaluation_artifacts(
                y_true,
                y_pred,
                classes,
                multi_label=False,
                local_output_dir=tmp_path / "single",
            )
            assert "plot_cm" in out_single
            assert Path(out_single["plot_cm"]).exists()


class TestTrackingManagerAndCLI:
    def _write_mlflow_cfg(self, tmp_path: Path) -> Path:
        db_file = tmp_path / "mlflow.db"
        art_dir = tmp_path / "mlruns"
        cfg_file = tmp_path / "mlflow-config.yml"
        cfg_file.write_text(
            f"server:\n"
            f"  host: 127.0.0.1\n"
            f"  port: 5000\n"
            f"  backend_store_uri: sqlite:///{db_file.as_posix()}\n"
            f"  default_artifact_root: {art_dir.as_posix()}\n",
            encoding="utf-8",
        )
        return cfg_file

    def _make_fake_run(self, tmp_path: Path) -> SimpleNamespace:
        art_dir = tmp_path / "run_artifacts"
        art_dir.mkdir(exist_ok=True)
        (art_dir / "plot.png").write_bytes(b"png")
        return SimpleNamespace(
            info=SimpleNamespace(
                run_id="1234567890abcdef",
                experiment_id="1",
                status="FINISHED",
                start_time=1_700_000_000_000,
                artifact_uri=art_dir.as_uri(),
            ),
            data=SimpleNamespace(
                params={"model_name": "Qwen/Qwen2.5-0.5B", "mode": "cls", "num_train_epochs": "3"},
                tags={"mlflow.runName": "easyslm-cls-qwen"},
                metrics={"eval_f1_macro": 0.88, "eval_loss": 0.22, "train_loss": 0.30},
            ),
        )

    def test_resolve_and_clear_mlflow_data(self, tmp_path: Path) -> None:
        cfg_file = self._write_mlflow_cfg(tmp_path)
        db_path, art_path = resolve_storage_paths(cfg_file)
        assert db_path is not None
        db_path.write_text("dummy", encoding="utf-8")
        art_path.mkdir(parents=True, exist_ok=True)
        (art_path / "sub").mkdir()
        (art_path / "sub" / "f.txt").write_text("x", encoding="utf-8")

        ok, msg = clear_mlflow_data(cfg_file, artifacts_only=True)
        assert ok is True
        assert db_path.exists()
        assert not (art_path / "sub").exists()

        ok2, msg2 = clear_mlflow_data(cfg_file, artifacts_only=False)
        assert ok2 is True
        assert not db_path.exists()
        assert "database file" in msg2

        art_path.rmdir()
        ok3, msg3 = clear_mlflow_data(cfg_file, artifacts_only=False)
        assert ok3 is True
        assert "already empty" in msg3

    def test_artifact_uri_to_path(self, tmp_path: Path) -> None:
        assert artifact_uri_to_path(tmp_path.as_uri()) == tmp_path
        assert artifact_uri_to_path(str(tmp_path)) == tmp_path

    def test_status_list_best_and_export(self, tmp_path: Path) -> None:
        cfg_file = self._write_mlflow_cfg(tmp_path)
        db_path, art_path = resolve_storage_paths(cfg_file)
        assert db_path is not None
        db_path.write_text("db", encoding="utf-8")
        art_path.mkdir(parents=True, exist_ok=True)

        fake_exp = SimpleNamespace(experiment_id="1", name="easySLM_finetuning")
        fake_run = self._make_fake_run(tmp_path)
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = [fake_exp]
        mock_client.search_runs.return_value = [fake_run]
        mock_client.get_run.return_value = fake_run

        with (
            patch("mlflow.tracking.MlflowClient", return_value=mock_client),
            patch("src.tracking.manager.check_mlflow_server_online", return_value=True),
        ):
            status = get_mlflow_status(cfg_file)
            assert status["server_online"] is True
            assert status["total_runs"] == 1
            assert "easySLM_finetuning" in status["best_f1_by_experiment"]

            runs = list_mlflow_runs(cfg_file, sort_by="f1", model="Qwen")
            assert len(runs) == 1
            assert runs[0]["mode"] == "cls"

            best = get_best_mlflow_run(cfg_file, model="Qwen", metric="eval_f1_macro")
            assert best is not None
            assert best["metric_value"] == pytest.approx(0.88)

            export_dir = tmp_path / "exported"
            copied = export_run_artifacts("1234567890abcdef", export_dir, cfg_file)
            assert len(copied) == 1
            assert (export_dir / "plot.png").exists()

    def test_mlflow_cli_subcommands(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_file = self._write_mlflow_cfg(tmp_path)
        table = format_cli_table(["ColA", "ColB"], [["1", "222"]])
        assert "ColA" in table

        with patch("src.tracking.cli.start_mlflow_server", return_value=0) as mock_start:
            assert handle_mlflow_cli(["start", "-c", str(cfg_file), "--port", "5009"]) == 0
            mock_start.assert_called_once()

        with patch("src.tracking.cli.clear_mlflow_data", return_value=(True, "Cleaned")):
            assert handle_mlflow_cli(["clear", "-c", str(cfg_file), "-y"]) == 0

        with patch("builtins.input", return_value="n"):
            assert handle_mlflow_cli(["clear", "-c", str(cfg_file)]) == 0

        fake_exp = SimpleNamespace(experiment_id="1", name="easySLM_finetuning")
        fake_run = self._make_fake_run(tmp_path)
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = [fake_exp]
        mock_client.search_runs.return_value = [fake_run]
        mock_client.get_run.return_value = fake_run

        with (
            patch("mlflow.tracking.MlflowClient", return_value=mock_client),
            patch("src.tracking.manager.check_mlflow_server_online", return_value=False),
        ):
            assert handle_mlflow_cli(["status", "-c", str(cfg_file)]) == 0
            assert handle_mlflow_cli(["list", "-c", str(cfg_file), "--sort-by", "loss"]) == 0
            export_dir = tmp_path / "cli_export"
            assert (
                handle_mlflow_cli(["best", "-c", str(cfg_file), "--export-to", str(export_dir)])
                == 0
            )
        out = capsys.readouterr().out
        assert "System & Storage Status" in out
        assert "Best Performing Run" in out
