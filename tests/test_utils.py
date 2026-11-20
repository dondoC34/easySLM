"""Unit tests for src/utils/* modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from src.utils.config import (
    DEFAULT_PROMPTS_CONFIG,
    EASYSLM_HOME_ENV_VAR,
    get_default_finetuned_dir,
    get_default_mlflow_artifact_root,
    get_default_mlflow_db_path,
    get_default_mlflow_tracking_uri,
    get_default_models_cache_dir,
    get_default_predictions_dir,
    get_easyslm_home,
    get_project_root,
    load_prompts_config,
    load_yaml,
)
from src.utils.device import get_default_device, get_default_dtype, print_cuda_info
from src.utils.help_tables import (
    get_combined_help_tables,
    get_ftune_help_table,
    get_mlflow_help_table,
    get_model_examples_table,
    get_predict_help_table,
    render_table,
)
from src.utils.model_card import (
    clean_slug,
    find_best_model,
    generate_model_card_markdown,
    get_metric_description,
    parse_model_card,
)


class TestUtilsConfig:
    def test_get_project_root_exists(self) -> None:
        root = get_project_root()
        assert root.is_dir()
        assert (root / "pyproject.toml").exists()

    def test_load_yaml_absolute_and_missing(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "sample.yml"
        cfg_file.write_text("a: 1\nb: hello\n", encoding="utf-8")
        assert load_yaml(cfg_file) == {"a": 1, "b": "hello"}

        empty_file = tmp_path / "empty.yml"
        empty_file.write_text("", encoding="utf-8")
        assert load_yaml(empty_file) == {}

        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yml")

    def test_load_yaml_relative_to_project_root(self) -> None:
        data = load_yaml("configs/prompts.yml")
        assert isinstance(data, dict)
        assert "cls" in data or "json" in data

    def test_load_yaml_relative_to_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "custom.yml").write_text("foo: bar\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert load_yaml("custom.yml") == {"foo": "bar"}

    def test_load_prompts_config_custom_and_fallback(self, tmp_path: Path) -> None:
        custom = tmp_path / "prompts.yml"
        custom.write_text("cls:\n  user_prompt: '{text}'\n", encoding="utf-8")
        loaded = load_prompts_config(custom)
        assert loaded == {"cls": {"user_prompt": "{text}"}}

        fallback = load_prompts_config(tmp_path / "missing_prompts.yml")
        assert fallback == DEFAULT_PROMPTS_CONFIG

    def test_easyslm_home_default_and_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(EASYSLM_HOME_ENV_VAR, raising=False)
        assert get_easyslm_home() == (Path.home() / ".easyslm").resolve()

        custom_home = tmp_path / "custom_easyslm_home"
        monkeypatch.setenv(EASYSLM_HOME_ENV_VAR, str(custom_home))
        resolved = custom_home.resolve()
        assert get_easyslm_home() == resolved
        assert get_default_models_cache_dir() == resolved / "models"
        assert get_default_finetuned_dir() == resolved / "finetuned"
        assert get_default_predictions_dir() == resolved / "predictions"
        assert get_default_mlflow_db_path() == resolved / "mlflow.db"
        assert get_default_mlflow_artifact_root() == resolved / "mlruns"
        assert (
            get_default_mlflow_tracking_uri() == f"sqlite:///{(resolved / 'mlflow.db').as_posix()}"
        )

    def test_easyslm_home_overridable_by_explicit_params(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.fine_tuning.config import FineTuningConfig
        from src.slm.slm_factory import ModelFactory
        from src.tracking.config import get_mlflow_config

        custom_home = tmp_path / "home_dir"
        monkeypatch.setenv(EASYSLM_HOME_ENV_VAR, str(custom_home))

        factory_default = ModelFactory()
        assert factory_default.cache_dir == (custom_home / "models").resolve()

        explicit_cache = tmp_path / "explicit_models"
        factory_override = ModelFactory(cache_dir=explicit_cache)
        assert factory_override.cache_dir == explicit_cache

        ft_default = FineTuningConfig.from_yaml("configs/training_config.yml")
        assert Path(ft_default.output_dir) == (custom_home / "finetuned").resolve()

        explicit_out = tmp_path / "explicit_finetuned"
        ft_override = FineTuningConfig.from_yaml(
            "configs/training_config.yml", output_dir=str(explicit_out)
        )
        assert Path(ft_override.output_dir) == explicit_out

        mlflow_default = get_mlflow_config()
        assert mlflow_default["server"]["backend_store_uri"] == (
            f"sqlite:///{(custom_home / 'mlflow.db').resolve().as_posix()}"
        )
        assert (
            Path(mlflow_default["server"]["default_artifact_root"])
            == (custom_home / "mlruns").resolve()
        )

        custom_mlflow_yml = tmp_path / "custom-mlflow.yml"
        custom_mlflow_yml.write_text(
            "server:\n  backend_store_uri: sqlite:///custom.db\n  default_artifact_root: /tmp/custom_runs\n",
            encoding="utf-8",
        )
        mlflow_override = get_mlflow_config(custom_mlflow_yml)
        assert mlflow_override["server"]["backend_store_uri"] == "sqlite:///custom.db"
        assert mlflow_override["server"]["default_artifact_root"] == "/tmp/custom_runs"


class TestUtilsDevice:
    def test_get_default_device_cpu_and_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            assert get_default_device() == torch.device("cpu")
            assert get_default_dtype() == torch.float32

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            assert get_default_device() == torch.device("cuda")
            assert get_default_dtype(torch.device("cuda")) == torch.bfloat16

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=False),
        ):
            assert get_default_dtype(torch.device("cuda")) == torch.float16

    def test_print_cuda_info_cpu_and_cuda(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            print_cuda_info()
        out = capsys.readouterr().out
        assert "CUDA Available: False" in out

        fake_props = SimpleNamespace(name="FakeGPU", total_memory=8 * (1024**3), major=8, minor=6)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch.object(torch.version, "cuda", "12.4"),
        ):
            print_cuda_info()
        out = capsys.readouterr().out
        assert "CUDA Available: True" in out
        assert "FakeGPU" in out
        assert "bfloat16 supported: True" in out


class TestHelpTables:
    def test_render_table_and_presets(self) -> None:
        table = render_table(
            title="Test",
            subtitle="Sub",
            rows=[("--flag", "str", "A very long description that wraps nicely across lines")],
            w_param=10,
            w_type=10,
            w_desc=20,
        )
        assert "=== TEST ===" in table
        assert "--flag" in table

        assert "RECOMMENDED BASE SLM MODELS" in get_model_examples_table()
        assert "COMMAND: EASYSLM FTUNE" in get_ftune_help_table()
        assert "COMMAND: EASYSLM PREDICT" in get_predict_help_table()
        assert "COMMAND: EASYSLM MLFLOW" in get_mlflow_help_table()
        combined = get_combined_help_tables()
        assert "easySLM CLI - PARAMETER REFERENCE" in combined
        assert "easyslm info" in combined


class TestModelCard:
    def test_get_metric_description_and_clean_slug(self) -> None:
        assert "Validation set loss" in get_metric_description("eval_loss")
        assert "F1" in get_metric_description("f1_macro")
        assert get_metric_description("totally_unknown_metric") == "Custom evaluation metric"
        assert clean_slug("Qwen/Qwen2.5-0.5B") == "Qwen2.5-0.5B"
        assert clean_slug("org/my model:v1") == "my_model_v1"

    def test_get_metric_description_train_prefix(self) -> None:
        assert "F1" in get_metric_description("train_f1_macro")
        assert get_metric_description("train_accuracy") != "Custom evaluation metric"

    def test_generate_and_parse_model_card(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "run_1"
        ckpt.mkdir()
        md = generate_model_card_markdown(
            model_name="Qwen/Qwen2.5-0.5B",
            model_type="cls",
            as_of="20260101_000000",
            metrics={
                "eval_loss": 0.25,
                "eval_f1_macro": 0.85,
                "eval_f1_micro": 0.88,
                "eval_exact_match_accuracy": 0.8,
                "eval_accuracy": 0.9,
            },
            training_params={"num_train_epochs": 3, "use_lora": True},
            classes=["economy", "sports"],
            checkpoint_dir=ckpt,
        )
        card_path = ckpt / "model_card_test.md"
        card_path.write_text(md, encoding="utf-8")

        parsed = parse_model_card(card_path)
        assert parsed["model"] == "Qwen/Qwen2.5-0.5B"
        assert parsed["type"] == "cls"
        assert parsed["metrics"]["eval_f1_macro"] == pytest.approx(0.85)
        assert parsed["_card_path"] == str(card_path)

    def test_parse_model_card_errors(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_model_card(tmp_path / "missing.md")

        no_fm = tmp_path / "no_fm.md"
        no_fm.write_text("# Just markdown", encoding="utf-8")
        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            parse_model_card(no_fm)

        bad_fm = tmp_path / "bad_fm.md"
        bad_fm.write_text("---\nonly_one_delimiter", encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            parse_model_card(bad_fm)

    def test_find_best_model(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_best_model(tmp_path / "does_not_exist", "cls")

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            find_best_model(empty_dir, "cls")

        # Create two CLS cards and one JSON card
        for name, mtype, f1, loss in [
            ("run_a", "cls", 0.72, 0.45),
            ("run_b", "cls", 0.91, 0.20),
            ("run_c", "json", 0.95, 0.10),
        ]:
            run_dir = tmp_path / name
            run_dir.mkdir()
            (run_dir / "easyslm_config.json").write_text("{}", encoding="utf-8")
            md = generate_model_card_markdown(
                model_name="tiny",
                model_type=mtype,
                as_of="20260101",
                metrics={"eval_f1_macro": f1, "eval_loss": loss},
                training_params={},
                classes=["a", "b"],
                checkpoint_dir=tmp_path / f"nonexistent_{name}",
            )
            (run_dir / f"model_card_{name}.md").write_text(md, encoding="utf-8")

        # Malformed card should be skipped with a warning
        (tmp_path / "model_card_broken.md").write_text("broken", encoding="utf-8")

        best_dir, best_card = find_best_model(tmp_path, "cls", metric="eval_f1_macro")
        assert best_dir == tmp_path / "run_b"
        assert best_card["metrics"]["eval_f1_macro"] == pytest.approx(0.91)

        # Test loss minimization and metric without eval_ prefix
        best_dir_loss, _ = find_best_model(tmp_path, "cls", metric="loss")
        assert best_dir_loss == tmp_path / "run_b"

        with pytest.raises(FileNotFoundError):
            find_best_model(tmp_path, "unknown_type")
