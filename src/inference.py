import json
import logging
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizerBase

from src.data.collator import SLMCollator
from src.slm.base_slm import SLMBase
from src.slm.cls_slm import SLMCls
from src.slm.json_slm import SLMJson
from src.slm.serialization import load_finetuned_model
from src.slm.slm_factory import ModelFactory
from src.utils.config import load_chat_template, load_prompts_config
from src.utils.model_card import find_best_model

logger = logging.getLogger(__name__)


class SLMPredictor:
    """
    Inference engine for fine-tuned `SLMCls` and `SLMJson` models.

    Supports:
    - Loading models directly from a checkpoint directory.
    - Automatically selecting and loading the best fine-tuned model from model cards.
    - Processing single strings, lines of text, individual `.txt` files, or folders of files.
    - Handling `'|||'` split logic (predicting strictly on element 0 if `'|||'` is present).
    - Exporting predictions to user-defined directories in formatted `.txt` and `.json`.
    """

    def __init__(
        self,
        model: SLMBase,
        tokenizer: PreTrainedTokenizerBase,
        prompts_config_path: str | Path = "configs/prompts.yml",
        chat_template_config_path: str | Path = "configs/chat_template.yml",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts_config_path = prompts_config_path
        self.chat_template_config_path = chat_template_config_path
        self.mode = "cls" if isinstance(model, SLMCls) else "json"
        self.classes = model.classes
        self.multi_label = model.multi_label

        # Ensure chat template is set
        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = load_chat_template(chat_template_config_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        # Load prompts configuration
        self.prompts_config = load_prompts_config(prompts_config_path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        prompts_config_path: str | Path = "configs/prompts.yml",
        chat_template_config_path: str | Path = "configs/chat_template.yml",
        factory: ModelFactory | None = None,
    ) -> "SLMPredictor":
        """Loads a model from a specific checkpoint folder and returns an initialized SLMPredictor."""
        model, tokenizer, _ = load_finetuned_model(checkpoint_dir, factory=factory)
        return cls(
            model=model,
            tokenizer=tokenizer,
            prompts_config_path=prompts_config_path,
            chat_template_config_path=chat_template_config_path,
        )

    @classmethod
    def from_best_model(
        cls,
        models_dir: str | Path,
        model_type: str,
        metric: str = "eval_f1_macro",
        prompts_config_path: str | Path = "configs/prompts.yml",
        chat_template_config_path: str | Path = "configs/chat_template.yml",
        factory: ModelFactory | None = None,
    ) -> tuple["SLMPredictor", Path, dict[str, Any]]:
        """
        Discovers the best model in `models_dir` using its model cards and returns an initialized predictor.

        Returns:
            `(predictor, best_checkpoint_dir, best_model_card_data)`
        """
        best_dir, best_card = find_best_model(models_dir, model_type=model_type, metric=metric)
        predictor = cls.from_checkpoint(
            checkpoint_dir=best_dir,
            prompts_config_path=prompts_config_path,
            chat_template_config_path=chat_template_config_path,
            factory=factory,
        )
        return predictor, best_dir, best_card

    def _build_system_prompt(self) -> str:
        """Constructs the system prompt dynamically from `configs/prompts.yml`."""
        classes_str = ", ".join(f"'{c}'" for c in self.classes)
        mode_descriptions = self.prompts_config.get("mode_descriptions", {})

        if self.mode == "cls":
            default_mode_desc = (
                mode_descriptions.get("multi_label", "one or more applicable classes (multi-label)")
                if self.multi_label
                else mode_descriptions.get(
                    "single_label", "the single most appropriate class (single-label)"
                )
            )
            template = (
                self.prompts_config.get("cls", {}).get("system_prompt")
                or "You are an expert text classification assistant. Classify the user text into {mode_desc} from the following valid classes: [{classes}]."
            )
            return template.format(classes=classes_str, mode_desc=default_mode_desc).strip()
        else:
            default_mode_desc = (
                mode_descriptions.get("multi_label", "one or more applicable classes")
                if self.multi_label
                else mode_descriptions.get("single_label", "the single most appropriate class")
            )
            example_labels = self.classes[: min(2, len(self.classes))]
            example_str = f"[{', '.join(repr(lbl) for lbl in example_labels)}]"
            template = (
                self.prompts_config.get("json", {}).get("system_prompt")
                or "You are a text classification model. Classify the user text into {mode_desc} from the following valid classes: [{classes}]. Respond ONLY with a valid list of predicted class names sorted in alphabetical order, for example: {example}. Do not output any additional text or explanation."
            )
            return template.format(
                classes=classes_str,
                mode_desc=default_mode_desc,
                example=example_str,
            ).strip()

    def _render_chat_ids(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Applies the tokenizer's chat template and returns a list of integer token IDs."""
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(rendered, list):
            if hasattr(rendered, "input_ids"):
                return list(rendered.input_ids)
            if isinstance(rendered, dict) and "input_ids" in rendered:
                return list(rendered["input_ids"])
        return [int(x) for x in rendered]  # type: ignore[union-attr, arg-type]

    def _prepare_prompt_messages(self, text: str) -> list[dict[str, str]]:
        system_prompt = self._build_system_prompt()
        user_prompt_template = self.prompts_config.get(self.mode, {}).get("user_prompt", "{text}")
        formatted_user = (
            user_prompt_template.format(text=text) if "{text}" in user_prompt_template else text
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user},
        ]

    @torch.no_grad()
    def predict_texts(self, texts: list[str], batch_size: int = 8) -> list[dict[str, Any]]:
        """
        Executes model inference on a list of clean texts.

        Returns:
            List of `{"text": text, "predictions": List[str], ...}`
        """
        results: list[dict[str, Any]] = []
        if not texts:
            return results

        collator = SLMCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            pad_labels=False,
            padding_side="left" if self.mode == "json" else "right",
            device=self.model.device,
        )

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded_samples: list[dict[str, torch.Tensor]] = []

            for t in batch_texts:
                messages = self._prepare_prompt_messages(t)
                if self.mode == "cls":
                    token_ids = self._render_chat_ids(messages, add_generation_prompt=False)
                    eos_id = self.tokenizer.eos_token_id
                    if eos_id is not None and (len(token_ids) == 0 or token_ids[-1] != eos_id):
                        token_ids = token_ids + [eos_id]
                else:
                    token_ids = self._render_chat_ids(messages, add_generation_prompt=True)

                encoded_samples.append(
                    {
                        "input_ids": torch.tensor(token_ids, dtype=torch.long),
                        "attention_mask": torch.tensor([1] * len(token_ids), dtype=torch.long),
                    }
                )

            batch = collator(encoded_samples)

            if self.mode == "cls":
                cls_model: SLMCls = cast(SLMCls, self.model)
                pred_out = cls_model.predict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    threshold=0.5,
                )
                predicted_labels_batch = pred_out["predictions"]
                probabilities = pred_out.get("probabilities", [])
                if isinstance(probabilities, torch.Tensor):
                    probabilities = probabilities.cpu().tolist()

                for j, t in enumerate(batch_texts):
                    preds = predicted_labels_batch[j]
                    # If multi-label produced empty set due to threshold, fallback to argmax top-1
                    if not preds and len(probabilities) > j:
                        probs_row = probabilities[j]
                        max_idx = int(torch.tensor(probs_row).argmax().item())
                        preds = [self.classes[max_idx]]

                    results.append(
                        {
                            "text": t,
                            "predictions": preds,
                            "probabilities": {
                                c: round(p, 4)
                                for c, p in zip(self.classes, probabilities[j], strict=False)
                            }
                            if len(probabilities) > j
                            else {},
                        }
                    )
            else:
                json_model: SLMJson = cast(SLMJson, self.model)
                pred_out = json_model.predict(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    tokenizer=self.tokenizer,
                    max_new_tokens=64,
                    do_sample=False,
                )
                predictions = pred_out.get("predictions", [])
                raw_outputs = pred_out.get("raw_outputs", [])

                for j, t in enumerate(batch_texts):
                    results.append(
                        {
                            "text": t,
                            "predictions": predictions[j] if j < len(predictions) else [],
                            "raw_output": raw_outputs[j] if j < len(raw_outputs) else "",
                        }
                    )

        return results

    def predict_lines(self, lines: list[str], batch_size: int = 8) -> list[dict[str, Any]]:
        """
        Parses each raw line: if '|||' is present, splits on '|||' and takes element 0 as input text.
        """
        extracted_data: list[tuple[str, str, str | None]] = []
        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            if "|||" in raw:
                parts = raw.split("|||", 1)
                text = parts[0].strip()
                ground_truth = parts[1].strip()
            else:
                text = raw
                ground_truth = None
            extracted_data.append((raw, text, ground_truth))

        clean_texts = [item[1] for item in extracted_data]
        preds = self.predict_texts(clean_texts, batch_size=batch_size)

        final_records: list[dict[str, Any]] = []
        for (raw, _text, gt), pred in zip(extracted_data, preds, strict=False):
            record = dict(pred)
            record["raw_line"] = raw
            if gt is not None:
                record["ground_truth"] = [lbl.strip() for lbl in gt.split(",") if lbl.strip()]
            final_records.append(record)

        return final_records

    def predict_file_or_dir(
        self,
        data_path: str | Path,
        output_dir: str | Path,
        batch_size: int = 8,
    ) -> list[Path]:
        """
        Processes a `.txt` file or all `.txt` files in a directory and writes predictions
        to `output_dir` in `.txt` format (`<text> ||| <comma_separated_labels>`) and `.json` format.

        Returns:
            List of generated output prediction files.
        """
        path = Path(data_path).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            raise FileNotFoundError(f"Input data path not found: '{path}'")

        if path.is_file():
            txt_files = [path]
        elif path.is_dir():
            txt_files = sorted(path.glob("*.txt"))
            if not txt_files:
                raise ValueError(f"No '.txt' files found in directory '{path}'")
        else:
            raise ValueError(f"Unsupported path type: '{path}'")

        saved_files: list[Path] = []

        for txt_file in txt_files:
            lines = txt_file.read_text(encoding="utf-8").splitlines()
            records = self.predict_lines(lines, batch_size=batch_size)

            stem = txt_file.stem
            out_txt = out_dir / f"{stem}_predictions_{self.mode}.txt"
            out_json = out_dir / f"{stem}_predictions_{self.mode}.json"

            # 1. Write formatted .txt (<text> ||| <labels>)
            txt_content = []
            for r in records:
                preds_str = ", ".join(r["predictions"])
                txt_content.append(f"{r['text']} ||| {preds_str}")
            out_txt.write_text("\n".join(txt_content) + "\n", encoding="utf-8")

            # 2. Write structured .json
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

            logger.info(
                "Saved %d predictions to '%s' and '%s'", len(records), out_txt.name, out_json.name
            )
            saved_files.append(out_txt)
            saved_files.append(out_json)

        return saved_files
