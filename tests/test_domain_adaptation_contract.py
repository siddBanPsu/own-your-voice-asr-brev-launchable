import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from voice_asr_lab.audio import load_fleurs_records, normalize_latin_text
from voice_asr_lab.nemo import nemo_tokenizer_coverage


ROOT = Path(__file__).resolve().parents[1]


class _FakeSentencePiece:
    def unk_id(self):
        return 0


class _FakeNeMoTokenizer:
    tokenizer = _FakeSentencePiece()

    def text_to_ids(self, text):
        return [0, 1] if "unknown" in text else [1, 2]


class _FakeDataset(list):
    def cast_column(self, *_args, **_kwargs):
        return self


class DomainAdaptationContractTests(unittest.TestCase):
    def test_latin_normalization_matches_base_output_contract(self):
        self.assertEqual(normalize_latin_text("Héél goed—Ĳssel! 42"), "heel goed ijssel")

    def test_nemo_tokenizer_coverage_reports_unknown_tokens(self):
        report = nemo_tokenizer_coverage(
            _FakeNeMoTokenizer(), ["known text", "unknown text"]
        )
        self.assertEqual(report["total_tokens"], 4)
        self.assertEqual(report["unknown_tokens"], 1)
        self.assertEqual(report["affected_examples"], ["unknown text"])

    def test_fleurs_limit_uses_maximum_available_records(self):
        datasets_module = ModuleType("datasets")
        datasets_module.Audio = lambda **_kwargs: object()
        datasets_module.load_dataset = lambda *_args, **_kwargs: _FakeDataset(
            [
                {"audio": object(), "transcription": f"sample {index}", "id": index}
                for index in range(3)
            ]
        )
        with (
            patch.dict(sys.modules, {"datasets": datasets_module}),
            patch(
                "voice_asr_lab.audio.decode_dataset_audio",
                return_value=([0.0] * 16_000, 16_000),
            ),
            self.assertWarnsRegex(RuntimeWarning, "using all 3"),
        ):
            records = load_fleurs_records(
                "nl_nl", "train", limit=4, max_audio_seconds=6
            )
        self.assertEqual(len(records), 3)

    def test_notebook_uses_nemo_fleurs_and_held_out_selection(self):
        notebook = json.loads(
            (ROOT / "labs" / "02_domain_adaptation.ipynb").read_text(encoding="utf-8")
        )
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        self.assertIn("LANGUAGE_CONFIG = 'nl_nl'", source)
        self.assertIn("nemo_asr.models.ASRModel.from_pretrained", source)
        self.assertIn("nemo_tokenizer_coverage", source)
        self.assertIn("monitor='val_wer'", source)
        self.assertIn("baseline_test", source)
        self.assertIn("selected_english", source)
        self.assertIn("model.save_to", source)
        self.assertIn("tokenizer_retrained': False", source)
        self.assertIn("deterministic='warn'", source)
        self.assertNotIn("deterministic=True", source)
        self.assertIn("TRAIN_EXAMPLES = COMMON_TRAIN_EXAMPLES", source)
        self.assertIn("VALIDATION_EXAMPLES = COMMON_VALIDATION_EXAMPLES", source)
        self.assertIn("TEST_EXAMPLES = COMMON_TEST_EXAMPLES", source)
        self.assertIn("MAX_AUDIO_SECONDS = COMMON_MAX_AUDIO_SECONDS", source)
        self.assertIn(
            "TRAINABLE_ENCODER_LAYERS = profile.trainable_encoder_layers", source
        )
        self.assertIn(
            "ACCUMULATE_GRAD_BATCHES = profile.gradient_accumulation_steps", source
        )
        self.assertIn(
            "TRAIN_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES == COMMON_EFFECTIVE_BATCH_SIZE",
            source,
        )
        self.assertIn("'train_examples': len(train_records)", source)
        self.assertIn("'actual_examples': actual_examples", source)
        self.assertNotIn("80 if profile.name", source)


if __name__ == "__main__":
    unittest.main()
