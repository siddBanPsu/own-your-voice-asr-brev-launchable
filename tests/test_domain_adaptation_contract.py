import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from voice_asr_lab.asr import tokenizer_coverage
from voice_asr_lab.audio import normalize_latin_text


ROOT = Path(__file__).resolve().parents[1]


class _FakeTokenizer:
    unk_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[0, 1] if "unknown" in text else [1, 2])


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class DomainAdaptationContractTests(unittest.TestCase):
    def test_latin_normalization_matches_base_output_contract(self):
        self.assertEqual(
            normalize_latin_text("Héél goed—Ĳssel! 42"),
            "heel goed ijssel",
        )

    def test_tokenizer_coverage_reports_unknown_tokens(self):
        report = tokenizer_coverage(_FakeProcessor(), ["known text", "unknown text"])
        self.assertEqual(report["total_tokens"], 4)
        self.assertEqual(report["unknown_tokens"], 1)
        self.assertEqual(report["affected_examples"], ["unknown text"])

    def test_notebook_uses_fleurs_and_held_out_selection(self):
        notebook = json.loads(
            (ROOT / "labs" / "02_domain_adaptation.ipynb").read_text(encoding="utf-8")
        )
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        self.assertIn("LANGUAGE_CONFIG = 'nl_nl'", source)
        self.assertIn("VALIDATION_EXAMPLES", source)
        self.assertIn("TEST_EXAMPLES", source)
        self.assertIn("tokenizer_coverage", source)
        self.assertIn("fine_tune_with_validation", source)
        self.assertIn("best_step", source)
        self.assertIn("english_guardrail", source)


if __name__ == "__main__":
    unittest.main()
