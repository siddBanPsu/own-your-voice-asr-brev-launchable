import json
from pathlib import Path
import unittest

from voice_asr_lab.triton import triton_config


ROOT = Path(__file__).resolve().parents[1]


class TritonContractTests(unittest.TestCase):
    def test_model_config_uses_fp32_boundary(self):
        config = triton_config(vocab_size=1025)
        self.assertIn('name: "input_features" data_type: TYPE_FP32', config)
        self.assertIn('name: "logits" data_type: TYPE_FP32', config)
        self.assertNotIn("TYPE_FP16", config)

    def test_notebook_checks_finite_logits_and_pytorch_parity(self):
        notebook = json.loads(
            (ROOT / "labs" / "03_onnx_triton.ipynb").read_text(encoding="utf-8")
        )
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        self.assertIn("export_fp32_onnx", source)
        self.assertIn("to(dtype=torch.float32)", source)
        self.assertIn("np.isfinite(logits).all()", source)
        self.assertIn("transcript_match", source)
        self.assertNotIn("export_fp16_onnx", source)

    def test_setup_opens_lab_zero_from_jupyter_root(self):
        setup = (ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
        self.assertIn(
            'c.ServerApp.default_url = "/lab/tree/labs/00_start_here.ipynb?reset"',
            setup,
        )
        self.assertIn('c.ServerApp.root_dir = str(_workshop_root)', setup)
        self.assertIn('_workshop_home / "workspace"', setup)


if __name__ == "__main__":
    unittest.main()
