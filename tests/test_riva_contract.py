import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RivaContractTests(unittest.TestCase):
    def test_build_script_uses_integrated_nemo_conversion(self):
        script = (ROOT / "scripts" / "build_riva_rmir.sh").read_text(encoding="utf-8")
        self.assertTrue(script.startswith("#!/bin/bash\n"))
        self.assertIn("riva-speech:${RIVA_VERSION}", script)
        self.assertIn("--entrypoint riva-build", script)
        self.assertIn("model.nemo", script)
        self.assertIn("nemo2riva", script)
        self.assertIn("onnx_opset:19", script)
        self.assertIn("own_your_voice_asr.rmir", script)

    def test_local_riva_script_deploys_rmir_and_exposes_grpc(self):
        script = (ROOT / "scripts" / "start_riva.sh").read_text(encoding="utf-8")
        self.assertIn("--entrypoint riva-deploy", script)
        self.assertIn("/data/models", script)
        self.assertIn("start-riva", script)
        self.assertIn("--publish 50051:50051", script)
        self.assertIn("--asr_service=true", script)

    def test_lab3_uses_riva_api_and_has_eks_path(self):
        notebook = json.loads(
            (ROOT / "labs" / "03_riva_deployment.ipynb").read_text(encoding="utf-8")
        )
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        self.assertIn("RIVA_VERSION = '2.26.0'", source)
        self.assertIn("scripts' / 'build_riva_rmir.sh", source)
        self.assertIn("scripts' / 'start_riva.sh", source)
        self.assertIn("riva.client.ASRService", source)
        self.assertIn("offline_recognize", source)
        self.assertIn("deploy' / 'eks'", source)
        self.assertIn("CHECK_EKS_PREREQUISITES = False", source)
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "own-your-voice-riva")
        self.assertEqual(
            notebook["metadata"]["kernelspec"]["display_name"],
            "Own Your Voice Riva Client",
        )

    def test_eks_override_selects_only_custom_model(self):
        values = (ROOT / "deploy" / "eks" / "values-custom-rmir.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("own_your_voice_asr:1.0", values)
        self.assertIn("asr: true", values)
        self.assertIn("nlp: false", values)
        self.assertIn("tts: false", values)

    def test_setup_opens_lab_zero_from_jupyter_root(self):
        setup = (ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
        self.assertIn(
            'c.ServerApp.default_url = "/lab/tree/labs/00_start_here.ipynb?reset"',
            setup,
        )
        self.assertIn('c.ServerApp.root_dir = str(_workshop_root)', setup)
        self.assertIn('RIVA_VENV_DIR="${HOME}/.venvs/own-your-voice-riva"', setup)
        self.assertIn('RIVA_KERNEL_NAME="own-your-voice-riva"', setup)
        self.assertIn("nvidia-riva-client==2.26.0", setup)


if __name__ == "__main__":
    unittest.main()
