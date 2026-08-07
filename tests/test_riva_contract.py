import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RivaContractTests(unittest.TestCase):
    def test_build_script_exports_riva_then_uses_servicemaker_cli(self):
        script = (ROOT / "scripts" / "build_riva_rmir.sh").read_text(encoding="utf-8")
        self.assertTrue(script.startswith("#!/bin/bash\n"))
        self.assertIn("parakeet-0-6b-ctc-en-us", script)
        self.assertIn("ASR_NIM_TAG:-3.1.0", script)
        self.assertIn("nvcr.io/nim/nvidia/", script)
        self.assertIn("NEMO2RIVA_BIN", script)
        self.assertIn("--onnx-opset 19", script)
        self.assertIn("--max-dim 1000", script)
        self.assertIn("--entrypoint riva-build", script)
        self.assertIn("NEMO_MODEL", script)
        self.assertIn("model.riva", script)
        self.assertIn("nemo2riva", script)
        self.assertIn("speech_recognition", script)
        self.assertIn("--decoder_type=greedy", script)
        self.assertIn("--endpointing.residue_blanks_at_start=-16", script)
        self.assertIn("--return_separate_utterances=True", script)
        self.assertNotIn("--nn.fp16_needs_obey_precision_pass", script)
        self.assertNotIn("--chunk_size", script)
        self.assertNotIn("--config-path", script)
        self.assertNotIn("source_path=", script)
        self.assertIn("own_your_voice_asr.rmir", script)

    def test_local_riva_script_deploys_rmir_and_exposes_grpc(self):
        script = (ROOT / "scripts" / "start_riva.sh").read_text(encoding="utf-8")
        self.assertIn("--entrypoint riva-deploy", script)
        self.assertIn("/data/models", script)
        self.assertIn("custom_model.tar.gz", script)
        self.assertIn("NIM_DISABLE_MODEL_DOWNLOAD=true", script)
        self.assertIn("http://127.0.0.1:9000/v1/health/ready", script)
        self.assertIn("--publish 9000:9000", script)
        self.assertIn("--publish 50051:50051", script)

    def test_lab3_uses_riva_api_and_has_eks_path(self):
        notebook = json.loads(
            (ROOT / "labs" / "03_riva_deployment.ipynb").read_text(encoding="utf-8")
        )
        source = "".join(
            line for cell in notebook["cells"] for line in cell.get("source", [])
        )
        self.assertIn("ASR_NIM_TAG = '3.1.0'", source)
        self.assertIn("scripts' / 'build_riva_rmir.sh", source)
        self.assertIn("scripts' / 'start_riva.sh", source)
        self.assertIn("riva.client.ASRService", source)
        self.assertIn("offline_recognize", source)
        self.assertIn("Selected WAV contains no audio frames.", source)
        self.assertIn("sample_rate != 16_000", source)
        self.assertIn("RIVA_URI = '127.0.0.1:50051'", source)
        self.assertNotIn("localhost:50051", source)
        self.assertIn("deploy' / 'eks'", source)
        self.assertIn("CHECK_EKS_PREREQUISITES = False", source)
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "own-your-voice-riva")
        self.assertEqual(
            notebook["metadata"]["kernelspec"]["display_name"],
            "Own Your Voice Riva Client",
        )

    def test_eks_override_selects_custom_rmir_for_speech_nim(self):
        values = (ROOT / "deploy" / "eks" / "values-custom-rmir.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("nvcr.io/nim/nvidia/parakeet-0-6b-ctc-en-us", values)
        self.assertIn("tag: 3.1.0", values)
        self.assertIn("s3://REPLACE_ME/rmir/own_your_voice_asr.rmir", values)
        self.assertIn("ngcModelConfigs:", values)
        self.assertIn("grpcPort: 50051", values)

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
        self.assertIn("nemo2riva==2.22.0", setup)
        self.assertIn("appdirs==1.4.4", setup)
        self.assertIn("--no-build-isolation-package nvidia-pyindex", setup)

    def test_setup_uses_driver_compatible_pytorch_backend(self):
        setup = (ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
        self.assertIn('TORCH_BACKEND="cu126"', setup)
        self.assertIn('--torch-backend "${TORCH_BACKEND}"', setup)
        self.assertIn('torch.version.cuda != "12.6"', setup)


if __name__ == "__main__":
    unittest.main()
