import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NimContractTests(unittest.TestCase):
    def test_start_script_uses_supported_06b_offline_profile(self):
        script = (ROOT / "scripts" / "start_nim.sh").read_text(encoding="utf-8")
        self.assertIn("parakeet-0-6b-ctc-en-us", script)
        self.assertIn("bs=1,mode=ofl,diarizer=disabled,vad=default", script)
        self.assertIn("/v1/health/ready", script)
        self.assertIn("--publish 9000:9000", script)
        self.assertIn("--publish 50051:50051", script)

    def test_deployment_notebook_uses_nim_http_api(self):
        notebook = json.loads(
            (ROOT / "labs" / "01_deploy_and_benchmark.ipynb").read_text(encoding="utf-8")
        )
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("getpass", source)
        self.assertIn("del nim_env['NGC_API_KEY']", source)
        self.assertNotIn(".pop('NGC_API_KEY'", source)
        self.assertIn("/v1/audio/transcriptions", source)
        self.assertIn("scripts/stop_nim.sh", source)


if __name__ == "__main__":
    unittest.main()
