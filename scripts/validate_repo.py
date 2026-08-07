#!/usr/bin/env python3
from __future__ import annotations

import compileall
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ".python-version",
    "README.md",
    "launchable/setup.sh",
    "launchable/brev-launchable.yaml",
    "scripts/start_nim.sh",
    "scripts/stop_nim.sh",
    "labs/00_start_here.ipynb",
    "labs/01_deploy_and_benchmark.ipynb",
    "labs/02_domain_adaptation.ipynb",
    "labs/03_onnx_triton.ipynb",
    "triton/model_repository/parakeet_ctc/config.pbtxt",
]


def main() -> int:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).exists()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")

    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    assert python_version == "3.12"

    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in pyproject_text

    setup_text = (ROOT / "launchable" / "setup.sh").read_text(encoding="utf-8")
    assert setup_text.startswith("#!/bin/bash\n")
    assert 'PYTHON_VERSION="3.12"' in setup_text
    assert '"${UV_BIN}" venv --managed-python --clear --python "${PYTHON_VERSION}"' in setup_text
    assert 'c.ServerApp.default_url = "/lab/tree/labs/00_start_here.ipynb?reset"' in setup_text
    assert 'c.ServerApp.root_dir = str(_workshop_root)' in setup_text
    embedded_python = re.findall(r"<<'PY'\n(.*?)\nPY", setup_text, flags=re.DOTALL)
    assert embedded_python, "setup.sh must contain embedded Python checks"
    for index, source in enumerate(embedded_python):
        compile(source, f"launchable/setup.sh:python-{index}", "exec")

    manifest_text = (ROOT / "launchable" / "brev-launchable.yaml").read_text(encoding="utf-8")
    assert "mode: VM" in manifest_text
    assert "gpu: 1x NVIDIA L4" in manifest_text
    assert "container_id: parakeet-0-6b-ctc-en-us" in manifest_text
    assert "bs=1,mode=ofl" in manifest_text

    nim_start_text = (ROOT / "scripts" / "start_nim.sh").read_text(encoding="utf-8")
    assert nim_start_text.startswith("#!/bin/bash\n")
    assert "parakeet-0-6b-ctc-en-us" in nim_start_text
    assert "bs=1,mode=ofl" in nim_start_text
    assert "/v1/health/ready" in nim_start_text

    for notebook in sorted((ROOT / "labs").glob("*.ipynb")):
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"], f"{notebook.name} contains no cells"
        notebook_python = payload.get("metadata", {}).get("language_info", {}).get("version")
        assert notebook_python == "3.12", f"{notebook.name} must declare Python 3.12"
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{notebook.name}:cell-{index}", "exec")

    domain_payload = json.loads(
        (ROOT / "labs" / "02_domain_adaptation.ipynb").read_text(encoding="utf-8")
    )
    domain_source = "".join(
        line for cell in domain_payload["cells"] for line in cell.get("source", [])
    )
    assert "LANGUAGE_CONFIG = 'nl_nl'" in domain_source
    assert "fine_tune_with_validation" in domain_source
    assert "baseline_test" in domain_source
    assert "english_guardrail" in domain_source

    triton_payload = json.loads(
        (ROOT / "labs" / "03_onnx_triton.ipynb").read_text(encoding="utf-8")
    )
    triton_source = "".join(
        line for cell in triton_payload["cells"] for line in cell.get("source", [])
    )
    assert "export_fp32_onnx" in triton_source
    assert "torch.float32" in triton_source
    assert "np.isfinite(logits).all()" in triton_source
    assert "pytorch_prediction" in triton_source

    triton_config_text = (
        ROOT / "triton" / "model_repository" / "parakeet_ctc" / "config.pbtxt"
    ).read_text(encoding="utf-8")
    assert "TYPE_FP32" in triton_config_text
    assert "TYPE_FP16" not in triton_config_text

    for script in sorted((ROOT / "scripts").glob("*.sh")) + [ROOT / "launchable" / "setup.sh"]:
        subprocess.run(["bash", "-n", str(script)], check=True)

    if not compileall.compile_dir(ROOT / "src", quiet=1):
        raise RuntimeError("Python source compilation failed")

    print("Repository structure, Python 3.12 contract, YAML, notebooks, shell syntax and Python syntax are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
