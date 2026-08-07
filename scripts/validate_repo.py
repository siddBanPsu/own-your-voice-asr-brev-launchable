#!/usr/bin/env python3
from __future__ import annotations

import compileall
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ".python-version",
    "README.md",
    "launchable/setup.sh",
    "launchable/brev-launchable.yaml",
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
    assert 'PYTHON_VERSION="3.12"' in setup_text
    assert '"${UV_BIN}" venv --managed-python --clear --python "${PYTHON_VERSION}"' in setup_text

    manifest_text = (ROOT / "launchable" / "brev-launchable.yaml").read_text(encoding="utf-8")
    assert "mode: VM" in manifest_text
    assert "gpu: 1x NVIDIA L4" in manifest_text

    for notebook in sorted((ROOT / "labs").glob("*.ipynb")):
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"], f"{notebook.name} contains no cells"
        notebook_python = payload.get("metadata", {}).get("language_info", {}).get("version")
        assert notebook_python == "3.12", f"{notebook.name} must declare Python 3.12"
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{notebook.name}:cell-{index}", "exec")

    for script in sorted((ROOT / "scripts").glob("*.sh")) + [ROOT / "launchable" / "setup.sh"]:
        subprocess.run(["bash", "-n", str(script)], check=True)

    if not compileall.compile_dir(ROOT / "src", quiet=1):
        raise RuntimeError("Python source compilation failed")

    print("Repository structure, Python 3.12 contract, YAML, notebooks, shell syntax and Python syntax are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
