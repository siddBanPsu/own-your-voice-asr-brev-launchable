#!/usr/bin/env python3
from __future__ import annotations

import compileall
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
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

    manifest_text = (ROOT / "launchable" / "brev-launchable.yaml").read_text(encoding="utf-8")
    assert "mode: VM" in manifest_text
    assert "gpu: 1x NVIDIA L4" in manifest_text

    for notebook in sorted((ROOT / "labs").glob("*.ipynb")):
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert payload["cells"], f"{notebook.name} contains no cells"
        for index, cell in enumerate(payload["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{notebook.name}:cell-{index}", "exec")

    for script in sorted((ROOT / "scripts").glob("*.sh")) + [ROOT / "launchable" / "setup.sh"]:
        subprocess.run(["bash", "-n", str(script)], check=True)

    if not compileall.compile_dir(ROOT / "src", quiet=1):
        raise RuntimeError("Python source compilation failed")

    print("Repository structure, YAML, notebooks, shell syntax and Python syntax are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
