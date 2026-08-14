#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_asr_lab.profiles import detect_profile  # noqa: E402


PACKAGES = [
    "torch",
    "nemo-toolkit",
    "nemo2riva",
    "datasets",
    "jiwer",
    "librosa",
    "soundfile",
    "requests",
    "tensorboard",
    "setuptools",
]


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 is required; detected {sys.version.split()[0]}.")

    import torch

    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "docker": shutil.which("docker") is not None,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "packages": {},
    }
    for package in PACKAGES:
        checks["packages"][package] = importlib.metadata.version(package)

    setuptools_version = checks["packages"]["setuptools"]
    if setuptools_version != "80.9.0":
        raise RuntimeError(
            "TensorBoard requires Setuptools 80.9.0 in this environment; "
            f"detected {setuptools_version}. Reinstall setuptools==80.9.0 and rerun preflight."
        )
    if importlib.util.find_spec("pkg_resources") is None:
        raise RuntimeError("TensorBoard requires pkg_resources from Setuptools 80.9.0.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this kernel.")
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    checks["gpu"] = props.name
    checks["vram_gb"] = round(props.total_memory / 1024**3, 1)
    checks["cuda"] = torch.version.cuda
    checks["compute_capability"] = f"{capability[0]}.{capability[1]}"
    checks["torch_arch_list"] = torch.cuda.get_arch_list()
    device_arch = f"sm_{capability[0]}{capability[1]}"
    checks["torch_supports_detected_architecture"] = device_arch in checks["torch_arch_list"]
    if not checks["torch_supports_detected_architecture"]:
        raise RuntimeError(
            f"The installed PyTorch wheel does not contain {device_arch} kernels. "
            "On an RTX PRO 6000 Blackwell, rerun the Launchable setup so it selects cu129."
        )
    checks["speech_nim_supported"] = capability[0] >= 8
    checks["profile"] = detect_profile().as_dict()

    if checks["nvidia_smi"]:
        subprocess.run(["nvidia-smi", "-L"], check=True)
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
