#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${HOME}/.venvs/own-your-voice-asr"
WORKSPACE_DIR="${HOME}/workspace"
KERNEL_NAME="own-your-voice-asr"
KERNEL_DISPLAY_NAME="Own Your Voice ASR Labs"
PROFILE_VALUE="${LAB_PROFILE:-auto}"

case "${PROFILE_VALUE}" in
  auto|t4|l4|a100) ;;
  *)
    echo "LAB_PROFILE must be auto, t4, l4, or a100; received '${PROFILE_VALUE}'." >&2
    exit 1
    ;;
esac

retry() {
  local attempt=1
  local max_attempts=4
  local delay_seconds=5
  until "$@"; do
    if (( attempt >= max_attempts )); then
      echo "Command failed after ${max_attempts} attempts: $*" >&2
      return 1
    fi
    echo "Attempt ${attempt} failed; retrying in ${delay_seconds}s..." >&2
    sleep "${delay_seconds}"
    attempt=$((attempt + 1))
    delay_seconds=$((delay_seconds * 2))
  done
}

echo "[1/5] Checking the NVIDIA GPU"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

echo "[2/5] Installing audio and system tools"
retry sudo apt-get update -y
retry sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg \
  libsndfile1 \
  sox \
  libsox-fmt-all \
  curl \
  git-lfs \
  build-essential

echo "[3/5] Creating the workshop Python environment"
python3 -m venv "${VENV_DIR}"
retry "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[4/5] Installing pinned lab dependencies"
retry "${VENV_DIR}/bin/python" -m pip install \
  torch==2.13.0 \
  transformers==5.14.1 \
  accelerate==1.14.0 \
  'datasets[audio]==5.0.0' \
  onnx==1.22.0 \
  'tritonclient[http]==2.69.0' \
  jiwer==4.0.0 \
  librosa==0.11.0 \
  soundfile==0.13.1 \
  'numpy>=2.0,<3' \
  'pandas>=2.2,<3' \
  'matplotlib>=3.9,<4' \
  'ipykernel>=6.29,<7' \
  'ipywidgets>=8.1,<9'

"${VENV_DIR}/bin/python" -m ipykernel install --user \
  --name "${KERNEL_NAME}" \
  --display-name "${KERNEL_DISPLAY_NAME}"

mkdir -p "${WORKSPACE_DIR}/.cache/huggingface" "${WORKSPACE_DIR}/artifacts"

export WORKSHOP_KERNEL_JSON="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}/kernel.json"
export WORKSHOP_PROFILE_VALUE="${PROFILE_VALUE}"
export WORKSHOP_HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
"${VENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

kernel_path = Path(os.environ["WORKSHOP_KERNEL_JSON"])
kernel = json.loads(kernel_path.read_text(encoding="utf-8"))
kernel["env"] = {
    **kernel.get("env", {}),
    "HF_HOME": os.environ["WORKSHOP_HF_HOME"],
    "LAB_PROFILE": os.environ["WORKSHOP_PROFILE_VALUE"],
    "TOKENIZERS_PARALLELISM": "false",
}
kernel_path.write_text(json.dumps(kernel, indent=2) + "\n", encoding="utf-8")
PY

echo "[5/5] Verifying CUDA from the lab kernel"
"${VENV_DIR}/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the NVIDIA GPU. Check the Brev GPU and driver state.")

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
if vram_gb < 14:
    raise RuntimeError(f"The labs need at least 14 GB VRAM; detected {vram_gb:.1f} GB.")

print(f"Ready: {props.name}, {vram_gb:.1f} GB VRAM, CUDA {torch.version.cuda}")
PY

echo
echo "Setup complete. Open Jupyter and select the '${KERNEL_DISPLAY_NAME}' kernel."
echo "Start with labs/00_start_here.ipynb."
