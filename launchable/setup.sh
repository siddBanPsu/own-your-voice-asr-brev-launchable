#!/bin/bash
set -euo pipefail

VENV_DIR="${HOME}/.venvs/own-your-voice-asr"
RIVA_VENV_DIR="${HOME}/.venvs/own-your-voice-riva"
WORKSPACE_DIR="${HOME}/workspace"
KERNEL_NAME="own-your-voice-asr"
KERNEL_DISPLAY_NAME="Own Your Voice ASR Labs"
RIVA_KERNEL_NAME="own-your-voice-riva"
RIVA_KERNEL_DISPLAY_NAME="Own Your Voice Riva Client"
REPOSITORY_NAME="own-your-voice-asr-brev-launchable"
PROFILE_VALUE="${LAB_PROFILE:-auto}"
PYTHON_VERSION="3.12"
TORCH_BACKEND="${LAB_TORCH_BACKEND:-auto}"
TORCH_CUDA_VERSION=""
UV_VERSION="0.11.32"
UV_BIN_DIR="${HOME}/.local/bin"
UV_BIN="${UV_BIN_DIR}/uv"

case "${PROFILE_VALUE}" in
  auto|t4|l4|a100) ;;
  *)
    echo "LAB_PROFILE must be auto, t4, l4, or a100; received '${PROFILE_VALUE}'." >&2
    exit 1
    ;;
esac

select_torch_backend() {
  local compute_capability

  compute_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
  case "${TORCH_BACKEND}" in
    auto)
      case "${compute_capability}" in
        12.*)
          TORCH_BACKEND="cu129"
          TORCH_CUDA_VERSION="12.9"
          ;;
        *)
          TORCH_BACKEND="cu126"
          TORCH_CUDA_VERSION="12.6"
          ;;
      esac
      ;;
    cu126)
      TORCH_CUDA_VERSION="12.6"
      ;;
    cu129)
      TORCH_CUDA_VERSION="12.9"
      ;;
    *)
      echo "LAB_TORCH_BACKEND must be auto, cu126, or cu129; received '${TORCH_BACKEND}'." >&2
      exit 1
      ;;
  esac

  echo "Detected compute capability ${compute_capability}; installing PyTorch ${TORCH_BACKEND}."
}

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

echo "[1/6] Checking the NVIDIA GPU"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader
select_torch_backend

echo "[2/6] Installing audio and system tools"
retry sudo apt-get update -y
retry sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg \
  libsndfile1 \
  sox \
  libsox-fmt-all \
  curl \
  git-lfs \
  build-essential

echo "[3/6] Installing uv and managed Python ${PYTHON_VERSION}"
mkdir -p "${UV_BIN_DIR}"
UV_INSTALLER="$(mktemp)"
retry curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o "${UV_INSTALLER}"
env UV_UNMANAGED_INSTALL="${UV_BIN_DIR}" sh "${UV_INSTALLER}"
rm -f "${UV_INSTALLER}"
retry "${UV_BIN}" python install "${PYTHON_VERSION}"

echo "[4/6] Creating the workshop Python ${PYTHON_VERSION} environment"
"${UV_BIN}" venv --managed-python --clear --python "${PYTHON_VERSION}" "${VENV_DIR}"
"${VENV_DIR}/bin/python" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Python 3.12 is required; detected {sys.version.split()[0]}.")

print(f"Using Python {sys.version.split()[0]}")
PY
retry "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade \
  pip \
  setuptools==80.9.0 \
  wheel \
  appdirs==1.4.4

echo "[5/6] Installing pinned lab dependencies"
retry "${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" \
  --torch-backend "${TORCH_BACKEND}" \
  --no-build-isolation-package nvidia-pyindex \
  torch==2.13.0 \
  'nemo_toolkit[asr]==2.7.3' \
  nemo2riva==2.22.0 \
  'datasets[audio]==5.0.0' \
  jiwer==3.1.0 \
  librosa==0.11.0 \
  soundfile==0.13.1 \
  'requests>=2.32,<3' \
  'numpy>=2.0,<3' \
  'pandas>=2.2,<3' \
  'matplotlib>=3.9,<4' \
  tensorboard==2.20.0 \
  setuptools==80.9.0 \
  'ipykernel>=6.29,<7' \
  'ipywidgets>=8.1,<9' \
  'Cython>=3.0,<4' \
  'packaging>=24,<27'

"${VENV_DIR}/bin/python" -m ipykernel install --user \
  --name "${KERNEL_NAME}" \
  --display-name "${KERNEL_DISPLAY_NAME}"

echo "Creating the isolated Riva 2.26 client environment"
"${UV_BIN}" venv --managed-python --clear --python "${PYTHON_VERSION}" "${RIVA_VENV_DIR}"
retry "${UV_BIN}" pip install --python "${RIVA_VENV_DIR}/bin/python" \
  nvidia-riva-client==2.26.0 \
  jiwer==4.0.0 \
  'ipykernel>=6.29,<7' \
  'packaging>=24,<27'
"${RIVA_VENV_DIR}/bin/python" -m ipykernel install --user \
  --name "${RIVA_KERNEL_NAME}" \
  --display-name "${RIVA_KERNEL_DISPLAY_NAME}"

mkdir -p "${WORKSPACE_DIR}/.cache/huggingface" "${WORKSPACE_DIR}/artifacts"

export WORKSHOP_KERNEL_JSON="${HOME}/.local/share/jupyter/kernels/${KERNEL_NAME}/kernel.json"
export WORKSHOP_PROFILE_VALUE="${PROFILE_VALUE}"
export WORKSHOP_HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export WORKSHOP_REPOSITORY_NAME="${REPOSITORY_NAME}"
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

config_path = Path.home() / ".jupyter" / "jupyter_server_config.py"
config_path.parent.mkdir(parents=True, exist_ok=True)
start_marker = "# BEGIN OWN YOUR VOICE LAB LANDING PAGE"
end_marker = "# END OWN YOUR VOICE LAB LANDING PAGE"
repository_name = os.environ["WORKSHOP_REPOSITORY_NAME"]
block = f'''{start_marker}
from pathlib import Path as _WorkshopPath

_workshop_home = _WorkshopPath.home()
_workshop_candidates = (
    _workshop_home / "workspace" / "{repository_name}",
    _workshop_home / "{repository_name}",
)
for _workshop_root in _workshop_candidates:
    if (_workshop_root / "labs" / "00_start_here.ipynb").is_file():
        c.ServerApp.root_dir = str(_workshop_root)
        c.ServerApp.default_url = "/lab/tree/labs/00_start_here.ipynb?reset"
        break
{end_marker}'''
existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
if start_marker in existing and end_marker in existing:
    before, remainder = existing.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)
    existing = before.rstrip() + "\n" + after.lstrip("\n")
config_path.write_text(existing.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
print(f"Jupyter landing page configured in {config_path}")
PY

echo "[6/6] Verifying Python and CUDA from the lab kernel"
export WORKSHOP_TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION}"
"${VENV_DIR}/bin/python" - <<'PY'
import os
import sys

import importlib.metadata
import importlib.util

import torch

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Python 3.12 is required; detected {sys.version.split()[0]}.")

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the NVIDIA GPU. Check the Brev GPU and driver state.")

expected_cuda = os.environ["WORKSHOP_TORCH_CUDA_VERSION"]
if torch.version.cuda != expected_cuda:
    raise RuntimeError(
        f"The workshop requires the CUDA {expected_cuda} PyTorch build for this GPU; "
        f"detected {torch.version.cuda}."
    )

if importlib.metadata.version("nemo2riva") != "2.22.0":
    raise RuntimeError("The workshop requires nemo2riva 2.22.0.")

if importlib.metadata.version("tensorboard") != "2.20.0":
    raise RuntimeError("The workshop requires TensorBoard 2.20.0.")

if importlib.metadata.version("setuptools") != "80.9.0":
    raise RuntimeError("The workshop requires Setuptools 80.9.0 for TensorBoard.")

if importlib.util.find_spec("pkg_resources") is None:
    raise RuntimeError("TensorBoard requires pkg_resources from Setuptools 80.9.0.")

props = torch.cuda.get_device_properties(0)
capability = torch.cuda.get_device_capability(0)
device_arch = f"sm_{capability[0]}{capability[1]}"
try:
    cuda_smoke_value = (torch.ones(1, device="cuda") + 1).item()
    torch.cuda.synchronize()
except RuntimeError as exc:
    raise RuntimeError(
        f"The installed PyTorch wheel cannot execute a CUDA kernel on {device_arch}. "
        "Rerun setup so it can select the correct CUDA wheel for this GPU."
    ) from exc
if cuda_smoke_value != 2.0:
    raise RuntimeError(f"The CUDA smoke test returned {cuda_smoke_value!r}; expected 2.0.")
vram_gb = props.total_memory / 1024**3
if vram_gb < 14:
    raise RuntimeError(f"The labs need at least 14 GB VRAM; detected {vram_gb:.1f} GB.")

print(
    f"Ready: Python {sys.version.split()[0]}, {props.name}, "
    f"{vram_gb:.1f} GB VRAM, CUDA {torch.version.cuda}, {device_arch}"
)
PY

"${RIVA_VENV_DIR}/bin/python" - <<'PY'
import importlib.metadata
import sys

import riva.client

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"Python 3.12 is required; detected {sys.version.split()[0]}.")

version = importlib.metadata.version("nvidia-riva-client")
if version != "2.26.0":
    raise RuntimeError(f"Riva client 2.26.0 is required; detected {version}.")

print(f"Ready: Python {sys.version.split()[0]}, NVIDIA Riva client {version}")
PY

echo
echo "Setup complete. Labs 0-2 use '${KERNEL_DISPLAY_NAME}'; Lab 3 uses '${RIVA_KERNEL_DISPLAY_NAME}'."
echo "Fresh managed-Jupyter sessions open labs/00_start_here.ipynb automatically."
