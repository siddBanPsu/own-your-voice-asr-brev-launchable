#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_NIM_CONTAINER_ID="${ASR_NIM_CONTAINER_ID:-parakeet-0-6b-ctc-en-us}"
ASR_NIM_TAG="${ASR_NIM_TAG:-3.1.0}"
ASR_NIM_IMAGE="${ASR_NIM_IMAGE:-nvcr.io/nim/nvidia/${ASR_NIM_CONTAINER_ID}:${ASR_NIM_TAG}}"
MODEL_KEY="${RIVA_MODEL_KEY:-tlt_encode}"
PIPELINE_NAME="${RIVA_PIPELINE_NAME:-own-your-voice-nl-asr-offline}"
NEMO_MODEL="${NEMO_MODEL:-${ROOT_DIR}/artifacts/parakeet-ctc-0.6b-nl.nemo}"
OUTPUT_DIR="${RIVA_OUTPUT_DIR:-${ROOT_DIR}/artifacts/riva}"
RIVA_MODEL="${RIVA_MODEL:-${OUTPUT_DIR}/own_your_voice_asr.riva}"
RMIR_NAME="${RIVA_RMIR_NAME:-own_your_voice_asr.rmir}"
NEMO2RIVA_BIN="${NEMO2RIVA_BIN:-${HOME}/.venvs/own-your-voice-asr/bin/nemo2riva}"
ASR_PYTHON="${ASR_PYTHON:-${HOME}/.venvs/own-your-voice-asr/bin/python}"
SAVE_INTERMEDIATE_ONNX="${SAVE_INTERMEDIATE_ONNX:-0}"
ONNX_MODEL="${ONNX_MODEL:-${ROOT_DIR}/artifacts/onnx/parakeet-ctc-0.6b-nl.onnx}"
ONNX_OPSET="${ONNX_OPSET:-19}"
RIVA_MIN_FREE_GB="${RIVA_MIN_FREE_GB:-20}"
RIVA_EXPORT_TMP_DIR="${RIVA_EXPORT_TMP_DIR:-${OUTPUT_DIR}/tmp}"

if [[ ! -f "${NEMO_MODEL}" ]]; then
  echo "NeMo model not found: ${NEMO_MODEL}" >&2
  echo "Run Lab 2 first, or set NEMO_MODEL to a full .nemo checkpoint." >&2
  exit 1
fi
if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is required to pull the NVIDIA ASR NIM container." >&2
  exit 1
fi
if [[ ! -x "${NEMO2RIVA_BIN}" ]]; then
  echo "nemo2riva was not found at ${NEMO2RIVA_BIN}." >&2
  echo "Rerun launchable/setup.sh, or set NEMO2RIVA_BIN to the executable." >&2
  exit 1
fi
if [[ "${SAVE_INTERMEDIATE_ONNX}" != "0" && "${SAVE_INTERMEDIATE_ONNX}" != "1" ]]; then
  echo "SAVE_INTERMEDIATE_ONNX must be 0 or 1; received ${SAVE_INTERMEDIATE_ONNX}." >&2
  exit 1
fi
if ! [[ "${RIVA_MIN_FREE_GB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "RIVA_MIN_FREE_GB must be a positive integer; received ${RIVA_MIN_FREE_GB}." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${RIVA_EXPORT_TMP_DIR}"
required_kb=$((RIVA_MIN_FREE_GB * 1024 * 1024))
available_kb="$(df -Pk "${OUTPUT_DIR}" | awk 'NR == 2 {print $4}')"
if [[ ! "${available_kb}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine free disk space for ${OUTPUT_DIR}." >&2
  exit 1
fi
available_gb=$((available_kb / 1024 / 1024))
echo "Riva build disk preflight: ${available_gb} GB free; ${RIVA_MIN_FREE_GB} GB required."
if (( available_kb < required_kb )); then
  echo "Insufficient disk space for NeMo -> .riva -> RMIR packaging." >&2
  df -h "${OUTPUT_DIR}" >&2
  echo "Inspect usage with: docker system df" >&2
  echo "Inspect workshop files with: du -sh '${ROOT_DIR}/artifacts' '${HOME}/.cache/nim' '${HOME}/.cache/huggingface' 2>/dev/null" >&2
  echo "Remove only confirmed-unused images or failed artifacts, then rerun Lab 3." >&2
  exit 1
fi

if [[ "${SAVE_INTERMEDIATE_ONNX}" == "1" ]]; then
  if [[ ! -x "${ASR_PYTHON}" ]]; then
    echo "NeMo Python was not found at ${ASR_PYTHON}." >&2
    echo "Rerun launchable/setup.sh, or set ASR_PYTHON to the NeMo environment." >&2
    exit 1
  fi
  echo "Saving a standalone ONNX copy to ${ONNX_MODEL}"
  "${ASR_PYTHON}" "${ROOT_DIR}/scripts/export_nemo_onnx.py" \
    --nemo-model "${NEMO_MODEL}" \
    --output "${ONNX_MODEL}" \
    --opset "${ONNX_OPSET}"
  test -s "${ONNX_MODEL}"
  echo "Saved standalone ONNX artifact: ${ONNX_MODEL}"
fi

echo "Exporting ${NEMO_MODEL} to ${RIVA_MODEL}"
TMPDIR="${RIVA_EXPORT_TMP_DIR}" CUDA_VISIBLE_DEVICES=0 "${NEMO2RIVA_BIN}" \
  --key "${MODEL_KEY}" \
  --onnx-opset "${ONNX_OPSET}" \
  --max-dim 1000 \
  --out "${RIVA_MODEL}" \
  "${NEMO_MODEL}"
test -s "${RIVA_MODEL}"

DOCKER_CONFIG_DIR="$(mktemp -d /tmp/own-your-voice-riva-build.XXXXXX)"
cleanup() {
  rm -rf "${DOCKER_CONFIG_DIR}"
}
trap cleanup EXIT

printf '%s' "${NGC_API_KEY}" | docker --config "${DOCKER_CONFIG_DIR}" login \
  nvcr.io --username '$oauthtoken' --password-stdin

echo "Building ${OUTPUT_DIR}/${RMIR_NAME} from ${RIVA_MODEL}"
docker --config "${DOCKER_CONFIG_DIR}" run --rm --gpus '"device=0"' \
  --entrypoint riva-build \
  --shm-size=8G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "${RIVA_MODEL}:/servicemaker-dev/model.riva:ro" \
  --volume "${OUTPUT_DIR}:/servicemaker-dev/output" \
  "${ASR_NIM_IMAGE}" \
  speech_recognition \
  -f \
  "/servicemaker-dev/output/${RMIR_NAME}:${MODEL_KEY}" \
  "/servicemaker-dev/model.riva:${MODEL_KEY}" \
  --offline \
  "--name=${PIPELINE_NAME}" \
  --return_separate_utterances=True \
  --featurizer.use_utterance_norm_params=False \
  --featurizer.precalc_norm_time_steps=0 \
  --featurizer.precalc_norm_params=False \
  --ms_per_timestep=80 \
  --endpointing.residue_blanks_at_start=-16 \
  --nn.use_trt_fp32 \
  --decoder_type=greedy \
  --greedy_decoder.asr_model_delay=-1 \
  --language_code=nl-NL

test -s "${OUTPUT_DIR}/${RMIR_NAME}"
echo "Created ${OUTPUT_DIR}/${RMIR_NAME}"
