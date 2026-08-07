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
RMIR_NAME="${RIVA_RMIR_NAME:-own_your_voice_asr.rmir}"

if [[ ! -f "${NEMO_MODEL}" ]]; then
  echo "NeMo model not found: ${NEMO_MODEL}" >&2
  echo "Run Lab 2 first, or set NEMO_MODEL to a full .nemo checkpoint." >&2
  exit 1
fi
if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is required to pull the NVIDIA ASR NIM container." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
DOCKER_CONFIG_DIR="$(mktemp -d /tmp/own-your-voice-riva-build.XXXXXX)"
cleanup() {
  rm -rf "${DOCKER_CONFIG_DIR}"
}
trap cleanup EXIT

printf '%s' "${NGC_API_KEY}" | docker --config "${DOCKER_CONFIG_DIR}" login \
  nvcr.io --username '$oauthtoken' --password-stdin

echo "Building ${OUTPUT_DIR}/${RMIR_NAME} from ${NEMO_MODEL}"
docker --config "${DOCKER_CONFIG_DIR}" run --rm --gpus '"device=0"' \
  --entrypoint riva-build \
  --shm-size=8G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "${NEMO_MODEL}:/servicemaker-dev/model.nemo:ro" \
  --volume "${OUTPUT_DIR}:/servicemaker-dev/output" \
  "${ASR_NIM_IMAGE}" \
  --config-path=pkg://servicemaker.configs.asr \
  --config-name=offline \
  "output_path=/servicemaker-dev/output/${RMIR_NAME}:${MODEL_KEY}" \
  "source_path=[{path:/servicemaker-dev/model.nemo,nemo2riva:{format:onnx,onnx_opset:19,max_dim:1000,key:${MODEL_KEY}}}]" \
  offline=true \
  return_separate_utterances=true \
  decoder=greedy \
  greedy_decoder.asr_model_delay=-1 \
  chunk_size=4.8 \
  left_padding_size=1.6 \
  right_padding_size=1.6 \
  language_code=nl-NL \
  "name=${PIPELINE_NAME}" \
  force=true

test -s "${OUTPUT_DIR}/${RMIR_NAME}"
echo "Created ${OUTPUT_DIR}/${RMIR_NAME}"
