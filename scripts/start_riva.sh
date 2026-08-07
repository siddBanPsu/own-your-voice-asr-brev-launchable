#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RIVA_VERSION="${RIVA_VERSION:-2.26.0}"
RIVA_IMAGE="${RIVA_IMAGE:-nvcr.io/nvidia/riva/riva-speech:${RIVA_VERSION}}"
MODEL_KEY="${RIVA_MODEL_KEY:-tlt_encode}"
RMIR_FILE="${RIVA_RMIR_FILE:-${ROOT_DIR}/artifacts/riva/own_your_voice_asr.rmir}"
MODEL_ROOT="${RIVA_MODEL_ROOT:-${ROOT_DIR}/artifacts/riva/runtime}"
CONTAINER_NAME="${RIVA_CONTAINER_NAME:-own-your-voice-riva}"
READY_TIMEOUT_SECONDS="${RIVA_READY_TIMEOUT_SECONDS:-1800}"

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "${CONTAINER_NAME} already exists. Stop it before starting another Riva server." >&2
  exit 1
fi
if [[ ! -f "${RMIR_FILE}" ]]; then
  echo "RMIR not found: ${RMIR_FILE}" >&2
  echo "Run scripts/build_riva_rmir.sh first." >&2
  exit 1
fi
if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is required to pull the NVIDIA Riva container." >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}/models"
DOCKER_CONFIG_DIR="$(mktemp -d /tmp/own-your-voice-riva-run.XXXXXX)"
cleanup() {
  rm -rf "${DOCKER_CONFIG_DIR}"
}
trap cleanup EXIT

printf '%s' "${NGC_API_KEY}" | docker --config "${DOCKER_CONFIG_DIR}" login \
  nvcr.io --username '$oauthtoken' --password-stdin

echo "Optimizing the RMIR for this GPU. First deployment can take many minutes."
docker --config "${DOCKER_CONFIG_DIR}" run --rm --gpus '"device=0"' \
  --entrypoint riva-deploy \
  --volume "${RMIR_FILE}:/servicemaker-dev/model.rmir:ro" \
  --volume "${MODEL_ROOT}:/data" \
  "${RIVA_IMAGE}" \
  -f "/servicemaker-dev/model.rmir:${MODEL_KEY}" /data/models

docker --config "${DOCKER_CONFIG_DIR}" run --detach --rm \
  --name "${CONTAINER_NAME}" \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --init \
  --shm-size=1G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "${MODEL_ROOT}:/data" \
  --publish 50051:50051 \
  --env CUDA_VISIBLE_DEVICES=0 \
  "${RIVA_IMAGE}" \
  start-riva --riva-uri=0.0.0.0:50051 \
  --asr_service=true --nlp_service=false --tts_service=false

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if docker logs "${CONTAINER_NAME}" 2>&1 | grep -Eq \
    "Riva server is ready|Riva Conversational AI Server listening"; then
    echo "Riva ASR is ready at localhost:50051."
    exit 0
  fi
  if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "The Riva container exited before becoming ready." >&2
    exit 1
  fi
  sleep 10
done

echo "Riva did not become ready within ${READY_TIMEOUT_SECONDS} seconds." >&2
docker logs --tail 100 "${CONTAINER_NAME}" >&2 || true
exit 1
