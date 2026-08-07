#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_NIM_CONTAINER_ID="${ASR_NIM_CONTAINER_ID:-parakeet-0-6b-ctc-en-us}"
ASR_NIM_TAG="${ASR_NIM_TAG:-3.1.0}"
ASR_NIM_IMAGE="${ASR_NIM_IMAGE:-nvcr.io/nim/nvidia/${ASR_NIM_CONTAINER_ID}:${ASR_NIM_TAG}}"
NIM_TAGS_SELECTOR="${NIM_TAGS_SELECTOR:-name=parakeet-0-6b-ctc-en-us,mode=ofl}"
MODEL_KEY="${RIVA_MODEL_KEY:-tlt_encode}"
RMIR_FILE="${RIVA_RMIR_FILE:-${ROOT_DIR}/artifacts/riva/own_your_voice_asr.rmir}"
MODEL_ROOT="${RIVA_MODEL_ROOT:-${ROOT_DIR}/artifacts/riva/runtime}"
NIM_EXPORT_DIR="${RIVA_NIM_EXPORT_DIR:-${ROOT_DIR}/artifacts/riva/nim_export}"
CONTAINER_NAME="${RIVA_CONTAINER_NAME:-own-your-voice-riva-nim}"
READY_URL="http://localhost:9000/v1/health/ready"
READY_TIMEOUT_SECONDS="${RIVA_READY_TIMEOUT_SECONDS:-1800}"

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  if curl --silent --fail "${READY_URL}" >/dev/null 2>&1; then
    echo "The custom Riva ASR NIM is already ready at ${READY_URL}."
    exit 0
  fi
  echo "${CONTAINER_NAME} exists but is not ready. Inspect it with: docker logs ${CONTAINER_NAME}" >&2
  exit 1
fi
if curl --silent --fail "${READY_URL}" >/dev/null 2>&1; then
  echo "Port 9000 is already serving another NIM. Stop Lab 1 before starting Lab 3." >&2
  exit 1
fi
if [[ ! -f "${RMIR_FILE}" ]]; then
  echo "RMIR not found: ${RMIR_FILE}" >&2
  echo "Run scripts/build_riva_rmir.sh first." >&2
  exit 1
fi
if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is required to pull the NVIDIA ASR NIM container." >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}/models" "${NIM_EXPORT_DIR}"
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
  --shm-size=8GB \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "${RMIR_FILE}:/servicemaker-dev/model.rmir:ro" \
  --volume "${MODEL_ROOT}:/data" \
  "${ASR_NIM_IMAGE}" \
  -f "/servicemaker-dev/model.rmir:${MODEL_KEY}" /data/models

echo "Packaging the optimized repository for custom NIM serving."
tar -C "${MODEL_ROOT}/models" -czf "${NIM_EXPORT_DIR}/custom_model.tar.gz" .
test -s "${NIM_EXPORT_DIR}/custom_model.tar.gz"

docker --config "${DOCKER_CONFIG_DIR}" run --detach --rm \
  --name "${CONTAINER_NAME}" \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --init \
  --shm-size=8GB \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "${NIM_EXPORT_DIR}:/opt/nim/export:ro" \
  --publish 9000:9000 \
  --publish 50051:50051 \
  --env NGC_API_KEY \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env "NIM_TAGS_SELECTOR=${NIM_TAGS_SELECTOR}" \
  --env NIM_DISABLE_MODEL_DOWNLOAD=true \
  --env NIM_HTTP_API_PORT=9000 \
  --env NIM_GRPC_API_PORT=50051 \
  --env NIM_EXPORT_PATH=/opt/nim/export \
  "${ASR_NIM_IMAGE}"

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
echo "Waiting for the custom Riva ASR NIM at ${READY_URL}."
while (( SECONDS < deadline )); do
  if curl --silent --fail "${READY_URL}" >/dev/null 2>&1; then
    echo "Custom Riva ASR NIM is ready at HTTP localhost:9000 and gRPC localhost:50051."
    exit 0
  fi
  if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "The custom Riva ASR NIM exited before becoming ready." >&2
    exit 1
  fi
  sleep 10
done

echo "Riva did not become ready within ${READY_TIMEOUT_SECONDS} seconds." >&2
docker logs --tail 100 "${CONTAINER_NAME}" >&2 || true
exit 1
