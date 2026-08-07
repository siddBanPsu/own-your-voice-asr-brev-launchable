#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_REPOSITORY="${ROOT_DIR}/triton/model_repository"
CONTAINER_NAME="own-your-voice-triton"
IMAGE="nvcr.io/nvidia/tritonserver:26.06-py3"

if [[ ! -f "${MODEL_REPOSITORY}/parakeet_ctc/1/model.onnx" ]]; then
  echo "Missing ONNX model. Complete the export section in Lab 3 first." >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "${CONTAINER_NAME} is already running."
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker start "${CONTAINER_NAME}"
else
  docker run --detach \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc host \
    --publish 8000:8000 \
    --publish 8001:8001 \
    --publish 8002:8002 \
    --volume "${MODEL_REPOSITORY}:/models:ro" \
    "${IMAGE}" \
    tritonserver --model-repository=/models --strict-model-config=true
fi

echo "Waiting for Triton readiness on http://localhost:8000/v2/health/ready"
for _ in $(seq 1 90); do
  if curl --silent --fail http://localhost:8000/v2/health/ready >/dev/null; then
    echo "Triton is ready."
    exit 0
  fi
  sleep 2
done

echo "Triton did not become ready. Inspect logs with: docker logs ${CONTAINER_NAME}" >&2
exit 1
