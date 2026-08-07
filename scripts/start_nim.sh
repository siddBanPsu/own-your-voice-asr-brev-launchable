#!/bin/bash
set -euo pipefail

CONTAINER_ID="${NIM_CONTAINER_ID:-parakeet-0-6b-ctc-en-us}"
NIM_TAGS_SELECTOR="${NIM_TAGS_SELECTOR:-name=parakeet-0-6b-ctc-en-us,bs=1,mode=ofl,diarizer=disabled,vad=default}"
NIM_IMAGE="${NIM_IMAGE:-nvcr.io/nim/nvidia/${CONTAINER_ID}:latest}"
LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-${HOME}/.cache/nim}"
READY_URL="http://localhost:9000/v1/health/ready"
READY_TIMEOUT_SECONDS="${NIM_READY_TIMEOUT_SECONDS:-1800}"

if curl --silent --fail "${READY_URL}" >/dev/null 2>&1; then
  echo "Parakeet CTC 0.6B NIM is already ready at ${READY_URL}."
  exit 0
fi

if docker inspect "${CONTAINER_ID}" >/dev/null 2>&1; then
  echo "Container '${CONTAINER_ID}' exists but is not ready." >&2
  echo "Inspect it with: docker logs ${CONTAINER_ID}" >&2
  exit 1
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is required to pull and run NVIDIA Speech NIM." >&2
  exit 1
fi

mkdir -p "${LOCAL_NIM_CACHE}"
chmod 777 "${LOCAL_NIM_CACHE}"

DOCKER_CONFIG_DIR="$(mktemp -d /tmp/own-your-voice-nim-docker.XXXXXX)"
cleanup() {
  rm -rf "${DOCKER_CONFIG_DIR}"
}
trap cleanup EXIT

printf '%s' "${NGC_API_KEY}" | docker --config "${DOCKER_CONFIG_DIR}" login \
  nvcr.io \
  --username '$oauthtoken' \
  --password-stdin

echo "Starting ${NIM_IMAGE}"
echo "Profile: ${NIM_TAGS_SELECTOR}"
docker --config "${DOCKER_CONFIG_DIR}" run --detach --rm \
  --name "${CONTAINER_ID}" \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  --env NGC_API_KEY \
  --env NIM_HTTP_API_PORT=9000 \
  --env NIM_GRPC_API_PORT=50051 \
  --env "NIM_TAGS_SELECTOR=${NIM_TAGS_SELECTOR}" \
  --publish 9000:9000 \
  --publish 50051:50051 \
  --volume "${LOCAL_NIM_CACHE}:/opt/nim/.cache" \
  "${NIM_IMAGE}"

deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
echo "Waiting for NIM readiness at ${READY_URL}. First startup can take up to 30 minutes."
while (( SECONDS < deadline )); do
  if curl --silent --fail "${READY_URL}" >/dev/null 2>&1; then
    echo "Parakeet CTC 0.6B NIM is ready."
    exit 0
  fi
  if ! docker inspect "${CONTAINER_ID}" >/dev/null 2>&1; then
    echo "The NIM container exited before becoming ready." >&2
    exit 1
  fi
  sleep 10
done

echo "NIM did not become ready within ${READY_TIMEOUT_SECONDS} seconds." >&2
docker logs --tail 100 "${CONTAINER_ID}" >&2 || true
exit 1
