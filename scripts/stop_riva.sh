#!/bin/bash
set -euo pipefail

CONTAINER_NAME="${RIVA_CONTAINER_NAME:-own-your-voice-riva-nim}"
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker stop "${CONTAINER_NAME}"
  echo "Stopped ${CONTAINER_NAME}."
else
  echo "${CONTAINER_NAME} is not running."
fi
