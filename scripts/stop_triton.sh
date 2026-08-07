#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="own-your-voice-triton"
if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker stop "${CONTAINER_NAME}"
else
  echo "${CONTAINER_NAME} is not running."
fi
