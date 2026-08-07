#!/bin/bash
set -euo pipefail

CONTAINER_ID="${NIM_CONTAINER_ID:-parakeet-0-6b-ctc-en-us}"

if docker inspect "${CONTAINER_ID}" >/dev/null 2>&1; then
  docker stop "${CONTAINER_ID}"
  echo "Stopped ${CONTAINER_ID}."
else
  echo "${CONTAINER_ID} is not running."
fi
