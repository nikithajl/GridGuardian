#!/usr/bin/env bash

set -euo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [[ -z "${PING_URL}" ]]; then
  echo "usage: ./scripts/validate-submission.sh <ping_url> [repo_dir]"
  exit 1
fi

echo "[check] pinging ${PING_URL}"
curl -fsSL "${PING_URL}" > /dev/null

echo "[check] running local python validator"
(
  cd "${REPO_DIR}"
  python validate_submission.py
)

echo "[check] building docker image"
docker build -t gridguardian-validator "${REPO_DIR}" > /dev/null

echo "[ok] submission checks completed"

