#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ERROR: ASCEND_HOME_PATH is not set" >&2
    exit 1
fi

source "${ASCEND_HOME_PATH}/set_env.sh"

cmake -S . -B build
cmake --build build -j4
python3 scripts/test_torch.py

