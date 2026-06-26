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
python3 scripts/validate_fixed_topk_compact_out.py --mode correctness --variant tiled --hidden-tile 256
python3 scripts/validate_fixed_topk_compact_out.py --mode benchmark --variant tiled --hidden-tile 256 --seq-lens 8192,8192 --hidden-size 4096 --iters 100 --warmup 20
