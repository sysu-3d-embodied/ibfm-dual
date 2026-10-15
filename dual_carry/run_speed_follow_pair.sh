#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${IBFM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl
for mode in live frozen-speed; do
  "$HERE/../runtime.sh" "${IBFM_PYTHON:-$ROOT/.venv/bin/python}" "$HERE/speed_follow_probe.py" \
    --tag "speed_follow_${mode}_${STAMP}" --mode "$mode" "${@:2}"
done
