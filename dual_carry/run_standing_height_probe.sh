#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${IBFM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl
exec "$HERE/../runtime.sh" "${IBFM_PYTHON:-$ROOT/.venv/bin/python}" "$HERE/standing_height_probe.py" "$@"
