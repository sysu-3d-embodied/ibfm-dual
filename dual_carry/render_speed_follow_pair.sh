#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${IBFM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
STAMP="${1:?Supply pair suffix}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl
exec "$HERE/../runtime.sh" "${IBFM_PYTHON:-$ROOT/.venv/bin/python}" "$HERE/render_speed_follow.py" \
  "$HERE/outputs/speed_follow_live_${STAMP}" \
  "$HERE/outputs/speed_follow_frozen-speed_${STAMP}"
