#!/usr/bin/env bash
set -euo pipefail
ROOT="${IBFM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="${1:?supply pair suffix}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl
exec "$HERE/../runtime.sh" "${IBFM_PYTHON:-$ROOT/.venv/bin/python}" "$HERE/render_height_probe.py" \
  "$HERE/outputs/standing_height_live_${STAMP}" \
  --compare "$HERE/outputs/standing_height_frozen_${STAMP}"
