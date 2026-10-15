#!/usr/bin/env bash
set -euo pipefail
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IBFM_ROOT="${IBFM_ROOT:-$APP}"
export IBFM_PYTHON="${IBFM_PYTHON:-$IBFM_ROOT/.venv/bin/python}"
export TMPDIR="$APP/.runtime/tmp" TMP="$APP/.runtime/tmp" TEMP="$APP/.runtime/tmp"
export XDG_CACHE_HOME="$APP/.runtime/cache" PIP_CACHE_DIR="$APP/.runtime/pip"
export UV_CACHE_DIR="$APP/.runtime/uv" MPLCONFIGDIR="$APP/.runtime/matplotlib"
export TORCH_HOME="$APP/.runtime/torch" TORCH_EXTENSIONS_DIR="$APP/.runtime/torch_extensions"
export HF_HOME="$APP/.runtime/huggingface" CUDA_CACHE_PATH="$APP/.runtime/cuda"
export TRITON_CACHE_DIR="$APP/.runtime/triton" TORCHINDUCTOR_CACHE_DIR="$APP/.runtime/inductor"
export MESA_SHADER_CACHE_DIR="$APP/.runtime/mesa" __GL_SHADER_DISK_CACHE_PATH="$APP/.runtime/nvidia"
export WP_CACHE_DIR="$APP/.runtime/warp" WANDB_DIR="$APP/.runtime/wandb" WANDB_MODE=disabled
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID:-0}}" MUJOCO_GL="${MUJOCO_GL:-egl}" MUJOCO_EGL_DEVICE_ID=0
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME"
export PYTHONPATH="$IBFM_ROOT/repo/ibfm${PYTHONPATH:+:$PYTHONPATH}"
cd "$IBFM_ROOT/repo/ibfm"
exec "$@"
