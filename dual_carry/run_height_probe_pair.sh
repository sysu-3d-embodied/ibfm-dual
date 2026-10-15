#!/usr/bin/env bash
# Sequential matched trials on one GPU, with distinct output directories.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
bash "$HERE/run_standing_height_probe.sh" --tag "standing_height_live_${STAMP}" --mode live
bash "$HERE/run_standing_height_probe.sh" --tag "standing_height_frozen_${STAMP}" --mode frozen-leader
