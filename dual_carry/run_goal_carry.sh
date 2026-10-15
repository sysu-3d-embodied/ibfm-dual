#!/usr/bin/env bash
# Full 5 m v18 experiment with contact-triggered retreat during lowering.
# Release only after stable ground contact; no height-triggered hand opening.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec bash "$HERE/run_revo2.sh" --seconds 240 --grip-x .19 \
  --placement-guide placement_guide_v6_static.pt --guide-mix .85 \
  --guide-height-lead .12 --placement-xy-feedback --compliant-release --contact-retreat --mass-adaptive-grasp "$@"
