#!/usr/bin/env bash
# Experimental shared-box controller. See assessment.json; phase_complete is
# not sufficient to call a run stable or successful.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${IBFM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
export IBFM_ROOT="$ROOT"
exec "$HERE/../runtime.sh" "${IBFM_PYTHON:-$ROOT/.venv/bin/python}" "$HERE/mission.py" \
  --tag "revo2_$(date +%Y%m%d_%H%M%S)" --seconds 180 \
  --canonical-symmetry --state-bank replay_bank_67974_90655a91.pt \
  --speed .10 --travel 5.0 --travel-direction forward --hz 10 --physics-hz 200 \
  --filtered-context --transport-z-angle .15 --coordinated-motion --axial-carry \
  --goal-steering --box-heading-goal --steering-blend .5 "$@"
