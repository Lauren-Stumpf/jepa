#!/usr/bin/env bash
# Regenerate the water data with equilibrated starts (decisions.md, 2026-11-05).
# Step 1 runs the 100 ps pilot and prints its relaxation curve, then STOPS so the
# per-run equilibration length can be checked against it. Step 2 (with GO=1) moves the
# first-generation data aside (kept, not deleted) and makes 280 new runs.
#   bash scripts/water_regenerate.sh            # pilot + curve (~10 min)
#   GO=1 bash scripts/water_regenerate.sh       # the 280 runs (~1.5 h at 20 in parallel)
set -euo pipefail
REPO=/storage3/lauren/code/jepa
cd "$REPO"
PILOT=/storage3/lauren/data/water_pilot
ROOT=/storage3/lauren/data/water
if [ "${GO:-0}" != 1 ]; then
  [ -f "$PILOT/water64_equilibrated.xyz" ] || bash data/water/run_pilot.sh "$PILOT"
  python checks/water_equil_curve.py "$PILOT/pilot.out"
  echo
  echo "Paste the above. If the relaxation time is well under ~8 ps (so 25 ps per run leaves"
  echo "a gap far below 1 K), run:  GO=1 bash scripts/water_regenerate.sh"
  exit 0
fi
[ -f "$PILOT/water64_equilibrated.xyz" ] || { echo "run the pilot first"; exit 1; }
if [ -d "$ROOT" ] && [ ! -d "${ROOT}_v1_5ps_unequilibrated" ]; then
  mv "$ROOT" "${ROOT}_v1_5ps_unequilibrated"
  echo "first-generation runs kept at ${ROOT}_v1_5ps_unequilibrated"
fi
mkdir -p "$ROOT"
seq 1 280 | xargs -P 20 -I{} sh -c "d=$ROOT/run_\$(printf %03d {}); [ -d \$d ] && exit 0; bash data/water/run_one.sh {} \$d > $ROOT/log_\$(printf %03d {}).txt 2>&1"
echo "generation finished; now: bash scripts/water_finish.sh"
