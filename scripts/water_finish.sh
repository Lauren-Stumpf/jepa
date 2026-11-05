#!/usr/bin/env bash
# Water track, from finished i-PI runs to checked training pairs, in one go.
# Stops at the first failing step. Run from the repo root:
#   bash scripts/water_finish.sh
set -euo pipefail
ROOT=${ROOT:-/storage3/lauren/data/water}
N=${N:-280}
step() { echo; echo "=== $1"; }

step "0. all $N runs present and complete"
missing=$(for s in $(seq 1 $N); do grep -qs "done:" $ROOT/log_$(printf %03d $s).txt || echo $s; done)
if [ -n "$missing" ]; then
  echo "missing or incomplete runs: $(echo $missing | tr '\n' ' ')"
  echo "running them now (20 in parallel)"
  for s in $missing; do d=$ROOT/run_$(printf %03d $s); grep -qs "done:" $ROOT/log_$(printf %03d $s).txt || rm -rf $d; done
  echo $missing | tr ' ' '\n' | xargs -P 20 -I{} sh -c "bash data/water/run_one.sh {} $ROOT/run_\$(printf %03d {}) > $ROOT/log_\$(printf %03d {}).txt 2>&1"
  still=$(for s in $(seq 1 $N); do grep -qs "done:" $ROOT/log_$(printf %03d $s).txt || echo $s; done)
  [ -z "$still" ] || { echo "still incomplete after rerun: $still"; exit 1; }
fi
echo "all $N runs complete"

step "1. the seven checks on every run (about a minute per run)"
fails=0
for d in $ROOT/run_*; do
  n=$(python checks/check_water_run.py $d | grep -c FAIL || true)
  [ "$n" -eq 0 ] || { echo "FAIL in $d ($n checks)"; fails=$((fails+1)); }
done
[ "$fails" -eq 0 ] || { echo "$fails runs failed checks; inspect them before parsing"; exit 1; }
echo "all runs pass"

step "2. parse into pairs at 1, 4 and 16 fs"
python data/water/parse_runs.py $ROOT configs/flashmd_water16_scratch.yaml --out_root data/water/pairs

step "3. check the pairs against the raw trajectories"
for s in 1 4 16; do python checks/check_water_pairs.py data/water/pairs/${s}fs $ROOT | tail -1 | grep -q "ALL PASS" || { echo "pairs check failed for ${s} fs"; exit 1; }; done
echo "all pairs checks pass"

step "done: commit the manifests, then launch the water baselines (one GPU each; check nvidia-smi first; source env_train.sh):"
echo "  git add data/water/pairs/*/manifest.json && git commit -m 'water pairs prepared' && git push"
echo "  CUDA_VISIBLE_DEVICES=1 mtt train configs/flashmd_water16_scratch.yaml -o results/models/flashmd_water16_scratch.pt"
echo "  CUDA_VISIBLE_DEVICES=2 mtt train configs/flashmd_water4_scratch.yaml  -o results/models/flashmd_water4_scratch.pt"
echo "  CUDA_VISIBLE_DEVICES=3 mtt train configs/flashmd_water1_scratch.yaml  -o results/models/flashmd_water1_scratch.pt"
