#!/usr/bin/env bash
# E22 (experiments/E22/card.md): rollouts of water models, then the exact-dynamics parts and the
# analysis. Runs on one node, one GPU for the rollouts and --workers CPU cores for the exact parts.
# Stops at the first failed check. Each output that already exists is not redone.
#   cd /storage3/lauren/code/jepa && tmux new-session -d -s e22 "bash scripts/run_e22.sh <GPU> 16 >> results/eval/e22/run.log 2>&1"
GPU=${1:?give the GPU index}
WORKERS=${2:-16}
cd /storage3/lauren/code/jepa || exit 1
source env_train.sh >/dev/null 2>&1   # before set -u: activation scripts may read unset variables
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$GPU
O=results/eval/e22
mkdir -p "$O"
CFG=configs/flashmd_water16_scratch.yaml   # every water config has the same masses and time step
PRIMARY=flashmd_water16_scratch
WORSE=flashmd_water16_scratch_10pct_s0
OPTIONAL="e16_baseline_water16_10pct_1400ep_s0 e16_baseline_water16_10pct_1400ep_s1 e16_baseline_water16_10pct_1400ep_s2
e05_plain_water16_10pct_1000ep_s0 e05_plain_water16_10pct_1000ep_s1 e05_plain_water16_10pct_1000ep_s2"
say() { echo "$(date '+%F %T') $*"; }

say "checks"
OMP_NUM_THREADS=1 python eval/qtip4pf_ase.py /storage3/lauren/data/water/run_011 "$CFG"
python eval/rollout_water.py results/runs/$PRIMARY/model.pt "$CFG" --check_onestep data/water/pairs/16fs/test.xyz
python eval/rollout_water.py results/runs/$PRIMARY/model.pt "$CFG" --check_wrap 10

roll() {  # name, output, extra flags
  [ -f "$2" ] && { say "exists: $2"; return; }
  say "rollout $1 -> $2 $3"
  python eval/rollout_water.py "results/runs/$1/model.pt" "$CFG" --out "$2" $3
}
roll $PRIMARY $O/rollout_primary.npz ""
roll $WORSE $O/rollout_worse.npz ""
roll $PRIMARY $O/rollout_drift.npz "--momentum_scale 1.00002"
roll $PRIMARY $O/rollout_fail.npz "--momentum_scale 1.01"
for m in $OPTIONAL; do roll $m $O/rollout_$m.npz ""; done

exact() {  # rollout, output
  [ -f "$2" ] && { say "exists: $2"; return; }
  say "exact parts for $1"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python eval/rollout_exact.py "$1" --out "$2" --workers "$WORKERS"
}
exact $O/rollout_primary.npz $O/exact_primary.npz
for m in $OPTIONAL; do exact $O/rollout_$m.npz $O/exact_$m.npz; done

say "analysis"
python eval/rollout_analysis.py --rollout $O/rollout_primary.npz --exact $O/exact_primary.npz \
  --worse $O/rollout_worse.npz --drift $O/rollout_drift.npz --fail $O/rollout_fail.npz \
  --checkpoint results/runs/$PRIMARY/model.ckpt --config "$CFG" --out_dir $O/primary
for m in $OPTIONAL; do
  python eval/rollout_analysis.py --rollout $O/rollout_$m.npz --exact $O/exact_$m.npz \
    --checkpoint results/runs/$m/model.ckpt --config "$CFG" --out_dir $O/$m
done
say "E22 finished"
