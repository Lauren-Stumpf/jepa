#!/usr/bin/env bash
# Test-set evaluation of every water 16 fs run trained on the 10% subset (E04, E05, E06).
# The command is the one used for the full-data water baseline
# (results/eval/test/run_eval_test.sh): all 1,120 test frames, mass-scaled targets.
# eval/eval_onestep.py reads only the atomic masses from --config, which are the same in
# every water config, so the full-data config is passed for all runs.
# Runs that have not finished (no model.pt) are looked at again on the next pass, every
# 10 min, until every listed run is tested or has failed 3 times, or 72 h have passed.
# A result that already exists is never redone.
#   cd /storage3/lauren/code/jepa && tmux new-session -d -s eval_water16 "bash scripts/eval_test_water16_queue.sh <GPU> >> results/eval/test/queue_water16.log 2>&1"
# A run's .log is appended to on each try; scripts/wandb_common.py test_metrics() keeps the last result in it.
GPU=${1:?give the GPU index}
MAX_TRIES=3
cd /storage3/lauren/code/jepa || exit 1
source env_train.sh >/dev/null 2>&1   # before set -u: activation scripts may read unset variables
set -u
python -c "import metatomic.torch, torch" || { echo "training environment did not load, nothing tested"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
T=results/eval/test
mkdir -p "$T"

runs() {
  for s in 0 1 2; do
    echo flashmd_water16_scratch_10pct_s$s
    for v in plain normtarget resid proj residrel; do echo jepa_finetune_${v}_water16_10pct_s$s; done
    for a in baseline plain normtarget; do echo e05_${a}_water16_10pct_1000ep_s$s; done
    echo e06_finetune_regonly_water16_10pct_s$s
  done
}

for pass in $(seq 1 432); do
  waiting=0; failed=0
  for r in $(runs); do
    o=$T/$r.csv
    [ -f "$o" ] && continue
    tries=$(cat "$T/$r.tries" 2>/dev/null || echo 0)
    if [ "$tries" -ge "$MAX_TRIES" ]; then failed=$((failed + 1)); continue; fi
    if [ ! -f "results/runs/$r/model.pt" ]; then waiting=$((waiting + 1)); continue; fi
    echo "$(date '+%F %T') testing $r (try $((tries + 1)))"
    if python eval/eval_onestep.py data/water/pairs/16fs/test.xyz --stride_fs 16 --every 1 --offset 0 --n 1120 \
        --model "results/runs/$r/model.pt" --scaled_targets --config configs/flashmd_water16_scratch.yaml \
        --out "$o.part" >> "$T/$r.log" 2>&1; then
      mv "$o.part" "$o"; rm -f "$T/$r.tries"
    else
      rm -f "$o.part"; echo $((tries + 1)) > "$T/$r.tries"
      echo "  failed, see $T/$r.log"
      [ $((tries + 1)) -ge "$MAX_TRIES" ] && failed=$((failed + 1)) || waiting=$((waiting + 1))
    fi
  done
  if [ "$waiting" -eq 0 ]; then
    echo "$(date '+%F %T') finished: $failed runs failed $MAX_TRIES times (see the .log and .tries files), the rest tested"
    exit $(( failed > 0 ))
  fi
  [ "$pass" -lt 432 ] && sleep 600
done
echo "$(date '+%F %T') stopped after 72 h: $waiting runs not tested yet, $failed failed"
exit 1
