#!/usr/bin/env bash
# Test-set evaluation of the organic 10% runs of E12, E14 and E15, as they finish.
# The command is the one used for every organic test result so far (E01, E03; see
# experiments/E07/card.md): data/organic8/test.xyz, every 5th frame from frame 0, 2,100 frames,
# mass-scaled targets, the organic config's masses.
# Runs that have not finished (no model.pt) are looked at again on the next pass, every
# 10 min, until every listed run is tested or has failed 3 times, or 72 h have passed.
# A result that already exists is never redone.
#   cd /storage3/lauren/code/jepa && tmux new-session -d -s eval_organic "bash scripts/eval_test_organic_queue.sh <GPU> >> results/eval/test/queue_organic.log 2>&1"
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
    echo e12_finetune_plain_gap100_organic8_10pct_s$s
    echo e14_finetune_supervised_organic8_10pct_s$s
    for p in plain nopred; do for e in 20 200; do echo e15_finetune_${p}_same10_ep${e}_organic8_10pct_s$s; done; done
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
    if python eval/eval_onestep.py data/organic8/test.xyz --stride_fs 8 --every 5 --offset 0 --n 2100 \
        --model "results/runs/$r/model.pt" --scaled_targets --config configs/flashmd_organic8_scratch.yaml \
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
