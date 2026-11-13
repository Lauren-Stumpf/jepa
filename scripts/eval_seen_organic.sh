#!/usr/bin/env bash
# E07 (hypothesis H1): evaluate the 18 organic 10% models (E01) on frames of the TRAINING
# molecules, using the same command as the test-set evaluation (results/eval/test/run_eval_test.sh).
#   labelled  positions 0, 40, 80, ...  in data/organic8/train.xyz (in the 10% labelled subset,
#             which is every frame at a position divisible by 10: data/make_fraction_subset.py)
#   unused    positions 5, 45, 85, ...  (never in the labelled subset)
# 2,100 frames each, spread over all 168 training molecules. Results go to
# results/eval/seen/<run>_<set>.{csv,log}; an existing result is never redone.
#   cd /storage3/lauren/code/jepa && tmux new-session -d -s eval_seen "bash scripts/eval_seen_organic.sh <GPU> >> results/eval/seen/queue.log 2>&1"
GPU=${1:?give the GPU index}
cd /storage3/lauren/code/jepa || exit 1
source env_train.sh >/dev/null 2>&1   # before set -u: activation scripts may read unset variables
set -u
python -c "import metatomic.torch, torch" || { echo "training environment did not load, nothing evaluated"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
OUT=results/eval/seen
mkdir -p "$OUT"

failed=0
for s in 0 1 2; do
  for r in flashmd_organic8_scratch_10pct_s$s $(for v in plain normtarget resid proj residrel; do echo jepa_finetune_${v}_organic8_10pct_s$s; done); do
    if [ ! -f "results/runs/$r/model.pt" ]; then echo "  $r: no model.pt, skipped"; failed=$((failed + 1)); continue; fi
    for set in labelled unused; do
      off=0; [ "$set" = unused ] && off=5
      o=$OUT/${r}_$set.csv
      [ -f "$o" ] && continue
      echo "$(date '+%F %T') $r on $set frames"
      if python eval/eval_onestep.py data/organic8/train.xyz --stride_fs 8 --every 40 --offset $off --n 2100 \
          --model "results/runs/$r/model.pt" --scaled_targets --config configs/flashmd_organic8_scratch.yaml \
          --out "$o.part" > "$OUT/${r}_$set.log" 2>&1; then
        mv "$o.part" "$o"
      else
        rm -f "$o.part"; failed=$((failed + 1)); echo "  failed, see $OUT/${r}_$set.log"
      fi
    done
  done
done
echo "$(date '+%F %T') finished: $failed problems (a missing model counts once, a failed evaluation once per set)"
exit $(( failed > 0 ))
