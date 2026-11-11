#!/usr/bin/env bash
# Start (or continue) the water 16 fs pretraining experiment on this node, mirroring the
# organic one. Safe to run any number of times: finished runs are skipped, running ones
# left alone, and fine-tunes wait until their pretraining has finished. Called at the end
# of scripts/launch_all.sh (so the reboot hook covers it); run it again by hand to start
# fine-tunes once pretrainings finish.
#   bash scripts/launch_water16.sh
#
#   jepa_pretrain_<variant>_water16              5 pretrainings, one per variant, each on an empty GPU
#   flashmd_water16_scratch_10pct_s<seed>        baseline on 10% of the pairs, seeds 0 1 2
#   jepa_finetune_<variant>_water16_10pct_s<seed>  fine-tunes, seeds 0 1 2
#
# Pretraining batch: 8 water frames of 192 atoms = 1,536 atoms per batch, chosen to match
# the organic pretraining (64 pairs x 23 atoms on average = 1,477 atoms), because the
# variance and covariance terms are estimated over the atoms of a batch. It also gives
# 1,120 steps per epoch against organic's 1,313. Everything else as organic: 20 epochs.
set -uo pipefail
REPO=/storage3/lauren/code/jepa
cd "$REPO"
FREE_MB=${FREE_MB:-500}
RUNS_PER_GPU=${RUNS_PER_GPU:-3}
D=data/water/pairs/16fs
W_CONFIG=configs/flashmd_water16_scratch.yaml
VARIANTS="plain|--predictor plain
normtarget|--predictor norm_target
resid|--predictor residual
proj|--projector_dim 2048
residrel|--predictor residual --relative_loss"

if [ ! -s "$D/train_10pct.xyz" ]; then
  ( source env_train.sh && python data/water/make_water_subset.py "$D/train.xyz" 10 "$D/train_10pct.xyz" "$W_CONFIG" ) \
    || { echo "could not make $D/train_10pct.xyz"; exit 1; }
fi

ours=""
for f in results/runs/*/gpu; do
  [ -f "$f" ] || continue
  n=$(basename "$(dirname "$f")")
  tmux has-session -t "=train_$n" 2>/dev/null || continue
  g=$(cat "$f")
  case "$n" in jepa_pretrain_*) for k in $(seq 1 "$RUNS_PER_GPU"); do ours="$ours $g"; done ;; *) ours="$ours $g" ;; esac
done
pick_gpu() {  # $1 = how many of our runs may share the GPU (1 = needs an empty GPU)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v lim="$FREE_MB" -v ours="$ours" -v per="$1" '
      BEGIN { n = split(ours, a, " "); for (i = 1; i <= n; i++) if (a[i] != "") cnt[a[i]]++ }
      { g = $1; c = (g in cnt) ? cnt[g] : 0
        if ($2 < lim && c == 0) { print g; exit }
        if (per > 1 && c > 0 && c < per) { print g; exit } }'
}
running() { tmux has-session -t "=train_$1" 2>/dev/null; }
start() {  # $1 = run name, $2 = runs allowed per GPU, $3 = command (GPU is replaced by the index)
  local gpu=""
  for attempt in $(seq 1 144); do gpu=$(pick_gpu "$2"); [ -n "$gpu" ] && break; echo "  $1: no GPU with room, waiting 10 min"; sleep 600; done
  [ -z "$gpu" ] && { echo "  $1: no GPU after 24 h, skipped"; return; }
  if [ "$2" = 1 ]; then for k in $(seq 1 "$RUNS_PER_GPU"); do ours="$ours $gpu"; done; else ours="$ours $gpu"; fi
  tmux new-session -d -s "train_$1" "${3//GPU/$gpu}"
  echo "  $1: started on GPU $gpu"
  sleep 60
}

echo "$(date '+%F %T') launch_water16"
# pretrainings first: each needs a whole card
while IFS='|' read -r v flags; do
  n=jepa_pretrain_${v}_water16
  if [ -f "results/runs/$n/manifest.json" ]; then echo "  $n: finished"
  elif running "$n"; then echo "  $n: already running"
  else start "$n" 1 "CONFIG=$W_CONFIG TRAIN=$D/train.xyz VAL=$D/val.xyz bash $REPO/scripts/pretrain_supervisor.sh $n GPU --batch_size 8 $flags"; fi
done <<< "$VARIANTS"
# baselines and fine-tunes
for s in 0 1 2; do
  for n in flashmd_water16_scratch_10pct_s$s $(while IFS='|' read -r v _; do echo jepa_finetune_${v}_water16_10pct_s$s; done <<< "$VARIANTS"); do
    if [ -f "results/runs/$n/model.pt" ]; then echo "  $n: finished"; continue; fi
    if running "$n"; then echo "  $n: already running"; continue; fi
    ckpt=$(grep -m1 "read_from: /storage3" "configs/$n.yaml" | awk '{print $2}')
    if [ -n "$ckpt" ] && [ ! -f "$(dirname "$ckpt")/manifest.json" ]; then echo "  $n: pretraining not finished, skipped"; continue; fi
    start "$n" "$RUNS_PER_GPU" "bash $REPO/scripts/train_supervisor.sh $n configs/$n.yaml GPU"
  done
done
