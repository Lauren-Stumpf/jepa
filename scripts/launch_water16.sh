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

# Our runs on each GPU, recounted on every call from the gpu files of live sessions (another
# launcher may have placed runs meanwhile). A pretraining fills its card. Sharing also needs
# SHARE_FREE_MB of free memory, because a run that is still loading data uses none yet.
MIN_TOTAL_FT_MB=${MIN_TOTAL_FT_MB:-40000}   # so water fine-tunes never go on a 24 GB card (pretraining, batch 8, fits)
SHARE_FREE_MB=${SHARE_FREE_MB:-30000}   # a water fine-tune (16 frames x 192 atoms) uses about 28 GB: never share a 46 GB card, never fit a 24 GB one
count_ours() {
  local f n g k out=""
  for f in results/runs/*/gpu; do
    [ -f "$f" ] || continue
    n=$(basename "$(dirname "$f")")
    tmux has-session -t "=train_$n" 2>/dev/null || continue
    g=$(cat "$f")
    case "$n" in jepa_pretrain_*) for k in $(seq 1 "$RUNS_PER_GPU"); do out="$out $g"; done ;; *) out="$out $g" ;; esac
  done
  echo "$out"
}
pick_gpu() {  # $1 = how many of our runs may share the GPU (1 = needs an empty GPU)
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v lim="$FREE_MB" -v ours="$(count_ours)" -v per="$1" -v need="$SHARE_FREE_MB" -v mintot="$MIN_TOTAL_FT_MB" '
      BEGIN { n = split(ours, a, " "); for (i = 1; i <= n; i++) if (a[i] != "") cnt[a[i]]++ }
      { g = $1; c = (g in cnt) ? cnt[g] : 0
        if (per > 1 && $3 < mintot) next      # fine-tunes need a card with room for about 28 GB
        if ($2 < lim && c == 0) { print g; exit }
        if (per > 1 && c > 0 && c < per && $3 - $2 > need) { print g; exit } }'
}
# alive here (tmux session), or alive on the OTHER node: its supervisor recorded that node and
# its log moved in the last 30 min (same rule as launch_all.sh)
running() {
  tmux has-session -t "=train_$1" 2>/dev/null && return 0
  local h; h=$(cat "results/runs/$1/host" 2>/dev/null)
  [ -n "$h" ] && [ "$h" != "$(hostname -s)" ] && [ -n "$(find "results/runs/$1" -maxdepth 1 -name 'train.log' -mmin -30 2>/dev/null)" ]
}
# A claim visible to both nodes (mkdir is atomic on the shared filesystem), so launchers on two
# nodes cannot start the same run in the same moment. A claim by THIS node may be reused (a
# reboot here); another node's claim is honoured unless the run looks dead (neither its
# train.log nor the claim changed for 30 min).
claim() {
  local c="results/runs/$1/.launch_claim"
  mkdir -p "results/runs/$1"
  if mkdir "$c" 2>/dev/null; then hostname -s > "$c/host"; return 0; fi
  [ "$(cat "$c/host" 2>/dev/null)" = "$(hostname -s)" ] && { touch "$c"; return 0; }
  if [ -z "$(find "results/runs/$1" -maxdepth 1 \( -name train.log -o -name .launch_claim \) -mmin -30 2>/dev/null)" ]; then
    # take over a dead claim atomically: only one node's rename can succeed
    mv "$c" "$c.stale.$(hostname -s).$$" 2>/dev/null && mkdir "$c" 2>/dev/null && { hostname -s > "$c/host"; return 0; }
  fi
  return 1
}
start() {  # $1 = run name, $2 = runs allowed per GPU, $3 = command (GPU is replaced by the index)
  # pick, launch and wait for the run to allocate memory under one lock on this node's own disk,
  # so two launchers never choose the same GPU in the same moment
  local gpu="" attempt
  for attempt in $(seq 1 144); do
    gpu=$( (
      flock 9
      # re-check under the lock: while this launcher waited, the run may have been started
      # here or by a launcher on another node (or finished)
      if running "$1" || [ -f "results/runs/$1/model.pt" ] || [ -f "results/runs/$1/manifest.json" ]; then echo SKIP; exit 0; fi
      g=$(pick_gpu "$2")
      if [ -n "$g" ]; then
        claim "$1" || { echo SKIP; exit 0; }
        tmux new-session -d -s "train_$1" "${3//GPU/$g}"
        sleep 90
      fi
      echo "$g"
    ) 9>/tmp/jepa_gpu_pick.lock )
    [ "$gpu" = SKIP ] && { echo "  $1: started elsewhere or finished while waiting, skipped"; return; }
    [ -n "$gpu" ] && break
    echo "  $1: no GPU with room, waiting 10 min"; sleep 600
  done
  [ -z "$gpu" ] && { echo "  $1: no GPU after 24 h, skipped"; return; }
  echo "  $1: started on GPU $gpu"
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
