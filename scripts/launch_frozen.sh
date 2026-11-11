#!/usr/bin/env bash
# Start (or continue) the frozen-encoder fine-tuning experiment on this node. Safe to run
# any number of times: finished runs are skipped and running ones left alone. Called at
# the end of scripts/launch_all.sh, so the reboot hook also restarts these.
#   bash scripts/launch_frozen.sh
#
# Runs (configs from scripts/make_frozen_configs.py):
#   frozen_<start>_organic8_10pct_s0                       procedure A, one run
#   headsfirst1_<start>_..._s<seed> -> headsfirst2_<start>_..._s<seed>
#                                                          procedure B, two stages in a chain:
#                                                          stage 2 starts from stage 1's model.ckpt
#   <start> = random (untrained encoder) or plain (plain-variant pretrained encoder)
set -uo pipefail
REPO=/storage3/lauren/code/jepa
cd "$REPO"
FREE_MB=${FREE_MB:-500}
RUNS_PER_GPU=${RUNS_PER_GPU:-3}
SEEDS_B="0 1 2"

# untrained encoders for the random arm: one per seed, made once (CPU, a few seconds each)
for s in $SEEDS_B; do
  out=results/runs/random_encoder_organic8_s$s
  if [ ! -f "$out/manifest.json" ]; then
    ( source env_train.sh && CUDA_VISIBLE_DEVICES="" python train/make_random_checkpoint.py \
        configs/flashmd_organic8_scratch.yaml --seed "$s" --out "$out" ) || { echo "could not make $out"; exit 1; }
  fi
done

# GPUs already used by our own running supervisors (same accounting as launch_all.sh)
ours=""
for f in results/runs/*/gpu; do
  [ -f "$f" ] || continue
  n=$(basename "$(dirname "$f")")
  tmux has-session -t "=train_$n" 2>/dev/null || continue
  g=$(cat "$f")
  case "$n" in jepa_pretrain_*) for k in $(seq 1 "$RUNS_PER_GPU"); do ours="$ours $g"; done ;; *) ours="$ours $g" ;; esac
done
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v lim="$FREE_MB" -v ours="$ours" -v per="$RUNS_PER_GPU" '
      BEGIN { n = split(ours, a, " "); for (i = 1; i <= n; i++) if (a[i] != "") cnt[a[i]]++ }
      { g = $1; c = (g in cnt) ? cnt[g] : 0
        if ($2 < lim && c == 0) { print g; exit }
        if (c > 0 && c < per) { print g; exit } }'
}
running() { tmux has-session -t "=train_$1" 2>/dev/null; }
start() {  # $1 = session/run name, $2 = command
  local gpu=""
  for attempt in $(seq 1 144); do gpu=$(pick_gpu); [ -n "$gpu" ] && break; echo "  $1: no GPU with room, waiting 10 min"; sleep 600; done
  [ -z "$gpu" ] && { echo "  $1: no GPU after 24 h, skipped"; return; }
  ours="$ours $gpu"
  tmux new-session -d -s "train_$1" "${2//GPU/$gpu}"
  echo "  $1: started on GPU $gpu"
  sleep 60
}

echo "$(date '+%F %T') launch_frozen"
for start_enc in random plain; do
  a=frozen_${start_enc}_organic8_10pct_s0
  if [ -f "results/runs/$a/model.pt" ]; then echo "  $a: finished"
  elif running "$a"; then echo "  $a: already running"
  else start "$a" "bash $REPO/scripts/train_supervisor.sh $a configs/$a.yaml GPU"; fi

  for s in $SEEDS_B; do
    s1=headsfirst1_${start_enc}_organic8_10pct_s$s
    s2=headsfirst2_${start_enc}_organic8_10pct_s$s
    stage2="bash $REPO/scripts/train_supervisor.sh $s2 configs/$s2.yaml GPU"
    if [ -f "results/runs/$s2/model.pt" ]; then echo "  $s2: finished"
    elif running "$s1" || running "$s2"; then echo "  $s1 / $s2: already running"
    elif [ -f "results/runs/$s1/model.pt" ]; then start "$s2" "$stage2"
    else
      # stage 1, then stage 2 in its own session on the same GPU (so both are counted by the launchers)
      start "$s1" "bash $REPO/scripts/train_supervisor.sh $s1 configs/$s1.yaml GPU && tmux new-session -d -s train_$s2 '$stage2'"
    fi
  done
done
