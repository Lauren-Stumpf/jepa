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
#                                                          stage 2 starts from stage 1's last epoch (model_last.ckpt)
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

# Our runs on each GPU, recounted on every call from the gpu files of live sessions (another
# launcher may have placed runs meanwhile). A pretraining fills its card. Sharing also needs
# SHARE_FREE_MB of free memory, because a run that is still loading data uses none yet.
SHARE_FREE_MB=${SHARE_FREE_MB:-7000}
count_ours() {
  local f n g k out=""
  for f in results/runs/*/gpu; do
    [ -f "$f" ] || continue
    n=$(basename "$(dirname "$f")")
    tmux has-session -t "=train_$n" 2>/dev/null || continue
    g=$(cat "$f")
    case "$n" in *pretrain*) for k in $(seq 1 "$RUNS_PER_GPU"); do out="$out $g"; done ;; *) out="$out $g" ;; esac
  done
  echo "$out"
}
pick_gpu() {  # $1 = how many of our runs may share the GPU (1 = needs an empty GPU)
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v lim="$FREE_MB" -v ours="$(count_ours)" -v per="$1" -v need="$SHARE_FREE_MB" '
      BEGIN { n = split(ours, a, " "); for (i = 1; i <= n; i++) if (a[i] != "") cnt[a[i]]++ }
      { g = $1; c = (g in cnt) ? cnt[g] : 0
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

echo "$(date '+%F %T') launch_frozen"
for start_enc in random plain; do
  a=frozen_${start_enc}_organic8_10pct_s0
  if [ -f "results/runs/$a/model.pt" ]; then echo "  $a: finished"
  elif running "$a"; then echo "  $a: already running"
  else start "$a" "$RUNS_PER_GPU" "bash $REPO/scripts/train_supervisor.sh $a configs/$a.yaml GPU"; fi

  for s in $SEEDS_B; do
    s1=headsfirst1_${start_enc}_organic8_10pct_s$s
    s2=headsfirst2_${start_enc}_organic8_10pct_s$s
    # stage 2 starts from stage 1's LAST epoch (scripts/stage2_supervisor.sh writes model_last.ckpt first)
    stage2="bash $REPO/scripts/stage2_supervisor.sh $s1 $s2 GPU"
    if [ -f "results/runs/$s2/model.pt" ]; then echo "  $s2: finished"
    elif running "$s1" || running "$s2"; then echo "  $s1 / $s2: already running"
    elif [ -f "results/runs/$s1/model.pt" ]; then start "$s2" "$RUNS_PER_GPU" "$stage2"
    else
      # stage 1, then stage 2 in its own session on the same GPU (so both are counted by the launchers)
      start "$s1" "$RUNS_PER_GPU" "bash $REPO/scripts/train_supervisor.sh $s1 configs/$s1.yaml GPU && tmux new-session -d -s train_$s2 \"$stage2\""
    fi
  done
done
