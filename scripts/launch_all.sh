#!/usr/bin/env bash
# Start (or resume) every baseline training that has not finished, each in its own
# detached tmux session on its own free GPU. Safe to run any number of times: running
# sessions are left alone and finished runs are skipped. Also run automatically after a
# reboot by the crontab line in scripts/install_reboot_hook.sh.
#   bash scripts/launch_all.sh
set -uo pipefail
REPO=/storage3/lauren/code/jepa
cd "$REPO"
# name  config
RUNS="flashmd_organic8_scratch configs/flashmd_organic8_scratch.yaml
flashmd_organic8_scratch_300 configs/flashmd_organic8_scratch_300.yaml
flashmd_water16_scratch configs/flashmd_water16_scratch.yaml
flashmd_water4_scratch configs/flashmd_water4_scratch.yaml
flashmd_water1_scratch configs/flashmd_water1_scratch.yaml"
FREE_MB=${FREE_MB:-500}    # a GPU counts as free if less than this much memory is in use
# GPUs already used by our own running supervisors (each writes its index to results/runs/<name>/gpu)
taken=""
for f in results/runs/*/gpu; do
  [ -f "$f" ] || continue
  n=$(basename "$(dirname "$f")")
  tmux has-session -t "train_$n" 2>/dev/null && taken="$taken $(cat "$f")"
done
echo "$(date '+%F %T') launch_all"
while read -r name cfg; do
  [ -z "$name" ] && continue
  if [ -f "results/runs/$name/model.pt" ]; then echo "  $name: finished"; continue; fi
  if tmux has-session -t "train_$name" 2>/dev/null; then echo "  $name: already running"; continue; fi
  train_file=$(grep -m1 "read_from:" "$cfg" | awk '{print $2}')
  if [ ! -s "$train_file" ]; then echo "  $name: no training data yet ($train_file), skipped"; continue; fi
  gpu=""
  for attempt in $(seq 1 144); do   # wait up to 24 h for a free GPU
    gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
          awk -F', ' -v lim="$FREE_MB" -v taken=" $taken " '$2 < lim && index(taken, " "$1" ") == 0 {print $1; exit}')
    [ -n "$gpu" ] && break
    echo "  $name: no free GPU, waiting 10 min"; sleep 600
  done
  [ -z "$gpu" ] && { echo "  $name: no GPU after 24 h, skipped"; continue; }
  taken="$taken $gpu"
  tmux new-session -d -s "train_$name" "bash $REPO/scripts/train_supervisor.sh $name $cfg $gpu"
  echo "  $name: started on GPU $gpu (tmux session train_$name)"
  sleep 60   # let it allocate GPU memory before the next one looks for a free GPU
done <<< "$RUNS"
