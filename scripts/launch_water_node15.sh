#!/usr/bin/env bash
# Start the three water baselines (1, 4 and 16 fs strides) on torrnode15, one per GPU, each
# in its own tmux session under the usual supervisor. Safe to rerun (also run by the
# @reboot crontab entry on torrnode15): finished or running arms are skipped.
#   bash scripts/launch_water_node15.sh [gpu_16fs gpu_4fs gpu_1fs]   (default GPUs 1 5 7)
set -uo pipefail
REPO=/storage3/lauren/code/jepa
cd "$REPO"
GPUS=(${1:-1} ${2:-5} ${3:-7})
ARMS=(flashmd_water16_scratch flashmd_water4_scratch flashmd_water1_scratch)
for k in 0 1 2; do
  name=${ARMS[$k]}; gpu=${GPUS[$k]}
  if [ -f "results/runs/$name/model.pt" ]; then echo "$name: finished"; continue; fi
  if tmux has-session -t "=train_$name" 2>/dev/null; then echo "$name: already running"; continue; fi
  tmux new-session -d -s "train_$name" "bash $REPO/scripts/train_supervisor.sh $name configs/$name.yaml $gpu"
  echo "$name: started on GPU $gpu"
done
