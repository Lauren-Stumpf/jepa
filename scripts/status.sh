#!/usr/bin/env bash
# One-screen status of every baseline: running or not, latest epoch line, GPU use.
REPO=/storage3/lauren/code/jepa
for d in $REPO/results/runs/*/; do
  n=$(basename "$d")
  if [ -f "$d/model.pt" ] || [ -f "$d/manifest.json" ]; then s="finished"
  elif tmux has-session -t "=train_$n" 2>/dev/null; then s="running on $(hostname -s)"
  else
    # may be running on another node: a log written in the last 15 minutes means it is alive somewhere
    if [ -n "$(find "$d" -maxdepth 1 -name train.log -mmin -15 2>/dev/null)" ]; then s="running elsewhere (log updated in the last 15 min)"
    else s="NOT RUNNING"; fi
  fi
  echo "== $n: $s"
  grep -h "Epoch\|^epoch" "$d/train.log" 2>/dev/null | tail -1 | cut -c1-200
  tail -2 "$d/supervisor.log" 2>/dev/null
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
