#!/usr/bin/env bash
# Start the three predictor-variant pretrainings on torrnode7, one per GPU, each in its own
# tmux session (they resume on their own if interrupted; rerun this script after a reboot,
# it skips arms that are running or finished). Their fine-tunes are started by
# scripts/launch_all.sh on torrnode12 once each pretraining has written manifest.json.
#   bash scripts/launch_pretrain_node7.sh [gpu_a gpu_b gpu_c]   (default GPUs 2 3 4)
set -uo pipefail
REPO=/storage3/lauren/code/jepa
cd "$REPO"
GPUS=(${1:-2} ${2:-3} ${3:-4})
ARMS=("jepa_pretrain_plain_organic8|--predictor plain"
      "jepa_pretrain_normtarget_organic8|--predictor norm_target"
      "jepa_pretrain_resid_organic8|--predictor residual")
for k in 0 1 2; do
  name=${ARMS[$k]%%|*}; extra=${ARMS[$k]#*|}; gpu=${GPUS[$k]}
  if [ -f "results/runs/$name/manifest.json" ]; then echo "$name: finished"; continue; fi
  if tmux has-session -t "train_$name" 2>/dev/null; then echo "$name: already running"; continue; fi
  tmux new-session -d -s "train_$name" "bash $REPO/scripts/pretrain_supervisor.sh $name $gpu $extra"
  echo "$name: started on GPU $gpu ($extra)"
done
