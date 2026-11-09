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
flashmd_water1_scratch configs/flashmd_water1_scratch.yaml
jepa_pretrain_organic8 PRETRAIN
jepa_pretrain_proj_organic8 PRETRAIN:--projector_dim 2048
jepa_finetune_organic8 configs/jepa_finetune_organic8.yaml
flashmd_organic8_scratch_10pct_s0 configs/flashmd_organic8_scratch_10pct_s0.yaml
flashmd_organic8_scratch_10pct_s1 configs/flashmd_organic8_scratch_10pct_s1.yaml
flashmd_organic8_scratch_10pct_s2 configs/flashmd_organic8_scratch_10pct_s2.yaml
jepa_finetune_organic8_10pct_s0 configs/jepa_finetune_organic8_10pct_s0.yaml
jepa_finetune_organic8_10pct_s1 configs/jepa_finetune_organic8_10pct_s1.yaml
jepa_finetune_organic8_10pct_s2 configs/jepa_finetune_organic8_10pct_s2.yaml
jepa_finetune_proj_organic8_10pct_s0 configs/jepa_finetune_proj_organic8_10pct_s0.yaml
jepa_finetune_proj_organic8_10pct_s1 configs/jepa_finetune_proj_organic8_10pct_s1.yaml
jepa_finetune_proj_organic8_10pct_s2 configs/jepa_finetune_proj_organic8_10pct_s2.yaml"
FREE_MB=${FREE_MB:-500}       # a GPU nobody uses: less than this much memory in use
RUNS_PER_GPU=${RUNS_PER_GPU:-3}   # our runs use ~4 GB each on 46 GB cards, so several can share one
# GPUs already used by our own running supervisors (each writes its index to results/runs/<name>/gpu)
ours=""
for f in results/runs/*/gpu; do
  [ -f "$f" ] || continue
  n=$(basename "$(dirname "$f")")
  tmux has-session -t "train_$n" 2>/dev/null && ours="$ours $(cat "$f")"
done
pick_gpu() {  # a GPU is eligible if it is empty, or if only our runs are on it and there is room for one more
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v lim="$FREE_MB" -v ours="$ours" -v per="$RUNS_PER_GPU" '
      BEGIN { n = split(ours, a, " "); for (i = 1; i <= n; i++) if (a[i] != "") cnt[a[i]]++ }
      { g = $1; c = (g in cnt) ? cnt[g] : 0
        if (c < per && (c > 0 || $2 < lim)) { print g; exit } }'
}
echo "$(date '+%F %T') launch_all"
while read -r name cfg; do
  [ -z "$name" ] && continue
  case "$cfg" in PRETRAIN*) is_pretrain=1;; *) is_pretrain=0;; esac
  if [ "$is_pretrain" = 1 ]; then
    done_file="results/runs/$name/manifest.json"
  else
    done_file="results/runs/$name/model.pt"
    train_file=$(grep -m1 "read_from:" "$cfg" | awk '{print $2}')
    ckpt=$(grep -m1 "read_from: results" "$cfg" | awk '{print $2}')
  fi
  if [ -f "$done_file" ]; then echo "  $name: finished"; continue; fi
  if tmux has-session -t "train_$name" 2>/dev/null; then echo "  $name: already running"; continue; fi
  if [ "$is_pretrain" = 0 ]; then
    if [ ! -s "$train_file" ]; then echo "  $name: no training data yet ($train_file), skipped"; continue; fi
    if [ -n "$ckpt" ] && [ ! -s "$ckpt" ]; then echo "  $name: pretrained checkpoint not ready ($ckpt), skipped"; continue; fi
  fi
  gpu=""
  for attempt in $(seq 1 144); do   # wait up to 24 h for a GPU
    gpu=$(pick_gpu)
    [ -n "$gpu" ] && break
    echo "  $name: no GPU with room, waiting 10 min"; sleep 600
  done
  [ -z "$gpu" ] && { echo "  $name: no GPU after 24 h, skipped"; continue; }
  ours="$ours $gpu"
  if [ "$is_pretrain" = 1 ]; then
    extra=${cfg#PRETRAIN}; extra=${extra#:}
    tmux new-session -d -s "train_$name" "bash $REPO/scripts/pretrain_supervisor.sh $name $gpu $extra"
  else
    tmux new-session -d -s "train_$name" "bash $REPO/scripts/train_supervisor.sh $name $cfg $gpu"
  fi
  echo "  $name: started on GPU $gpu (tmux session train_$name)"
  sleep 60   # let it allocate GPU memory before the next one looks for a free GPU
done <<< "$RUNS"
