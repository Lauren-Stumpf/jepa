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
jepa_pretrain_proj_organic8 PRETRAIN:--projector_dim 2048
jepa_finetune_organic8 configs/jepa_finetune_organic8.yaml
flashmd_organic8_scratch_10pct_s0 configs/flashmd_organic8_scratch_10pct_s0.yaml
flashmd_organic8_scratch_10pct_s1 configs/flashmd_organic8_scratch_10pct_s1.yaml
flashmd_organic8_scratch_10pct_s2 configs/flashmd_organic8_scratch_10pct_s2.yaml
jepa_finetune_proj_organic8_10pct_s0 configs/jepa_finetune_proj_organic8_10pct_s0.yaml
jepa_finetune_proj_organic8_10pct_s1 configs/jepa_finetune_proj_organic8_10pct_s1.yaml
jepa_finetune_proj_organic8_10pct_s2 configs/jepa_finetune_proj_organic8_10pct_s2.yaml
jepa_finetune_plain_organic8_10pct_s0 configs/jepa_finetune_plain_organic8_10pct_s0.yaml
jepa_finetune_plain_organic8_10pct_s1 configs/jepa_finetune_plain_organic8_10pct_s1.yaml
jepa_finetune_plain_organic8_10pct_s2 configs/jepa_finetune_plain_organic8_10pct_s2.yaml
jepa_finetune_normtarget_organic8_10pct_s0 configs/jepa_finetune_normtarget_organic8_10pct_s0.yaml
jepa_finetune_normtarget_organic8_10pct_s1 configs/jepa_finetune_normtarget_organic8_10pct_s1.yaml
jepa_finetune_normtarget_organic8_10pct_s2 configs/jepa_finetune_normtarget_organic8_10pct_s2.yaml
jepa_finetune_resid_organic8_10pct_s0 configs/jepa_finetune_resid_organic8_10pct_s0.yaml
jepa_finetune_resid_organic8_10pct_s1 configs/jepa_finetune_resid_organic8_10pct_s1.yaml
jepa_finetune_resid_organic8_10pct_s2 configs/jepa_finetune_resid_organic8_10pct_s2.yaml
jepa_finetune_residrel_organic8_10pct_s0 configs/jepa_finetune_residrel_organic8_10pct_s0.yaml
jepa_finetune_residrel_organic8_10pct_s1 configs/jepa_finetune_residrel_organic8_10pct_s1.yaml
jepa_finetune_residrel_organic8_10pct_s2 configs/jepa_finetune_residrel_organic8_10pct_s2.yaml"
FREE_MB=${FREE_MB:-500}       # a GPU nobody uses: less than this much memory in use
RUNS_PER_GPU=${RUNS_PER_GPU:-3}   # baselines and fine-tunes use ~5 GB each on 46 GB cards, so several can share one
# GPUs already used by our own running supervisors (each writes its index to results/runs/<name>/gpu)
ours=""
for f in results/runs/*/gpu; do
  [ -f "$f" ] || continue
  n=$(basename "$(dirname "$f")")
  tmux has-session -t "=train_$n" 2>/dev/null && ours="$ours $(cat "$f")"
done
pick_gpu() {  # $1 = how many of our runs may share the GPU (1 = needs an empty GPU)
  # eligible: empty, or (when sharing is allowed) only our runs on it and room for one more
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v lim="$FREE_MB" -v ours="$ours" -v per="$1" '
      BEGIN { n = split(ours, a, " "); for (i = 1; i <= n; i++) if (a[i] != "") cnt[a[i]]++ }
      { g = $1; c = (g in cnt) ? cnt[g] : 0
        if ($2 < lim && c == 0) { print g; exit }
        if (per > 1 && c > 0 && c < per) { print g; exit } }'
}
echo "$(date '+%F %T') launch_all"
while read -r name cfg; do
  [ -z "$name" ] && continue
  case "$cfg" in PRETRAIN*) is_pretrain=1;; *) is_pretrain=0;; esac
  if [ "$is_pretrain" = 1 ]; then
    done_file="results/runs/$name/manifest.json"
  else
    done_file="results/runs/$name/model.pt"
    train_file=$(awk '/^training_set:/{f=1} f && /read_from:/{print $2; exit}' "$cfg")
    ckpt=$(grep -m1 "read_from: /storage3" "$cfg" | awk '{print $2}')
  fi
  if [ -f "$done_file" ]; then echo "  $name: finished"; continue; fi
  if tmux has-session -t "=train_$name" 2>/dev/null; then echo "  $name: already running"; continue; fi
  if [ "$is_pretrain" = 0 ]; then
    if [ ! -s "$train_file" ]; then echo "  $name: no training data yet ($train_file), skipped"; continue; fi
    # a fine-tune must start from the FINISHED pretraining: pretrained.ckpt is rewritten after every
    # epoch, so its existence is not enough; the pretraining writes manifest.json only when it ends
    if [ -n "$ckpt" ] && [ ! -f "$(dirname "$ckpt")/manifest.json" ]; then echo "  $name: pretraining not finished ($(dirname "$ckpt")), skipped"; continue; fi
  fi
  gpu=""
  for attempt in $(seq 1 144); do   # wait up to 24 h for a GPU
    # pretraining runs two encoder passes at batch 64 and needs about 15 GB: give it an empty GPU.
    # fine-tunes and baselines use about 5 GB and may share up to RUNS_PER_GPU per card.
    if [ "$is_pretrain" = 1 ]; then gpu=$(pick_gpu 1); else gpu=$(pick_gpu "$RUNS_PER_GPU"); fi
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
