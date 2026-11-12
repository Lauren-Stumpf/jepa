#!/usr/bin/env bash
# Start every run listed in an experiment's runs file on this node, as GPUs allow, and keep
# going until all of them have finished (or 48 h have passed). Safe to run on several nodes
# at once and to re-run at any time: finished runs are skipped, running ones (here or on
# another node) are left alone, and a run is claimed on the shared disk before it starts.
#   bash scripts/launch_runs.sh experiments/E05/runs.txt
#
# Runs file, one run per line (lines starting with # are ignored):
#   name|train|<min GPU memory, MB>|<runs per GPU>|configs/<name>.yaml
#   name|pretrain|<min GPU memory, MB>|<runs per GPU>|VAR=value ... -- <flags for train/pretrain_jepa.py>
# A train run whose config fine-tunes from results/runs/<p>/... waits until <p> has finished
# (manifest.json for a pretraining, model.pt for a training).
# "min GPU memory" is the card's total memory needed (a water fine-tune needs a 46 GB card);
# "runs per GPU" 1 means the card must hold no other run of ours and be otherwise empty.
set -uo pipefail
RUNS_FILE=${1:?runs file}
REPO=/storage3/lauren/code/jepa
cd "$REPO"
FREE_MB=${FREE_MB:-500}                  # an "empty" card has less than this in use
SHARE_FREE_MB=${SHARE_FREE_MB:-30000}    # free memory needed to add a run to a card we already use
PASS_MIN=${PASS_MIN:-10}                 # minutes between passes
MAX_HOURS=${MAX_HOURS:-48}
HOST=$(hostname -s)

count_ours() {   # GPU index of every live run of ours on this node (a pretraining counts as filling its card)
  local f n g k out=""
  for f in results/runs/*/gpu; do
    [ -f "$f" ] || continue
    n=$(basename "$(dirname "$f")")
    tmux has-session -t "=train_$n" 2>/dev/null || continue
    g=$(cat "$f")
    case "$n" in *pretrain*) for k in 1 2 3 4; do out="$out $g"; done ;; *) out="$out $g" ;; esac
  done
  echo "$out"
}
pick_gpu() {  # $1 = min total MB, $2 = runs per GPU
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
    awk -F', ' -v lim="$FREE_MB" -v ours="$(count_ours)" -v mintot="$1" -v per="$2" -v need="$SHARE_FREE_MB" '
      BEGIN { n = split(ours, a, " "); for (i = 1; i <= n; i++) if (a[i] != "") cnt[a[i]]++ }
      { g = $1; c = (g in cnt) ? cnt[g] : 0
        if ($3 < mintot) next
        if ($2 < lim && c == 0) { print g; exit }
        if (per > 1 && c > 0 && c < per && $3 - $2 > need) { print g; exit } }'
}
# alive here (tmux session), or alive on another node: its supervisor recorded that node and its
# log changed in the last 30 min
running() {
  tmux has-session -t "=train_$1" 2>/dev/null && return 0
  local h; h=$(cat "results/runs/$1/host" 2>/dev/null)
  [ -n "$h" ] && [ "$h" != "$HOST" ] && [ -n "$(find "results/runs/$1" -maxdepth 1 -name 'train.log' -mmin -30 2>/dev/null)" ]
}
finished() { [ -f "results/runs/$1/model.pt" ] || [ -f "results/runs/$1/manifest.json" ]; }
# a claim both nodes can see (mkdir is atomic on the shared disk); this node may reuse its own
# claim (after a reboot); another node's claim counts unless the run looks dead (no change to its
# train.log or the claim for 30 min), and is then taken over by an atomic rename
claim() {
  local c="results/runs/$1/.launch_claim"
  mkdir -p "results/runs/$1"
  if mkdir "$c" 2>/dev/null; then echo "$HOST" > "$c/host"; return 0; fi
  [ "$(cat "$c/host" 2>/dev/null)" = "$HOST" ] && { touch "$c"; return 0; }
  if [ -z "$(find "results/runs/$1" -maxdepth 1 \( -name train.log -o -name .launch_claim \) -mmin -30 2>/dev/null)" ]; then
    mv "$c" "$c.stale.$HOST.$$" 2>/dev/null && mkdir "$c" 2>/dev/null && { echo "$HOST" > "$c/host"; return 0; }
  fi
  return 1
}
deps_done() {  # $1 = kind, $2 = spec
  [ "$1" = train ] || return 0
  local ckpt; ckpt=$(grep -m1 "read_from: /storage3" "$2" | awk '{print $2}')
  [ -z "$ckpt" ] && return 0
  [ -f "$(dirname "$ckpt")/manifest.json" ] || [ -f "$(dirname "$ckpt")/model.pt" ]
}
command_for() {  # $1 = name, $2 = kind, $3 = spec; GPU is replaced by the card index
  if [ "$2" = train ]; then
    echo "bash $REPO/scripts/train_supervisor.sh $1 $3 GPU"
  else
    local envs flags
    envs=${3%% -- *}; flags=${3#* -- }
    echo "$envs bash $REPO/scripts/pretrain_supervisor.sh $1 GPU $flags"
  fi
}
try_start() {  # $1 name, $2 kind, $3 min MB, $4 per GPU, $5 spec; starts the run if a GPU is free now
  local out
  out=$( (
    flock 9
    if running "$1" || finished "$1"; then echo SKIP; exit 0; fi
    g=$(pick_gpu "$3" "$4")
    [ -z "$g" ] && { echo NOGPU; exit 0; }
    claim "$1" || { echo SKIP; exit 0; }
    cmd=$(command_for "$1" "$2" "$5")
    tmux new-session -d -s "train_$1" "${cmd//GPU/$g}"
    sleep 90   # let it allocate memory before the next choice
    echo "$g"
  ) 9>/tmp/jepa_gpu_pick.lock )
  case "$out" in
    SKIP) ;;
    NOGPU) echo "  $1: no suitable GPU free" ;;
    *) echo "  $1: started on GPU $out" ;;
  esac
}

start_time=$(date +%s)
while :; do
  echo "$(date '+%F %T') pass over $RUNS_FILE on $HOST"
  pending=0
  while IFS='|' read -r name kind minmb per spec; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if finished "$name"; then continue; fi
    if grep -q "giving up" "results/runs/$name/supervisor.log" 2>/dev/null; then
      echo "  $name: its supervisor gave up after repeated failures; needs a person (not restarted)"; continue
    fi
    pending=$((pending + 1))
    if running "$name"; then continue; fi
    if ! deps_done "$kind" "$spec"; then echo "  $name: waiting for the run it starts from"; continue; fi
    try_start "$name" "$kind" "$minmb" "$per" "$spec"
  done < "$RUNS_FILE"
  [ "$pending" -eq 0 ] && { echo "$(date '+%F %T') all runs in $RUNS_FILE finished"; break; }
  [ $(( $(date +%s) - start_time )) -gt $((MAX_HOURS * 3600)) ] && { echo "stopping after $MAX_HOURS h; $pending runs not finished"; break; }
  sleep $((PASS_MIN * 60))
done
