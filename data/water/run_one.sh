#!/usr/bin/env bash
# One q-TIP4P/f water run following the FlashMD water protocol:
#   Langevin equilibration at 300 K (0.5 fs) from an equilibrated structure, then
#   2 ps constant-energy production (0.25 fs) with positions and momenta every step.
#
# Usage: bash data/water/run_one.sh <seed> <out_dir>
# Requires i-pi and i-pi-driver (Fortran, built from i-PI's drivers/f90) on PATH.
set -euo pipefail
# if this script dies for any reason, take its i-PI and driver with it (no orphans holding files open)
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
# one core per run: numpy inside i-PI otherwise starts a thread per core, and 20 parallel
# runs on a 64-core node would start 1280 threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEED=${1:?seed}
OUT=${2:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
# Equilibration per run, starting from the structure equilibrated by run_pilot.sh (as in
# FlashMD App. B.3). Length and coupling are set from the pilot's relaxation time; see
# notes/decisions.md (2026-11-05). Override with EQUIL_STEPS / TAU_FS.
EQUIL_STEPS=${EQUIL_STEPS:-50000}   # 50000 x 0.5 fs = 25 ps
TAU_FS=${TAU_FS:-100}               # Langevin time constant (fs)
INIT=${INIT:-/storage3/lauren/data/water_pilot/water64_equilibrated.xyz}
NVE_STEPS=${NVE_STEPS:-8000}        # 8000 x 0.25 fs = 2 ps
ADDR="qtip4pf_${SEED}_$$"

wait_for_socket() {  # i-PI creates /tmp/ipi_<address>; the driver exits if it is not there yet.
  # $1 = socket address, $2 = i-PI pid, $3 = i-PI log. Startup can take minutes when many
  # runs import i-PI at once from network storage, so wait up to 20 minutes, but stop as
  # soon as i-PI itself has died.
  for _ in $(seq 1 2400); do
    [ -S "/tmp/ipi_$1" ] && return 0
    if ! kill -0 "$2" 2>/dev/null; then
      echo "i-PI exited before opening socket /tmp/ipi_$1; last lines of $3:"; tail -5 "$3"; exit 1
    fi
    sleep 0.5
  done
  echo "i-PI socket /tmp/ipi_$1 did not appear within 20 minutes"; kill "$2" 2>/dev/null; exit 1
}

mkdir -p "$OUT" && cd "$OUT"
[ -f "$INIT" ] || { echo "no equilibrated start $INIT: run data/water/run_pilot.sh first"; exit 1; }
cp "$INIT" init.xyz

sed -e "s|__SEED__|$SEED|; s|__ADDRESS__|$ADDR|; s|__EQUIL_STEPS__|$EQUIL_STEPS|; s|__TAU__|$TAU_FS|; s|__INIT_XYZ__|init.xyz|" \
    "$HERE/equil.xml.template" > equil.xml
i-pi equil.xml > equil.log 2>&1 &
IPI=$!
wait_for_socket "$ADDR" "$IPI" equil.log
i-pi-driver -u -a "$ADDR" -m qtip4pf > driver_equil.log 2>&1 &
wait $IPI
# checkpoint stride equals EQUIL_STEPS, so equil.chk is the final state
test -s equil.chk
grep -q "<step>$EQUIL_STEPS</step>" equil.chk || { echo "equil.chk is not at step $EQUIL_STEPS"; exit 1; }

sed -e "s|__SEED__|$SEED|; s|__ADDRESS__|${ADDR}_nve|; s|__NVE_STEPS__|$NVE_STEPS|; s|__EQUIL_CHK__|equil.chk|" \
    "$HERE/nve.xml.template" > nve.xml
i-pi nve.xml > nve.log 2>&1 &
IPI=$!
wait_for_socket "${ADDR}_nve" "$IPI" nve.log
i-pi-driver -u -a "${ADDR}_nve" -m qtip4pf > driver_nve.log 2>&1 &
wait $IPI
echo "seed $SEED done: $(wc -l < nve.out) property lines"
