#!/usr/bin/env bash
# One q-TIP4P/f water run following the FlashMD water protocol:
#   5 ps Langevin equilibration at 300 K (0.5 fs, tau 10 fs), then
#   2 ps constant-energy production (0.25 fs) with positions and momenta every step.
#
# Usage: bash data/water/run_one.sh <seed> <out_dir>
# Requires i-pi and i-pi-driver (Fortran, built from i-PI's drivers/f90) on PATH.
set -euo pipefail
# one core per run: numpy inside i-PI otherwise starts a thread per core, and 20 parallel
# runs on a 64-core node would start 1280 threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SEED=${1:?seed}
OUT=${2:?output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
EQUIL_STEPS=${EQUIL_STEPS:-10000}   # 10000 x 0.5 fs = 5 ps
NVE_STEPS=${NVE_STEPS:-8000}        # 8000 x 0.25 fs = 2 ps
ADDR="qtip4pf_${SEED}_$$"

wait_for_socket() {  # i-PI creates /tmp/ipi_<address>; the driver exits if it is not there yet
  for _ in $(seq 1 120); do
    [ -S "/tmp/ipi_$1" ] && return 0
    sleep 0.5
  done
  echo "i-PI socket /tmp/ipi_$1 did not appear"; exit 1
}

mkdir -p "$OUT" && cd "$OUT"
cp "$HERE/water64_init_ipi.xyz" init.xyz

sed -e "s|__SEED__|$SEED|; s|__ADDRESS__|$ADDR|; s|__EQUIL_STEPS__|$EQUIL_STEPS|; s|__INIT_XYZ__|init.xyz|" \
    "$HERE/equil.xml.template" > equil.xml
i-pi equil.xml > equil.log 2>&1 &
IPI=$!
wait_for_socket "$ADDR"
i-pi-driver -u -a "$ADDR" -m qtip4pf > driver_equil.log 2>&1 &
wait $IPI
# checkpoint stride equals EQUIL_STEPS, so equil.chk is the final state
test -s equil.chk
grep -q "<step>$EQUIL_STEPS</step>" equil.chk || { echo "equil.chk is not at step $EQUIL_STEPS"; exit 1; }

sed -e "s|__SEED__|$SEED|; s|__ADDRESS__|${ADDR}_nve|; s|__NVE_STEPS__|$NVE_STEPS|; s|__EQUIL_CHK__|equil.chk|" \
    "$HERE/nve.xml.template" > nve.xml
i-pi nve.xml > nve.log 2>&1 &
IPI=$!
wait_for_socket "${ADDR}_nve"
i-pi-driver -u -a "${ADDR}_nve" -m qtip4pf > driver_nve.log 2>&1 &
wait $IPI
echo "seed $SEED done: $(wc -l < nve.out) property lines"
