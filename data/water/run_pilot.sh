#!/usr/bin/env bash
# One long (100 ps) equilibration of 64 q-TIP4P/f waters at experimental density, to
# (a) measure how long the structure takes to relax, and (b) provide an equilibrated
# starting structure for all production runs, as FlashMD did (App. B.3: "equilibrated with
# q-TIP4P/f"). About 10 minutes.
#   bash data/water/run_pilot.sh /storage3/lauren/data/water_pilot
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT
OUT=${1:?output directory}
STEPS=${STEPS:-200000}   # 200000 x 0.5 fs = 100 ps
HERE=$(cd "$(dirname "$0")" && pwd)
ADDR="qtip4pf_pilot_$$"
mkdir -p "$OUT" && cd "$OUT"
python "$HERE/make_start.py" "$HERE/water64_init_ipi.xyz" start.xyz
sed -e "s|__ADDRESS__|$ADDR|; s|__STEPS__|$STEPS|; s|__INIT_XYZ__|start.xyz|" "$HERE/pilot.xml.template" > pilot.xml
i-pi pilot.xml > pilot.log 2>&1 &
IPI=$!
for _ in $(seq 1 2400); do [ -S "/tmp/ipi_$ADDR" ] && break; kill -0 $IPI 2>/dev/null || { tail pilot.log; exit 1; }; sleep 0.5; done
i-pi-driver -u -a "$ADDR" -m qtip4pf > driver.log 2>&1 &
wait $IPI
# last written frame becomes the equilibrated starting structure
n=$(head -1 pilot.pos_0.xyz)
tail -n $((n + 2)) pilot.pos_0.xyz > water64_equilibrated.xyz
echo "pilot done: $(wc -l < pilot.out) property lines; last frame -> $OUT/water64_equilibrated.xyz"
