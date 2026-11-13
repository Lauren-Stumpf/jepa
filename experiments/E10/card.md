# E10: are the organic runs deterministic and energy-conserving? (hypothesis H3, noisy data)

Status: agreed (13 Nov 2026). Data check only.

**Question:** Pretraining assumes the future follows from the present. If the organic futures contain random kicks (a thermostat) or the forces are rough, part of the future cannot be predicted, and the encoder can lower its loss by dropping fast detail. Is that the case?

**Hypothesis H3 (noise):** the organic data are noisy in this sense; the water data are not (we generated them at constant energy, thermostat off).

**Prediction (confidence):**
- If H3 (noise) is true: total energy per atom varies within runs well above rounding, or rerunning PET-MAD from stored starts does not give back the stored futures.
- If false: energy is constant within every run (spread below 1e-3 eV per atom in 99% of runs), and reruns reproduce the futures to rounding.
- My confidence that H3 (noise) is true: 5%. Already seen on 13 Nov: one molecule's runs keep total energy constant to 4 decimals, and a rerun of 200 stored starts (30 Oct, results/eval/verlet_8fs.csv) reproduced the futures with mass-scaled error about 1e-7, against about 0.02 for our models.

**Measured:** for every 50-frame run of every organic molecule (all splits), the spread and drift of the total energy per atom (from the catalogue: 1.5 k_B T + stored potential energy); the jump between consecutive runs (to confirm the 50-frame run structure); the 30 Oct rerun summary.

**Controls:**
- Must work: blocks never span two molecules; consecutive runs of a molecule differ in energy (else the 50-frame structure is wrong).

**Assumptions:**
1. The catalogue's temperature counts kinetic energy as 1.5 k_B T per atom, so the total is exact up to a constant per molecule, which does not affect spread within a run.
2. The rerun covered 200 frames of the source file, not only organic ones; this run reports the energy check on every organic run.

**Measure and rule:** the spread distribution over runs, against the 1e-3 eV per atom threshold.

**Runs:** eval/check_organic_energy.py, seconds on a CPU.

## Result
Run 13 Nov (results/eval/organic_energy_check.json), all 2,100 runs of the 210 organic molecules:
- Total energy per atom within a run: median spread 0.0001 eV, 99% of runs below 0.0009 eV (under the 0.001 eV threshold).
- Worst runs are hot: 0.23 eV at 1,890 K and 0.13 eV at 1,406 K, then 0.012 eV at 395 K; all others below 0.003 eV. About 0.1% of runs do not conserve energy well.
- Rerunning PET-MAD from 200 stored starts (30 Oct) gives back the stored futures: mass-scaled error median 0.00000005, largest 0.000004, against about 0.02 for our models.

The prediction for "H3 is false" holds: the organic data are deterministic and energy-conserving, apart from a handful of very hot runs. H3 in its noise form is rejected.

## Assumptions check
- Every 50-frame block lies within one molecule, and consecutive runs of a molecule differ in energy (median jump 0.13 eV per atom), confirming the 50-frame run structure.
- The smallest jump between runs is 0.00001 eV per atom, so two runs of one molecule can have almost the same energy; this does not affect the spread within runs.

## Next
No further work on data noise. The rare hot runs are covered by the frame-type split (E09, temperature bands).
