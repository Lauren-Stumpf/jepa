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
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
