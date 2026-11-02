# Decisions

## 2026-10-30: PET-MAD version pinned to 1.0.0

Regenerating 50 frames of the FlashMD Zenodo 8 fs file (record 17904449) with
velocity Verlet at 0.25 fs for 32 steps:

- PET-MAD 1.0.0: median position difference 5.1e-8 A, relative momentum
  difference 1.3e-6, energy drift over the stride 0.011 meV/atom.
- PET-MAD 1.0.2: median position difference 2.5e-3 A, relative momentum
  difference 3.8e-2.

So the file was generated with PET-MAD 1.0.0, and our potential, integrator,
time step and units match the authors' exactly. All rollout references and any
data we extend from the file use PET-MAD 1.0.0.

Note on file layout (corrected 2026-11-02): 5 million frames, 10,000
structures, 500 consecutive frames per structure: 10 constant-energy runs at
random temperatures, 25 forward pairs per run plus their time-reversed copies
(adjacent, same displacement size). Taking the first N frames gives one
structure; sample across the file instead.

Outputs: results/checks/check_1.0.0.csv, results/checks/check_1.0.2.csv
(the force-vs-stored column is NaN in these two runs because of a reading bug
fixed in the same commit).

## 2026-11-02: metatrain's FlashMD loss is not the paper's mass-scaled loss

metatrain's experimental.flashmd trains on raw displacements and raw momenta,
each divided by its dataset standard deviation (no sqrt(mass) scaling). The
paper (App. B.1) uses mass-scaled targets. So the from-scratch baseline is
metatrain's formulation, and the validation errors metatrain logs are in A and
(eV u)^1/2, not comparable with the paper's Fig. 8. The mass-scaled numbers
for comparison are computed afterwards by eval/eval_onestep.py.

## 2026-11-02: independent code review before first training

A reviewing agent that had not seen the code written checked the repo against
the metatrain, flashmd 0.1.2 and i-PI sources. Fixed: evaluator could not read
metatrain exports (different output names and timestep buffer); evaluator
default sampling covered only the first 2000 structures (the 8 fs checkpoint
numbers in results/eval/pretrained_8fs.csv are over that biased sample and
should be rerun with --every 2500); catalogue grouping merged adjacent
same-composition structures (cell volume added to the key); configs lacked
atomic_types; water checkpoint only written when EQUIL_STEPS was a multiple of
2000; driver launch raced the i-PI socket.
