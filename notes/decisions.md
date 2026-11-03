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

## 2026-11-03: second review, fidelity to the paper's training recipe

Adopted:
- Targets stored mass-scaled in data/organic8 (future_positions := q + dq*sqrt(m),
  future_momenta := p'/sqrt(m); input momenta raw). With these, metatrain's
  PositionAdditive + scaler + MSE computes exactly the paper's Eq. 6, and the
  logged validation RMSEs are in the paper's units. eval_onestep.py undoes the
  scaling with --scaled_targets.
- Per-atom wrap check (any atom displaced > 2 A over the stride) and a
  molecule-spans-half-the-box check in the split; the rms-based broken filter
  would have missed a single wrapped atom.
- Masses set explicitly in the configs to the i-PI values the file was
  generated with (H 1.00794, C 12.0107, N 14.00674, O 15.9994); metatrain
  otherwise uses ASE masses and rejects the file's masses at inference.
- Every epoch's checkpoint kept; final selection by the paper's criterion
  (position RMSE x momentum RMSE x energy RMSE of predicted structure under
  PET-MAD 1.0.0) is done afterwards by a separate script.
- Augmentation check (checks/check_augmentation.py) must pass before training.

Known, accepted differences from the paper (apply equally to baseline and to
our model): metatrain uses 1% warmup then cosine decay instead of step decay;
the loss weights each atom equally rather than each structure; no
centre-of-mass constraint in the model (subtract net momentum per step in the
rollout evaluator). The Zenodo file (25 pairs per run every 100 fs) is not the
dataset described in App. B.3 (5 per run every 500 fs); we reproduce the
released-checkpoint version. The organic-vacuum subset is easier than the
paper's full MAD mix, so its errors are not comparable with Fig. 8; the fair
comparison is baseline versus our model on the same test split. The released
checkpoint was trained on these same structures, so its number on our test
split is a training-set number.

## 2026-11-03: organic subset preparation rewritten after a third review

A reviewing agent checked the metatrain and i-PI sources and found that the
unwrapping, recentring and nearest-image placement added over the previous two
days were unnecessary: metatrain's only geometric input is a neighbour list
with cell shifts (vesin), which gives bit-identical edge vectors whether or not
atoms sit inside the box (tested on shifted and triclinic cases); its position
target is the stored future minus now with no wrapping anywhere; and i-PI
writes unfolded positions (folding is applied only to the copy sent to the
force driver). The review also found that dropping 2,395 "fragmented" frames
had silently changed the distribution the authors trained on, and that the
selection rules were retyped across five scripts.

Data path now: Zenodo 8 fs file -> data/catalogue.py (one pass, per-frame
table) -> data/prepare_organic.py -> data/organic8/{train,val,test}.xyz,
frames.csv, manifest.json. The five selection scripts and test_unwrap.py are
removed; checks/test_prepare.py covers the new one.

Structure rules (ours, not the paper's; every exclusion is printed and listed
in manifest.json with its reason): elements within H, C, N, O and containing
carbon; periodic cell with more than 60 A^3 per atom (the printed distribution
shows where the threshold sits); non-periodic entries excluded and counted;
the minimum-atom-count rule is off by default because no reason for it was
written down; where the file holds the same molecule twice (same formula,
atom count and cell volume), the lowest group id is kept and the other copy
excluded, so the model never sees two copies of one molecule. Split: random
by structure, seed 0, 80/10/10.

Frame rules: none, except exploded reference runs (kinetic temperature above
5000 K or raw per-atom displacement above 3 A over 8 fs; the distributions
are printed before the rule is applied). Hot frames where the molecule has
come apart and structures that see their own periodic image are kept, because
the authors trained on them and the comparison is baseline versus our model on
the same frames. The script stops if a kept frame has a raw displacement
larger than half the shortest cell height (evidence of after-the-fact
wrapping); none is expected.

Recorded per frame in frames.csv, never used to drop: number of bonded pieces
(1.25 x covalent radii, periodic), number of neighbour pairs with a nonzero
cell shift at the model cutoff read from the config yaml, raw max
displacement, kinetic temperature, split, whether written.

Targets stored mass-scaled with the config masses (largest relative
difference to the file's masses printed). manifest.json holds sha256 of the
three xyz files, frames.csv, the catalogue and the config, plus the source
file size, so both models can be shown to have read identical data.
