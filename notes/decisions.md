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

## 2026-11-03: water pairs from our own q-TIP4P/f runs

data/water/parse_runs.py turns the i-PI runs into FlashMD pairs, one dataset
per stride (1, 4, 16 fs, as in the paper's water experiments). From each 2 ps
constant-energy run a pair is taken every 100 fs (the spacing the Zenodo file
uses), plus its time-reversed copy, so 40 pairs per run per stride. Runs are
split 80/10/10 by run (frames within a run are correlated), seed 0. Momenta
are converted from i-PI atomic units to ASE units with the factor verified in
check_water_run.py; positions stay unfolded as i-PI writes them; targets are
mass-scaled with the config masses (i-PI's: H 1.00794, O 15.9994). Each
dataset carries a manifest with the run split and sha256 of the files.
checks/check_water_pairs.py verifies the pairs against the raw trajectories
(pairs recovered to 5e-9 A, reversed copies consistent, temperature, no
box-crossing displacement). Water runs are pinned to one thread each so 20
can run in parallel on a 64-core node. Still to do on this track: a level 1
check of the force driver (an independent q-TIP4P/f energy function
differentiated numerically against the driver's forces) and the comparison
of diffusion coefficient and vibrational spectrum with published values.

## 2026-11-04: momentum target for the 1 fs water set

metatrain's predict_momenta_as_difference adds the raw current momenta to the
model's momentum prediction; its documentation recommends it when the stride
is short against the momentum decorrelation time. From our water run the
hydrogen velocity autocorrelation first crosses zero at about 5 fs, so 1 fs is
short and 4 and 16 fs are not: the 1 fs config sets it true, the others false.
With mass-scaled targets this changes what must be stored: for the 1 fs set
parse_runs.py stores future_momenta := p0 + (p1 - p0)/sqrt(m) so the residual
metatrain learns is the mass-scaled momentum change; eval_onestep.py undoes it
with --momenta_difference, and --config makes the evaluator use the config
masses (the same ones the data were scaled with) instead of ASE's.

## 2026-11-05: water run 070 and the temperature check

Of 280 runs, 279 passed all seven per-run checks; run 070 failed only the
temperature test (run-mean 264 K against a 30 K tolerance; energy conservation,
units, structure and drift all pass). The per-run tolerance was wrong, not the
run: a constant-energy run's mean temperature is set by its total energy when
the thermostat is switched off, which is a random draw, so run means spread by
roughly 12 K (300 K / sqrt(heat capacity of 64 flexible waters in units of k)).
264 K is about 3 of those below 300 K, which among 280 runs is expected about
once. The run is kept: removing the tails would narrow the energy distribution
the protocol produces. The per-run test now only catches gross errors (60 K),
and checks/check_water_temperatures.py tests across all runs that the means are
centred on 300 K, that their spread matches the estimate, and that no run is an
outlier beyond what chance gives for 280 runs.

## 2026-11-05: water runs were not equilibrated; regenerated with equilibrated starts

Finding: the 280 first-generation runs average 294.7 +- 0.7 K, and the
run-averaged temperature falls by 5.4 +- 0.4 K during the 2 ps constant-energy
stage (297.7 K first 0.5 ps, 292.2 K last 0.5 ps), while the thermostatted
stage ends at 300.0 +- 0.1 K with no jump at the switch-over and the potential
energy is still rising (+51 +- 16 meV per box per ps) at the end of
equilibration. From equilibrium starts, constant-energy dynamics cannot change
the run average, so the starts were not equilibrated: every run began from the
same external file and had only 5 ps of strongly damped Langevin dynamics
(tau 10 fs slows structural relaxation). An independent review agreed and ruled
out a time-step effect (a fixed offset, and measured drift is ~200 times
smaller), thermostat artefacts on sampling, the 3N vs 3N-3 convention, and the
force driver. It also found that FlashMD did not equilibrate from an outside
file: App. B.3 equilibrates a structure at experimental density with the same
force field first, then uses 5 ps of Langevin (0.5 fs, tau 10 fs) to bring it to
each temperature, then 2 ps constant energy at 0.25 fs. And that sharing one
barely relaxed start makes the runs less independent than the split by run
assumes.

The training pairs from the first generation are exact constant-energy dynamics,
but not at equilibrium at 300 K and not independent across runs, so they are
not used (kept at /storage3/lauren/data/water_v1_5ps_unequilibrated).

New protocol:
1. Density set to experimental (0.997 g/cm3, box 12.430 A, as in App. B.3; the
   developers' file was 1.010 g/cm3), by moving whole molecules (bonds unchanged;
   data/water/make_start.py).
2. One 100 ps pilot equilibration (stochastic velocity rescaling thermostat,
   tau 100 fs, which does not slow structural relaxation); its potential energy
   curve is fitted to measure the relaxation time (checks/water_equil_curve.py),
   and its last frame becomes the common equilibrated start.
3. Each run: 25 ps Langevin at 300 K (0.5 fs, tau 100 fs) from that start with
   its own seed, so runs decorrelate (separate seeds diverge within ~1 ps;
   hydrogen-bond and rotational relaxation take ~1-5 ps), then the unchanged
   2 ps constant-energy production. Langevin is kept as in the paper; tau is
   100 fs rather than 10 fs because the equilibration has to relax structure,
   not only reheat it. The 25 ps is checked against the pilot's relaxation time
   before the runs are made.
4. checks/check_water_temperatures.py now also tests that the run-averaged
   temperature is flat in time, which is the test that detects this problem.

## 2026-11-09: first comparison at 100 epochs

The organic baseline takes about 8 minutes per epoch on an A40, so the planned
300 epochs is about 40 hours. For a first comparison by 11 November the
baseline and the JEPA-pretrained model are both fine-tuned for 100 epochs with
the full warmup-plus-cosine schedule compressed into that length, so each is a
completed training run rather than an interrupted one. Both arms see the same
data for the same number of epochs; the pretrained arm has had extra compute in
pretraining, which is stated with the result. Longer runs follow later.

## 2026-11-09: JEPA pretraining design (first version) and the comparison plan

Pretraining (train/pretrain_jepa.py): the FlashMD network's per-atom "feature"
output (768 numbers per atom) is the embedding. The same encoder is applied to
the current frame (positions and momenta) and to the frame one stride later; a
3-layer MLP predictor maps the current embedding to the future one. Loss:
25 x mean squared prediction error + 25 x VICReg variance hinge (each dimension's
spread over the atoms of a batch pushed towards 1) on both embeddings + 1 x
VICReg covariance penalty on both. No momentum (EMA) encoder and no
stop-gradient: the regulariser alone prevents collapse, which is the
"world-model" design decided on 2 November (Bruno's SIGReg variant is the next
arm). Batch 64 (the regulariser is estimated over the atoms of a batch), AdamW
3e-4 with the baseline's warmup-plus-cosine schedule, 20 epochs on the same
84,000 training pairs as the baseline.

Independent review of the design led to three changes before the first run:
1. The two frames of a pair get independent random rotations, not the same one.
   A molecule barely turns in 8 fs, so with a shared rotation absolute
   orientation would be a high-variance, perfectly predictable feature and the
   predictor could satisfy the loss from it alone (PET features are only
   approximately rotation invariant). --shared_rotation keeps the old behaviour
   for a later ablation.
2. The headline pretraining diagnostic is the ratio of the predictor's error to
   the error of "future embedding = current embedding"; a ratio near 1 means
   nothing about motion was learned (element, bonding and molecule identity are
   already perfectly predictable over 8 fs and satisfy the variance terms).
   eval/probe_features.py adds a linear probe from frozen features to the
   mass-scaled displacement and future-momentum magnitudes, pretrained encoder
   versus untrained encoder; if the pretrained probe is not better, the result is
   presented as a different initialisation rather than dynamics pretraining.
   Logged per epoch: minimum per-dimension spread, fraction of dimensions with
   spread below 0.5, effective rank of the embedding covariance.
3. Main comparison at 10% of the training frames (every 10th frame of train.xyz,
   so every molecule is represented), three seeds per arm, 100 epochs each;
   pretraining effects show first at low data and three seeds give a spread.
   The two full-data runs (one seed each) are a secondary result with the
   compute mismatch stated: pretraining adds roughly 40 to 50 epochs of encoder
   compute to the pretrained arm.

Fine-tuning (configs/jepa_finetune_organic8*.yaml) is the baseline config with
`finetune: {read_from: <pretrained checkpoint>, method: full}`: same data, same
schedule, every weight trainable, heads randomly initialised; metatrain fits the
composition model and scaler at the start of fine-tuning as it does for the
baseline.

Evaluation (eval/eval_onestep.py) now reports, alongside the model, the two
trivial predictors "no motion" and "ballistic" (q + p/m dt, p unchanged) and
the errors by element; hydrogen's 8 fs is most of a stretch period, so neither
trivial predictor is good for it. A bug was fixed: with mass-scaled data files
the reference was not unscaled before comparison, so earlier smoke numbers were
not meaningful (no trained model was affected).

## 2026-11-09 (later): code review of the pretraining; rotation decision reversed

Independent code review against the metatrain and metatomic sources found that
a pretraining model built with the position and momentum targets would carry an
unfitted target scaler (all ones) in its checkpoint, and metatrain's fine-tuning
would keep it rather than fit it, while the baseline fits the scaler to the
data: the two arms would minimise different losses. Fixed by building the
pretraining model with no targets; fine-tuning then adds the heads and fits the
scaler exactly as from scratch (tested: the restarted model reports new
targets and the scaler lists them as new outputs, encoder weights preserved).

The same review disagreed with the design review on rotations, with the
stronger argument: the FlashMD heads are linear maps from the features to
Cartesian vectors, so the features must keep orientation; independent
rotations for the two frames would train the encoder to discard it and
fine-tuning would have to undo that. The orientation shortcut is three numbers
and cannot satisfy the regulariser over 640 dimensions. Decision: shared
rotation is the default (as in FlashMD training); the independent-rotation
version runs as an ablation (jepa_pretrain_indep_organic8, with its own three
10% fine-tune seeds) so the question is settled by measurement. Also from the
review: the feature dimension is 640, not 768; the evaluation-mode crash with
feature-only outputs is a metatrain bug worked around by encoding in training
mode (PET has no dropout or batch normalisation).

## 2026-11-09 (later): how the results will be read

Three layers. (1) Fine-tuned one-step error on the 21 held-out molecules,
baseline versus pretrained, three seeds at 10% data plus one pair at full data,
against a ladder of trivial predictors: no motion; straight line at the current
velocity (ballistic); straight line plus the bend from one force evaluation
(second order, PET-MAD forces at the current frame); and the exact 32-step
integration, which is the reference itself. A useful model sits between second
order and exact. Errors also split by element. (2) During pretraining, the
ratio of the predictor's error to the "future embedding = current embedding"
error. (3) Linear probes on the frozen features (eval/probe_features.py),
pretrained versus untrained encoder: dynamic targets (displacement and future
momentum magnitudes), static targets (element, bonded-neighbour count), and the
static fraction of the feature vector (variance explained by the static
descriptors alone). Lauren's request: measure how static the embedding is
before trying to fix it; a static/dynamic split of the vector, with the
predictor acting on the change (residual) of the dynamic part, is the candidate
next arm if the measurement shows the problem.

## 2026-11-09 (afternoon): projector arm

The 300-frame smoke run of the pretraining showed the untrained encoder's
features have effective rank about 7 of 640 with 71% of dimensions nearly
constant, and the covariance penalty is the largest term of the loss. In
standard VICReg the variance and covariance terms act on the output of a
projector network ("expander") on top of the encoder, not on the encoder's
features; ours act on the encoder directly, which could push the encoder in
directions that do not help the heads. Rather than guess, a third pretraining
runs alongside (jepa_pretrain_proj_organic8: projector 640 -> 2048 -> 2048,
prediction and regulariser in the projector's output space; the encoder's own
rank and dead fraction are logged separately), with its own three 10% fine-tune
seeds. Check after the first epoch of each: encoder rank rising into the
hundreds and dead fraction falling, and inv/trivial below 1.

## 2026-11-09: different-spin arm dropped

Lauren's decision: no independent-rotation arm. Giving the two frames different
random rotations teaches the encoder to be completely independent of
orientation, which the heads cannot work with; there is no reason to spend
compute confirming it. The --independent_rotations flag stays in the script but
is not run.

## 2026-11-09 (evening): the predictor's normalisation, and three clean arms

The first pretraining (jepa_pretrain_organic8) normalised the predictor's input
but not its target, so the predictor had to undo the normalisation to reproduce
a near-copy of its input; after three epochs its error was still twice that of
plain copying (inv/trivial 2.05), and about 88% of the loss was the covariance
term. Lauren's points: normalising only one side has no purpose; the current run
is not a fair baseline for a residual predictor; and the question of whether
the regulariser rather than prediction is steering the encoder should be
measured, not inferred from loss sizes.

Three clean arms, identical apart from the predictor:
  plain        no normalisation anywhere
  norm_target  one fixed normalising step at the encoder output, both frames
               (what I-JEPA and V-JEPA do; the new default)
  residual     guess = now + f(now), f's last layer started at zero so the
               predictor begins as an exact copy (Lauren's idea)
They run on torrnode7, one per GPU; their 10% fine-tunes (three seeds each) run
on torrnode12 through the launcher once each pretraining finishes. The first
run (norm_input) and the projector run continue on torrnode12 for comparison.

Measuring the pull of each term: every 200 steps the script computes, for each
weighted loss term separately, the size of its gradient over the encoder's
weights, and the cosine between the prediction gradient and the regulariser
gradient; logged to pulls.json and the training log. Rule for the weights: the
regulariser should pull just hard enough to keep the features spread out
(dead fraction near 0, rank not falling) and no harder; if its pull is several
times the prediction pull while inv/trivial does not improve, the weights come
down. The 25/25/1 weights are the image defaults and will be reset from these
measurements. The covariance term's division by 640 is the standard definition;
the balance is set through the weights, not by changing the definition.

## 2026-11-09: first pretraining cancelled

jepa_pretrain_organic8 (predictor normalises its input but not its target) is
stopped: its result would mainly reflect that flaw, and it shared GPU 1 with the
100-epoch baseline. Its fine-tune configs are removed. The full-data JEPA
fine-tune now starts from jepa_pretrain_normtarget_organic8 (normalise both
sides, the standard design).

## 2026-11-09 (evening): measured pull of the loss terms; relative prediction loss

First-epoch measurements on torrnode7 (pull = size of each weighted term's
gradient over the encoder weights, logged every 200 steps):
  residual arm: prediction's share of the total pull 0 -> 2 -> 6 -> 7% through
  the epoch; rank 6 -> 374; inv/trivial stuck at 0.996-0.999; cosine between
  prediction and regulariser pulls +0.3 falling to +0.06 (not opposed, just weak).
  normalise-both-sides: prediction share 11%, inv/trivial 1.43.
  plain: prediction share 27% (because its predictor is 2.6x worse than copying,
  so its error is large), inv/trivial 2.56.
Conclusion: with the standard 25/25/1 weights the regulariser steers the
encoder almost entirely; the predictor learns nothing about motion. Cause: the
change of an embedding over 8 fs is tiny in absolute terms, so even a predictor
that gets the change completely wrong has a small error and a small gradient.

Fix added as a fourth arm (jepa_pretrain_residrel_organic8, torrnode7 GPU 5):
residual predictor with the prediction error divided by the error of plain
copying, i.e. the loss term is inv/trivial itself. Copying scores 1 and
explaining half the change scores 0.25 whatever the absolute size of the change,
so the prediction pull is proportional to how much of the change is explained
rather than to the scale of the embedding; no stride- or dataset-specific
weight retuning. The three running arms are kept as the controls that document
the problem. Weight retuning for a next round is to be done from the logged
pulls of all four arms, not by hand tonight.

## 2026-11-09 (night): second review of the day's changes, four fixes

1. Fine-tune configs pointed to the pretrained checkpoint by a relative path,
   but each run works from its own folder; every fine-tune would have failed to
   start. Paths are now absolute. (No fine-tune had started.)
2. Resuming from a checkpoint is not used any more: metatrain's FlashMD trainer,
   restarted from a checkpoint, trains num_epochs MORE epochs on a learning-rate
   curve that no longer matches, and a restarted fine-tune loses its optimiser
   state. An interrupted run is started again from scratch (previous outputs
   moved aside). Result tables must report the run's final epoch.
3. The relative prediction loss had its denominator detached from the gradient.
   The review showed with a toy model that this gives the encoder a direct
   incentive to make the two frames' embeddings identical (delete everything
   that changes in time), growing as the change shrinks; the undetached ratio
   is scale-free and can only be lowered by making the change predictable. The
   residrel arm is restarted with the undetached version.
4. The logged inv/trivial for the relative-loss arm was divided twice.
Also: exact tmux session matching (prefix matching could have made the launcher
skip restarting the 100-epoch baseline while the 300-epoch one was alive); the
training-set path is read from the training_set block; the probe's static
fraction uses a pooled R^2 (robust to near-constant feature dimensions).
To state with the results: the three fine-tune seeds per arm share one
pretraining seed, so their spread understates the pretrained arm's variance;
and residrel differs from resid by an effective prediction weight of
25/trivial, which is thousands of times larger, under gradient clipping at 1.

## 2026-11-11: test-set evaluation, and decisions after the supervisor meeting

Test set (21 unseen molecules, 2,100 frames, eval/eval_onestep.py; results/eval/test/).
Mass-scaled RMSE, position | momentum:
  baseline 10% (seeds 0,1,2)          0.0099 0.0098 0.0102 | 0.0223 0.0224 0.0231
  plain 10%                           0.0101 0.0102 0.0102 | 0.0232 0.0237 0.0234
  normtarget 10%                      0.0102 0.0103 0.0104 | 0.0237 0.0242 0.0244
  resid 10%                           0.0105 0.0106 0.0106 | 0.0240 0.0242 0.0242
  proj 10%                            0.0118 0.0113 0.0117 | 0.0272 0.0262 0.0272
  baseline all data, 100 epochs       0.0061 | 0.0138
  baseline all data, 300 epochs       0.0062 | 0.0141
  normtarget pretrained, all data     0.0064 | 0.0144
  authors' released universal model   0.0082 | 0.0177   (it was trained on these molecules)
  straight-line guess 0.1544, momentum unchanged 0.3330
Water (test, position | momentum; straight-line guess, momentum unchanged in brackets):
  1 fs 0.0001 | 0.0004 (0.0027, 0.0547); 4 fs 0.0003 | 0.0011 (0.0364, 0.1678);
  16 fs 0.0024 | 0.0038 (0.2105, 0.2036).
Reading: on the test set, every pretrained arm is worse than the baseline at 10% data; at
full data the 3.7% validation gain of the pretrained model does not hold (test 4% worse),
and the 300-epoch baseline is no better than the 100-epoch one, so differences of a few
percent between single full-data runs are within the difference between validation and
test molecules. The full-data compute-matched comparison is not pursued (Lauren): a few
percent is not worth a day of eight cards. Earlier quoted test numbers for the 100-epoch
baseline (0.0061 | 0.0135) came from a different frame sample; the numbers above replace them.
An earlier claim that pretrained models fit training molecules better was based on seed 0
only; over three seeds only the plain variant is (within seed spread) and the others are worse.

Measured water time scales (eval/water_bond_periods.py, 5 runs): O-H stretch 3,602 cm^-1
(9.3 fs), bend 1,668 cm^-1 (20.0 fs), rocking 584 cm^-1 (57 fs). 16 fs covers 1.7 stretch cycles.

Priority list agreed: (1) frozen-encoder fine-tuning, (2) relative-loss probe and fine-tunes,
(3) test-set evaluation, (4) shuffled validation diagnostics, (5) water 16 fs mirroring
organic, (6) rollout evaluator. Parked: compute-matched full-data comparison, 10% split by
molecule, supervised-encoder check, new objectives, density functional theory comparison.

Frozen-encoder design (Lauren added procedure A): procedure A, encoder frozen for all 100
epochs (one seed); procedure B, heads only for 10 epochs then everything for 90, from stage
1's last epoch (three seeds); procedure C (everything from step 1) is the existing runs.
Random and plain-pretrained encoders go through the same metatrain fine-tuning path
(random encoder saved by train/make_random_checkpoint.py). Decision rule set in advance: a
difference counts if larger than about three seed spreads (about 0.0006 momentum).

Pretraining validation batches are now shuffled once with a fixed seed: in file order each
validation batch held one molecule, so validation variance, covariance and rank were not
comparable with training (recomputing the plain run on 640 pairs: rank 76 in file order,
182 shuffled). Prediction-error ratios are not affected.

Water pretraining batch is 8 frames (1,536 atoms), matching organic's 64 pairs x 23 atoms
(1,477) because the regulariser is estimated over the atoms of a batch. Water 10% subset
keeps every 10th forward pair together with its time-reversed copy (896 frames). Caveat:
water 10% gives 5,600 optimiser steps in 100 epochs against about 52,000 for organic 10%.

All new code was reviewed by two independent agents before running; fixes in commits
0fab73f and ea2fa62 (branch frozen-water-experiments).

## 2026-11-11 (evening): first results of the freezing experiment, recomputed diagnostics

Freezing (10% organic, validation, epoch 100; momentum | position):
  A frozen 100 epochs, random encoder     0.2486          (heads alone barely learn)
  A frozen 100 epochs, plain encoder      0.2525
  B heads first, random (3 seeds)         0.0245 +/- 0.0009 | 0.0107   (training 0.0187)
  B heads first, plain (3 seeds)          0.0268 +/- 0.0004 | 0.0115   (training 0.0159)
  C everything from step 1 (earlier)      random 0.0238 | plain 0.0245
Reading: protecting the encoder does not rescue pretraining; under heads-first the plain
arm is 9.5% worse than the random arm (four times the decision threshold), fits the
training molecules better and new molecules worse. With the encoder frozen throughout,
the heads cannot learn the task from either encoder, so the pretrained features are no
more usable by the heads than random ones.
Relative-loss fine-tunes (current procedure): 0.0246, 0.0249, 0.0254, about 5% worse than baseline.

Recomputed validation diagnostics with shuffled batches (eval/recompute_val_diagnostics.py):
the logged validation ranks (36 to 83) were an artefact of one-molecule batches; shuffled
ranks are 250 to 437 (projector encoder 40). The relative-loss variant's predictor explains
96% of the feature change on validation molecules too (error / copying 0.036 shuffled, 0.31
in file order): the earlier "it memorises training molecules" reading was wrong, and so was
the statement that prediction-error ratios are unaffected by batch makeup (they are a ratio
of per-batch means).

Feature change on water (eval/feature_change_vs_jump.py, change / spread across atoms):
untrained encoder 0.4% at 1 fs, 9.5% at 8 fs, 19% at 16 fs, 43% at 64 fs; trained water
baseline encoder 35% at 4 fs, 71% at 16 fs. So the near-zero change seen in organic
pretraining (copying error about 0.4%) is produced by the pretraining itself, not a
property of 8 fs dynamics.

Water fine-tunes use about 28 GB at batch 16: they ran out of memory on torrnode7's 24 GB
cards (two runs moved to results/runs_invalid/*_oom_torrnode7_20261111) and run on A40s on
torrnode15 and torrnode12. Launchers now claim a run atomically on the shared filesystem.

## 12 Nov 2026

New Weights and Biases project jepa-flashmd-experiments: one group per experiment (E00 to
E06), job type = arm, a "<E> summary" run per experiment with bar charts (test and validation
errors, seeds as dots) and learning curves; the card is the summary run's notes. 53 finished
runs imported from the old project's logs (scripts/wandb_import.py), which stays unchanged.
Runs that fail or restart from scratch are deleted from the new project by the supervisors
(scripts/wandb_delete_runs.py, new project only).

Water 10% test evaluation (all 1,120 test frames, same command as the full-data water test):
test momentum baseline 0.0312, plain 0.0268 (14% better), normalised target 0.0252 (19%
better), matching validation. Single seeds so far: residual 0.0249, projector 0.0291,
residual relative loss 0.0213 (32% better). Details in experiments/E04/card.md.

Started E05 (hypothesis A, 9 runs of 1,000 epochs) and E06 (hypothesis B, regulariser-only
pretraining then 3 fine-tunes) with scripts/launch_runs.sh. The E06 pretraining logs a
prediction-term gradient share of 0.00, as the card requires.

Checkpoints thinned (agreed with Lauren): in every finished fine-tuning run, per-epoch
checkpoints were deleted except every 10th epoch and the last (scripts/thin_checkpoints.sh):
4,541 files, 1,742 GB; results/runs went from 2.2 TB to 511 GB. Runs still training were
skipped and can be thinned once finished by running the script again. New runs save every
10th epoch (checkpoint_interval 10; E05 stays at 50). This does not change training:
metatrain tracks the best-validation model every epoch whatever the interval. The paper's
checkpoint selection, if run, works on a 10-epoch grid; it should also include the final
model.ckpt, because new 100-epoch runs save model_90 as their last numbered file while
thinned older runs kept model_99.

## 13 Nov 2026: organic file layout

Checked on the data: in data/organic8/*.xyz each start frame (even position within a molecule)
is followed by its time-reversed copy (positions equal to the start's stored 8 fs future,
momenta equal to minus its stored future momenta), as the FlashMD file provides. Every stored
future is correct for its own frame, so all training and evaluation so far is unaffected.
Consequences: the organic 10% subset (every 10th frame) holds forward frames only, whereas
the water 10% subset keeps each forward pair with its reversed copy (one more difference
between the two systems, same for both arms within each); E07 adds a forward-only unused set;
E08 measures forward frames only; E12 pairs reversed copies along their own direction of time.
