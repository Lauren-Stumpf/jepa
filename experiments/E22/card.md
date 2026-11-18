# E22: rolling out the water baseline: where does it go wrong?

Status: agreed (18 Nov 2026), with both optional arms. Lauren gave the go-ahead to run; the code runs only after it passes independent review.

**Why this exists:** the end goal is latent rollout: encode once, step forward many times in feature space, and recover positions and momenta at the end. Before building anything for that, we need to know whether our ordinary model has a rollout problem worth fixing, and if so what kind. Every result so far is one-step error. In the literature review ("Latent rollout: how other fields do it", 18 Nov), one paper (A control theory of predictability in latent world models, 2026) found that one-step validation error was essentially unrelated to planning success (rank correlation 0.02), while accuracy over rollouts was related (0.62).

**What the model is:** FlashMD has no separate step in feature space. One pass is: positions and momenta now → encoder → features → output layers → positions and momenta 16 fs later. A rollout feeds that output back in as the next input.

**Lauren's prediction (the mechanism this card tests):** in a rollout, the model is fed its own predictions, which are slightly unlike the true states it was trained on. If that mismatch matters, the model's one-step error should be larger on its own outputs than on true states, and the encoder's features of its own outputs should drift away from the features of true states.

**Question:** When our best water model is fed its own predictions for many steps: how fast does its error grow, is its error on its own outputs larger than on true states, do temperature and energy drift, how does it fail, and do its features drift away from those of true states?

**Models rolled out:**

| Model | Frames seen | Labels seen | Optimiser steps | One-step test error | Role | Tested on |
|---|---|---|---|---|---|---|
| flashmd_water16_scratch (E00) | 8,960 | 8,960 | 168,000 | 0.0038 (one seed) | The model studied | The 28 water test runs: same liquid at 300 K, runs never seen |
| flashmd_water16_scratch_10pct_s0 (E04 baseline, seed 0) | 896 | 896 | 5,600 | 0.0310 | Control: must be worse | Same |
| E16 baseline 1,400 epochs, seeds 0 to 2 (included) | 896 | 896 | 78,400 | 0.0061 | How rollout depends on one-step error, with three training seeds | Same |
| E05 plain pretrained, seeds 0 to 2 (included) | 8,960 (pretraining), 896 (labels) | 896 | 22,400 + 56,000 = 78,400 | 0.0063 | Same one-step error as the E16 arm: does pretraining change rollout behaviour? The last open benefit of one-step pretraining | Same |

The two optional arms differ in how the same 78,400 steps were spent (and E05 saw more frames during pretraining); they are a check of behaviour, not a clean single-variable comparison.

**Data:** each test run is 2 ps of constant-energy dynamics stored every 0.25 fs, made with the i-PI simulation program and the q-TIP4P/f water force field (an exact formula for the energy of water, cheap enough to compute the true behaviour from any state).

**Set-up, the same for every model:** start at the first frame of each test run's constant-energy stage (momenta converted from i-PI's units as in data/water/parse_runs.py); 16 fs per step; no thermostat; no energy filter (FlashMD's optional correction that rescales momenta after each step to keep the total energy constant). Each step uses the same one-step function as eval/eval_onestep.py, and its output is fed back after undoing the mass scaling exactly as eval_onestep.py does, with no other correction. A rollout stops at its first unphysical step (part B) and is left out of later measurements; the number left out is reported at every point.

**Exact dynamics used in A2 and A3:** i-PI with the q-TIP4P/f force field, 0.25 fs time step, constant energy, the same settings as the data (data/water/nve.xml.template).

**Measurements:**

| Part | What | Length | Compared with |
|---|---|---|---|
| A1. Error growth | Position and momentum error against the reference at each step (mass-scaled, as the one-step error), by element | 2 ps (125 steps), as long as the reference | A3 |
| A2. Error on its own outputs | Take the model's predicted state, run exact dynamics for 16 fs from it, and compare the model's next step with that. This is the one-step error measured on the model's own outputs. Done every 5th step for the first 2 ps, then every 50th step to 100 ps | 100 ps | First 2 ps: the model's one-step error on the true states at the same times. After that: the model's mean one-step error on the true states of the first 2 ps on these runs (expected close to its test error, 0.0038), since the liquid stays at equilibrium |
| A3. Chaos reference | For each start frame, two exact simulations: one with every position moved in a random direction by the size of the model's one-step position error (converted out of mass scaling, per element; momenta unchanged), and one rerun with no change, to measure the error from restarting at the stored precision (56 simulations in all). Their separation from the reference shows how fast two exact trajectories separate from an error of that size | 2 ps | A1, by the time for the error to double, not its level |
| B. Drift and failure | Per run: temperature averaged over 10-ps windows; total energy (kinetic plus potential, the potential computed by the same force field); O–H bond lengths and H–O–H angles (bonds found by molecule, distances to the nearest periodic copy of each atom); oxygen–oxygen distance distribution; the first unphysical step (an O–H bond over 1.5 Å, two atoms closer than 0.7 Å, or an instantaneous temperature over 600 K) | 100 ps (6,250 steps) | Exact constant-energy dynamics keep total energy constant (the reference runs' own drift is read from i-PI's conserved-energy column); distributions pooled from the 28 reference runs |
| C. Feature drift | For each element and each 10-ps window: the distance between the mean feature vector over all predicted states (pooled over the 28 rollouts) and the mean feature vector over all true states (pooled over the 28 reference runs), divided by the typical spread of features across atoms (as in E08). Noise level: the same distance between the reference pool and resamples of it (the 28 reference runs drawn with replacement, pooled the same way). The reference runs are 2 ps long, so this noise level is larger than for a 10-ps window, which makes the rule cautious | 100 ps | Drift counts if the distance exceeds 3 times the noise level |

**Predictions (confidence):**
- A2: the model's one-step error on its own outputs is larger than on true states. For the first 2 ps, "larger" means the mean of the per-run differences exceeds 3 standard errors of those differences; after 2 ps, the mean over the 28 runs exceeds the model's one-step error on true states (above) by more than 3 standard errors across runs. Measured on momentum error, with position error reported beside it: Lauren's prediction, her confidence to be added; Claude's 60%.
- A1 against A3: model error doubles faster than the chaos reference over the first 0.5 ps: Claude's 50%.
- B: the 10-ps-averaged temperature, averaged over the runs, changes between the first and last window by more than 10 K and by more than 3 standard errors across runs: Claude's 55%. Total energy drift is reported on the same scale: 3.9 meV per molecule equals 10 K of kinetic energy.
- B: at least one of the 28 rollouts becomes unphysical within 100 ps: Claude's 30%.
- C: features drift by the rule above in at least one element: Lauren's prediction, her confidence to be added; Claude's 60%.

**Controls:**
- Works: the rollout code run for one step on the 1,120 test pairs reproduces eval/eval_onestep.py's predictions to floating-point precision.
- Works: replaying the reference frames through the energy calculation reproduces the potential energy i-PI recorded, to within 1e-5 eV per frame; and 16 fs of exact dynamics from a stored frame reproduce the stored frame 16 fs later to within 1e-5 Å. Exact dynamics are run with the same force driver program as the data, from Python (eval/qtip4pf_ase.py).
- Works: the model's output does not change when a frame is wrapped back into the box (the data's positions are unwrapped, so atoms may leave the box), checked on 10 frames.
- Must fail (drift rule): the primary model's rollout with every momentum multiplied by 1.00002 after each step. Over 6,250 steps this adds about 85 K worth of kinetic energy; at constant energy about half of it moves into potential energy, so the temperature rises by about 40 K. The drift rule in B, applied to the difference from the unperturbed rollout of the same model, must flag it.
- Must fail (failure detector): the same with momenta multiplied by 1.01 after each step (past 600 K within about 35 to 70 steps). The failure detector must stop it.
- Must be worse: the E04 baseline (seed 0, one-step error 0.0310) must have a larger A1 error than the primary model at every step of the first 0.5 ps, averaged over runs.

**Assumptions:**
1. 2 ps of reference is long enough for A1 and A3; whether exact trajectories separate within that time is measured by A3, not assumed.
2. model.pt is the epoch metatrain kept: the lowest product of validation position and momentum errors.
3. One seed of the primary model: this shows what this model does, not the spread between training seeds (the optional E16 arm has three seeds).

**What we do with the result:**
- Error close to the chaos reference, no drift, no failures: the baseline has little rollout error to remove; the case for latent rollout would rest on speed alone, and we rethink before building it.
- Drift or failures: we know when and how the baseline fails, and that becomes the number a latent rollout has to beat.
- A2 and C show the model is worse on its own outputs: supports Lauren's mechanism on the real-space side; the next test is whether training on the model's own outputs for a few steps reduces it.

**Cost:** GPU, inference only: about 1 to 2 hours each for the primary model, the E04 control and the drift must-fail rollout (the 28 starts run together); the failure-detector rollout stops within minutes; 6 to 12 hours more if both optional arms (6 models) are run. CPU: the force-field energy of the rolled-out frames; about 4,200 short exact simulations of 16 fs for A2 (700 in the first 2 ps, 3,500 after); 56 exact simulations of 2 ps for A3 (one thread each, as the data runs).

**Code** (reviewed by an independent agent before anything runs): eval/rollout_water.py (rollouts and the one-step and wrap checks), eval/qtip4pf_ase.py (exact force field and its checks), eval/rollout_exact.py (A2, A3, energies), eval/rollout_analysis.py (every measure, verdict and figure), scripts/run_e22.sh (runs it all in order and stops at the first failed check).

**Later, not now:** the same on organic molecules. Their reference runs are only 392 fs long, so a longer reference would have to be computed with PET-MAD (the machine-learned potential that generated the organic data).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
