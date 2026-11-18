# E22: rolling out the water baseline: where does it go wrong?

Status: proposed (18 Nov 2026). Card first; no code is written until it is agreed, and nothing runs until Lauren gives the go-ahead in writing.

**Why this exists:** the end goal is latent rollout: encode once, step forward many times in feature space, and recover positions and momenta at the end. Before building anything for that, we need to know whether our ordinary model has a rollout problem worth fixing, and if so what kind. Every result so far is one-step error. In the literature review ("Latent rollout: how other fields do it", 18 Nov), one paper (A control theory of predictability in latent world models, 2026) found that one-step validation error was essentially unrelated to planning success (rank correlation 0.02), while accuracy over rollouts was related (0.62).

**What the model is:** FlashMD has no separate step in feature space. One pass is: positions and momenta now → encoder → features → output layers → positions and momenta 16 fs later. A rollout feeds that output back in as the next input.

**Lauren's prediction (the mechanism this card tests):** in a rollout, the model is fed its own predictions, which are slightly unlike the true states it was trained on. If that mismatch matters, the model's one-step error should be larger on its own outputs than on true states, and the encoder's features of its own outputs should drift away from the features of true states.

**Question:** When our best water model is fed its own predictions for many steps: how fast does its error grow, is its error on its own outputs larger than on true states, do temperature and energy drift, how does it fail, and do its features drift away from those of true states?

**Model rolled out:** flashmd_water16_scratch (E00): trained on all 8,960 water training frames for 300 epochs, one seed, test momentum error 0.0038. It is our most accurate water model, so its failures are the ones that matter.

Optional extra arms, Lauren to decide (no training needed, only more rollouts):
- E16 baseline, 10% labels, 1,400 epochs (0.0061, 3 seeds): how rollout quality depends on one-step error, with three training seeds.
- E05 plain pretrained, 1,000 epochs (0.0063, 3 seeds): pretrained and baseline with the same one-step error; does pretraining change rollout behaviour? This is the last open benefit of one-step pretraining.

**Tested on:** the 28 water test runs: the same liquid at 300 K as training, in constant-energy runs the model never saw. Each run is 2 ps long, stored every 0.25 fs. The runs were made with the i-PI simulation program and the q-TIP4P/f water force field (an exact, cheap formula for the energy of water, so a reference can be computed for any state).

**Set-up, the same for everything rolled out:** start at the first frame of each test run's constant-energy stage (momenta converted from i-PI's units as in data/water/parse_runs.py); 16 fs per step; no thermostat; no energy filter. Each step uses the same one-step function as eval/eval_onestep.py, and its output is fed back after undoing the mass scaling exactly as eval_onestep.py does, with no other correction.

**Measurements:**

| Part | What | Length | Compared with |
|---|---|---|---|
| A1. Error growth | Position and momentum error against the reference at each step (mass-scaled, as the one-step error), by element | 2 ps (125 steps), as long as the reference | Chaos reference (A3) |
| A2. Error on its own outputs | At every 5th rollout step, run exact dynamics (i-PI and the force field) for 16 fs from the model's own predicted state, and compare the model's next step with it. This is the one-step error measured on the model's own outputs | 2 ps (25 points per run) | The same model's one-step error on the true states at the same times |
| A3. Chaos reference | Exact simulations from each start frame with every position moved randomly by the size of the model's one-step position error. Their separation from the reference shows how fast two exact trajectories separate from an error of that size. Compared by growth rate (time for the error to double), not level. Plus one exact rerun with no change, to measure the error from restarting at the stored precision | 2 ps | A1 |
| B. Drift and failure | Per run: temperature averaged over 10-ps windows; total energy (kinetic plus potential, the potential computed by the same force field); O–H bond lengths and H–O–H angles (bonds found by molecule, distances to the nearest periodic copy of each atom); oxygen–oxygen distance distribution; the first step at which a rollout becomes unphysical (an O–H bond over 1.5 Å, two atoms closer than 0.7 Å, or an instantaneous temperature over 600 K) | 100 ps (6,250 steps) | Exact constant-energy dynamics keep total energy constant (the reference runs' own drift is read from i-PI's conserved-energy column); distributions pooled from the 28 reference runs |
| C. Feature drift | For each element, the mean and spread of the encoder's features over all predicted states in each 10-ps window, against the same statistics for the true states pooled over the 28 reference runs. Noise level: how much these statistics vary between the 28 reference runs | 100 ps | Drift counts if the gap exceeds 3 times the noise level |

**Prediction (confidence):**
- A2: the model's one-step error on its own outputs is larger than on true states, by more than the spread across runs: Lauren's prediction, her confidence to be added; Claude's 60%.
- A1 against A3: model error doubles faster than the chaos reference over the first 0.5 ps: Claude's 50%.
- B: the 10-ps-averaged temperature, averaged over the 28 runs, changes between the first and last window by more than 10 K and by more than 3 times its standard error across runs: Claude's 55%. Total energy drift is reported against the same scale: a change of 3.9 meV per molecule equals 10 K of kinetic energy.
- B: at least one of the 28 rollouts becomes unphysical within 100 ps: Claude's 30%.
- C: features drift by the rule above in at least one element: Lauren's prediction, her confidence to be added; Claude's 60%.

**Controls:**
- Works: the rollout code run for one step on the 1,120 test pairs reproduces eval/eval_onestep.py's predictions to floating-point precision. Otherwise the rollout code is broken.
- Works: replaying the reference frames through the energy calculation reproduces the potential energy i-PI recorded, to within 1e-5 eV per frame. Otherwise the energy measurement is broken.
- Works: the model's output does not change when a frame is wrapped back into the box (positions in the data are unwrapped and atoms may leave the box), checked on 10 frames.
- Must fail: the same rollout with every momentum multiplied by 1.01 after each step. Temperature must rise by about 2% per step and part B must flag it. Otherwise part B cannot detect drift.
- Must be worse: the 10% baseline at 100 epochs (0.0303) must show faster error growth in A1 than the primary model. Otherwise A1 cannot tell models apart.

**Assumptions:**
1. 2 ps of reference is long enough for parts A1 to A3; whether exact trajectories separate within that time is measured by A3 itself, not assumed.
2. model.pt is the epoch metatrain kept: the lowest product of validation position and momentum errors.
3. One seed of the primary model: this shows what this model does, not the spread between training seeds (the optional E16 arm has three seeds).

**What we do with the result:**
- Error close to the chaos reference, no drift, no failures: the baseline has little rollout error to remove; the case for latent rollout would rest on speed alone, and we rethink before building it.
- Drift or failures: we know when and how the baseline fails, and that becomes the number a latent rollout has to beat.
- A2 and C show the model is worse on its own outputs: supports Lauren's mechanism on the real-space side; the next test is whether training on the model's own outputs for a few steps reduces it.

**Cost:** inference only: about 1 to 2 GPU hours for the primary model (the 28 starts run together). CPU: the force-field energy of 28 × 6,250 frames, 700 short exact simulations for A2 (16 fs each), and 56 exact simulations of 2 ps for A3 (one thread each, as the data runs).

**Code needed after the card is agreed** (reviewed by an independent agent before anything runs): a rollout script, an analysis script for parts A to C, and a script for the exact simulations and the energy replay with i-PI and the force field.

**Later, not now:** the same on organic molecules. Their reference runs are only 392 fs long, so a longer reference would have to be computed with PET-MAD (the machine-learned potential that generated the organic data).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
