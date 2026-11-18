# E21: rolling out the water baseline: where does it go wrong?

Status: proposed (18 Nov 2026). Card first; no code is written until it is agreed, and nothing runs until Lauren gives the go-ahead in writing.

**Why this exists:** the end goal is latent rollout: encode once, step forward many times in feature space, and recover positions and momenta at the end. Before building anything for that, we need to know whether our ordinary model (FlashMD, which steps in real space: positions and momenta in, positions and momenta 16 fs later out, fed back in) has a rollout problem worth fixing, and if so what kind. Every result so far is one-step error, and the literature review (Latent rollout: how other fields do it) found that one-step error is a weak guide to rollout quality.

**Question:** When our best water model is fed its own predictions for many steps, how fast does its error grow, does the temperature or energy drift, how does it fail, and do its inputs move away from what the encoder saw in training?

**What the model is:** FlashMD has no separate step in feature space. One pass is: positions and momenta now → encoder → features → output layers → positions and momenta 16 fs later. A rollout feeds that output back in as the next input.

**Model rolled out:** flashmd_water16_scratch (E00): trained on all 8,960 water training frames for 300 epochs, one seed, test momentum error 0.0038. It is our most accurate water model, so its failures are the ones that matter.

Optional extra arms, Lauren to decide (no training needed, only more rollouts):
- E16 baseline, 10% labels, 1,400 epochs (0.0061, 3 seeds): how rollout quality depends on one-step error.
- E05 plain pretrained, 1,000 epochs (0.0063, 3 seeds): pretrained and baseline with the same one-step error; does pretraining change rollout behaviour? This is the last open benefit of one-step pretraining.

**Tested on:** the 28 water test runs: the same liquid at 300 K as training, in constant-energy runs the model never saw. Each run is 2 ps long, stored every 0.25 fs (data/water, generated with i-PI and q-TIP4P/f).

**Set-up, matched across everything rolled out:** start at the first frame of each test run's constant-energy stage; 16 fs per step; no thermostat; no energy filter; the same one-step function as eval/eval_onestep.py, with the model's output fed back in unchanged.

**Measurements:**

| Part | What | Length | Compared with |
|---|---|---|---|
| A. Error growth | Position and momentum error against the reference at each step (mass-scaled, as the one-step error), by element | 2 ps (125 steps), as long as the reference | **Chaos floor**: two exact simulations from the same start, one with momenta changed by one part in a million. Their difference shows how fast any two trajectories separate even with no model error. Model error only means something above this floor |
| B. Drift and failure | Temperature over time; total energy (kinetic + potential, the potential computed by the same q-TIP4P/f driver that made the data); O–H bond lengths, H–O–H angles and the oxygen–oxygen distance distribution; the first step at which a rollout becomes unphysical (an O–H bond over 1.5 Å, two atoms closer than 0.7 Å, or temperature over 600 K) | 100 ps (6,250 steps) | Exact constant-energy dynamics keep total energy constant (the reference runs' own drift is read from i-PI's conserved-energy column); distributions from the 28 reference runs |
| C. Feature gap (Lauren's mechanism) | (1) At each step, how far the encoder's features of the predicted state are from its features of the true state at that time, divided by the spread of features across atoms (as in E08). (2) The same for the chaos-floor trajectory, which shows the gap that comes from trajectories separating, not from the model. (3) Whether the predicted states' features, taken together, have drifted to a different mean or spread from the true states' features at the same time. Part (3) does not depend on chaos, because chaos changes which state you are in but not the overall distribution | 2 ps for (1) and (2); 100 ps for (3), against the reference distribution | Gap (1) above gap (2), and gap (3) growing with steps, would mean the encoder is being fed inputs unlike its training data |

**Prediction (confidence):**
- A: model error rises above the chaos floor within the first 0.5 ps: 80% (mine).
- B: the 10-ps-averaged temperature moves by more than 10 K over 100 ps, in either direction: 55% (mine).
- B: at least one of the 28 rollouts becomes unphysical within 100 ps: 30% (mine).
- C: the predicted states' features drift away from the true states' features as the rollout goes on (gap 3 grows): Lauren's prediction; her confidence to be added. Mine: 60%.

**Controls:**
- Works: replaying the reference trajectory itself through the energy calculation reproduces the potential energy i-PI recorded, to within 1e-5 eV per frame. Otherwise the energy measurement is broken.
- Works: a rollout of one step from each start reproduces the model's one-step test error (0.0038 order). Otherwise the rollout code is broken.
- Fails: a model that always predicts "no change" must leave the chaos floor at once and fail the temperature check. Otherwise the measures cannot detect a bad model.

**Assumptions:**
1. 2 ps of reference is long enough for part A, because exact trajectories separate within about 1 ps anyway (checked by the chaos floor itself).
2. Positions are unfolded (atoms may leave the box), as in the training data. The model's output must not depend on whether a frame is wrapped back into the box: checked on 10 frames before the rollouts.
3. The model kept by metatrain (best validation loss) is the one rolled out.
4. One seed of the primary model. A single model can only show what this model does, not the spread between training seeds; the optional E16 arm has three seeds.

**What we do with the result:**
- Rollout stays close to the chaos floor, no drift, no failures: the baseline has little rollout error to remove; the case for latent rollout would rest on speed alone, and we rethink before building it.
- Drift or failures: we know when and how the baseline fails, and that becomes the number a latent rollout has to beat.
- Feature gap grows (C3): supports Lauren's distribution-shift mechanism on the real-space side; the next test is whether training on the model's own outputs for a few steps reduces it.

**Cost:** inference only: about 1 to 2 GPU hours for the primary model (the 28 starts run together), plus CPU time for the energy calculation and the 28 chaos-floor simulations (2 ps each, one thread each, as the data runs).

**Code needed after the card is agreed** (reviewed by an independent agent before anything runs): a rollout script, an analysis script for parts A to C, and a script for the chaos-floor simulations and the energy replay with i-PI and the q-TIP4P/f driver.

**Later, not now:** the same on organic molecules (their reference runs are only 392 fs long, so a longer reference would have to come from PET-MAD).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
