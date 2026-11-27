# E28: rolling out the water baseline the way the FlashMD paper does (energy correction and thermostat)

Status: done (run 23 Nov, read 26 Nov 2026). Card written afterwards: the runs were agreed in conversation ("lets get this done", 23 Nov) before a card existed.

**Update, 26 Nov (independent review of the plans, then checked by Claude):**
- This is not yet the paper's full set-up. At inference the paper also (1) keeps the total momentum of the box equal to its input value and makes the centre of mass move uniformly (its appendix A.3), and (2) applies a random rotation before each model step and the inverse rotation after it (appendix A.4). eval/rollout_water.py does neither.
- Measured on the saved rollouts: the model does not keep the total momentum. In the data it is zero. In the rollout with no correction, 32% of the kinetic energy at 100 ps is the whole box moving in one direction. With the energy correction it is 0.7%, and with a thermostat 0.7% (an equal share for three degrees of freedom out of 576 would be 0.5%).
- "Both elements should have the same temperature" (written below) is wrong for the data. With the total momentum fixed at zero, an atom of mass m in a box of total mass M has mean kinetic energy lower by the factor (1 - m/M). For 64 molecules that makes oxygens 1.4% colder and hydrogens 0.1% colder: a gap of 3.9 K at 300 K. The 7.0 K (standard error 2.9) of the stored frames agrees with that.
- Prediction for the exact 100 ps control, written before it is run: hydrogens 3.9 K hotter than oxygens. If so, the model's 12.2 K under the energy correction is 8 K or more too large and is a fault of the model.

**Why this exists:** E22 rolled the water baseline out with no correction and it cooled by 164 K in 100 ps. The FlashMD paper (arXiv 2505.19350) never runs the model that way. After each step it rescales the momenta so that the total energy is unchanged, and it adds a thermostat. It then compares averaged properties (temperature of each element, structure), not trajectories. Without the same set-up we cannot compare our model with the paper's.

**Question:** with the paper's energy correction, with a thermostat, and with both, does the rolled-out liquid hold its temperature for 100 ps, for each element?

**Model:** flashmd_water16_scratch (E00): trained with labels on all 8,960 training frames, one-step test error 0.0038, one seed. No pretraining anywhere in this experiment.

**Tested on:** the 28 water test runs (the same liquid at 300 K as training, runs never seen). Each rollout starts from the first frame of a test run: 16 fs per step, 6,250 steps, 100 ps.

| Arm | Energy correction | Thermostat | Total energy needs the exact force field |
|---|---|---|---|
| No correction (E22) | no | no | no |
| Energy correction | yes | no | yes |
| Energy correction and thermostat | yes | yes | yes |
| Thermostat only | no | yes | no |

- Energy correction: after each step every momentum of a run is multiplied by alpha = sqrt(1 - (E' - E) / K'). E is the total energy before the step, E' after it, K' the kinetic energy after it. The potential energy comes from the exact q-TIP4P/f force field.
- Thermostat: Langevin, time constant 100 fs, target 300 K, half a thermostat step before and half after each model step (as in the paper).
- Code: eval/rollout_water.py with --conserve_energy, --langevin_fs 100, --target_T 300. Reviewed twice by an independent agent; the two fixes from the second review were applied without a third review.

**Prediction:** none was written down before the runs. That breaks the rule of this folder.

## Result
Temperatures in kelvin, means over the 28 rollouts. Results in results/eval/e28/.

| Arm | First 10 ps | Last 10 ps | Change (standard error) | Hydrogen, 50 to 100 ps | Oxygen, 50 to 100 ps | Hydrogen minus oxygen (standard error) |
|---|---|---|---|---|---|---|
| No correction (E22) | 290.8 | 126.7 | -164.1 (9.9) | 132.3 | 188.2 | -55.9 (11.6) |
| Energy correction | 302.7 | 303.6 | +0.9 (0.8) | 308.5 | 296.4 | +12.2 (0.9) |
| Energy correction and thermostat | 299.5 | 300.6 | +1.1 (0.4) | 300.1 | 300.2 | -0.2 (0.4) |
| Thermostat only | 299.4 | 299.6 | +0.3 (0.5) | 299.7 | 299.8 | -0.1 (0.4) |
| Exact dynamics, the stored test frames (first 2 ps only) | 299.9 | | | 302.2 | 295.2 | +7.0 (2.9) |

- No rollout stopped or became unphysical in any arm (28 of 28 reached 100 ps).
- The energy correction is small at each step: the momenta change by 0.15% on average, and no step failed to be rescaled.
- The energy correction alone removes the cooling. So the cooling in E22 was a loss of total energy, as the paper says of uncorrected runs.
- With the energy correction alone, hydrogens are 12 K hotter than oxygens, and the gap is there from the start (9.4 K in the first 2 ps) and steady (10.7 K at 10 to 30 ps, 12.8 K at 70 to 90 ps).
- The thermostat alone is enough for the temperature of both elements.

### Exact reference and scores (run 26 Nov, read 27 Nov)
Exact dynamics for 100 ps from the same 28 starts (eval/exact_reference_water.py): both checks passed (the stored frame at 16 fs is reproduced; total energy after 100 ps differs from the start by 0.006 meV per molecule on average, at most 0.07). Total momentum stays zero. Scores from eval/rollout_scores.py (results/eval/e28/scores.json); both scripts were reviewed twice. Temperatures, shape and structure are over 50 to 100 ps; diffusion is over the whole run.

| Set-up | Hydrogen minus oxygen, K (standard error) | Height of the first oxygen-oxygen peak (at 2.78 A) | Structure difference from exact (O-O; noise level 0.0035) | Oxygen diffusion, A^2 per ps (standard error) |
|---|---|---|---|---|
| Exact | +1.6 (1.0) | 3.27 | reference | 0.178 (0.007) |
| Model, energy correction | +11.9 (0.9) | 3.39 | 0.0177 | 0.148 (0.008) |
| Model, thermostat only | +0.1 (0.4) | 3.31 | 0.0052 | 0.091 (0.002) |
| Model, energy correction and thermostat | -0.1 (0.4) | 3.32 | 0.0079 | 0.091 (0.002) |
| Model, no correction | -55.9 (11.6) | 5.86 | 0.2563 | 0.018 (0.002) |

- Prediction for the exact runs (hydrogen minus oxygen 3.9 K): measured +2.9 (0.8) over 2 to 50 ps and +1.6 (1.0) over 50 to 100 ps. Close to the prediction but not equal to it (the second window is 2.3 standard errors below). Either way the model's +11.9 K is 8 to 10 K too large: a fault of the model.
- Molecule shape is right in every corrected set-up (O-H bond 0.9639 against 0.9633 A; angle 104.79 against 104.79 degrees).
- Under the energy correction the model's liquid is too ordered (peak 3.39 against 3.27, five times the noise level) and its oxygens move 17% more slowly, although it is 4 K hotter, which on its own would make it more disordered and faster.
- With a thermostat, diffusion is about half of the exact value. A 100 fs Langevin thermostat slows diffusion by itself, so this cannot be read as a model fault until exact dynamics are run with the same thermostat (not yet done).

### One-step error along the energy-corrected rollout (run 26 Nov, read 27 Nov)
16 fs of exact dynamics from the model's own state at saved steps, compared with the model's next state (eval/rollout_exact.py, existing code). Mass-scaled momentum error:

| States | Momentum error (standard error over runs) |
|---|---|
| True states, first 2 ps | 0.00375 (0.00003) |
| Model's own states, first 2 ps | 0.00376 (0.00002) |
| Model's own states, 2 to 50 ps | 0.00380 (0.00002) |
| Model's own states, 50 to 100 ps | 0.00380 (0.00002) |

After 100 ps of being fed its own states the one-step error is 1% higher than on true states. Without the cooling of E22 to confuse the comparison, being fed its own states does not make the model's steps worse. Its faults are small biases in every step (energy lost, energy moved to hydrogens, a liquid that is too ordered) that add up over many steps.

## Assumptions check
- The exact starting frames already have hydrogens 7 K hotter than oxygens (standard error 2.9 K, over 2 ps). An earlier version of this line said both elements should reach the same temperature in exact dynamics. That was wrong: see the update at the top (3.9 K is expected in a box of this size with the total momentum fixed). The exact dynamics for 100 ps have not been run yet.
- The rollouts do not fix the total momentum or apply the random rotation, so "paper-style" in the title is only partly true (update at the top).
- Temperature of each element is the only averaged property read so far. The structure of the liquid (distances between atoms) and how fast molecules move through it have not been compared with exact dynamics.
- The thermostat replaces part of each atom's momentum with random numbers every step, so the thermostat arms say nothing about whether the model's dynamics are right, only about its averages.
- One model, one training seed.

## Next
- Our baseline behaves as the paper describes: without correction it loses energy, with the paper's corrections it is stable for 100 ps.
- Done 27 Nov: exact dynamics for 100 ps from the same starts (see the result above).
- Still missing: exact dynamics with the same thermostat, to see how much of the slower diffusion under a thermostat is the thermostat; the paper's total-momentum constraint and random rotation; the spread of every score between training seeds.
- For any rollout comparison (E29), the corrected set-up is the reference, and the averaged properties to compare are those of the paper.
