# E28: rolling out the water baseline the way the FlashMD paper does (energy correction and thermostat)

Status: done (run 23 Nov, read 26 Nov 2026). Card written afterwards: the runs were agreed in conversation ("lets get this done", 23 Nov) before a card existed.

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

## Assumptions check
- The exact starting frames already have hydrogens 7 K hotter than oxygens (standard error 2.9 K, over 2 ps). So the 12 K gap under the energy correction cannot yet be called a fault of the model. In exact dynamics both elements should reach the same temperature; we have not run the exact dynamics for 100 ps to see whether they do.
- Temperature of each element is the only averaged property read so far. The structure of the liquid (distances between atoms) and how fast molecules move through it have not been compared with exact dynamics.
- The thermostat replaces part of each atom's momentum with random numbers every step, so the thermostat arms say nothing about whether the model's dynamics are right, only about its averages.
- One model, one training seed.

## Next
- Our baseline behaves as the paper describes: without correction it loses energy, with the paper's corrections it is stable for 100 ps.
- Missing control: 100 ps of exact dynamics from the same 28 starts, read the same way, to see what the hydrogen and oxygen temperatures should be. Cheap (the exact force field), not yet run.
- For any rollout comparison (E29), the corrected set-up is the reference, and the averaged properties to compare are those of the paper.
