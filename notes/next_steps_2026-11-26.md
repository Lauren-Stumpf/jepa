# Plans for next steps, 26 Nov 2026 (written for an independent review)

## Objective

Lauren's two research questions, developed separately:
- (a) Does stepping in feature space accumulate less error than stepping in real space?
- (b) What would a useful pretraining objective be?

Her stated first priority: an evaluation set-up that shows when we make progress.

The model is FlashMD (metatrain `experimental.flashmd`, PET backbone): it reads positions and momenta and predicts them one large step later (16 fs for liquid water, 8 fs for organic molecules). The water data are 64 molecules of q-TIP4P/f water at 300 K, constant energy, 2 ps runs; the exact force field is cheap, so the true dynamics can be computed from any state.

## What we know (numbers on the cards in experiments/)

- Pretraining (JEPA style) gives no gain at equal data and equal optimiser steps (E15, E16, E20, E21), except in long runs on 896 labelled water frames, where the pretrained model is 30% better (E27). That arm saw 8,960 frames against the baseline's 896; a third arm pretrained on the 896 frames only is running.
- The water baseline rolled out with no correction cools by 164 K in 100 ps (E22). With the FlashMD paper's energy correction (all momenta of the box multiplied by one number after each step so that the total energy is unchanged) it holds 303 K. With a Langevin thermostat (100 fs) both elements sit at 300 K (E28).
- With the energy correction only, hydrogens are 12.2 K (standard error 0.9) hotter than oxygens over 50 to 100 ps. The exact stored frames (first 2 ps only) show 7.0 K (standard error 2.9).
- The model's own energy error at each step, before the correction: mean -0.045 meV per molecule (a loss), root mean square 0.445 meV per molecule. Kinetic energy per molecule at 300 K is 116 meV.
- Against the true trajectory from the same start: after one step the position error is 0.0045 A for hydrogens (the atom moved 0.28 A) and 0.0009 A for oxygens; momentum error 2.6% and 1.7% of the typical momentum. After 0.5 ps the position error is about half the distance moved; after 1 ps it equals the distance moved and the momentum error is 141% of the typical momentum (unrelated vectors).
- Momentum error doubles every 48 fs for the model and every 272 fs for exact dynamics started from positions shifted by the model's one-step error (E22).
- The model's one-step error on its own outputs is not larger than on true states (E22, first 2 ps: difference -0.000016, standard error 0.000015). After 2 ps the comparison is confounded by the cooling.
- In water 5.3% of neighbour pairs (cutoff 4.5 A) change after one 16 fs step and 27% after 64 steps.

## Plan 1: evaluation set-up for rollouts (first priority)

Three levels, each answering a different question:

| Level | Question | Measure | Valid for |
|---|---|---|---|
| One step | Is each prediction right? | Position and momentum error after one step, from true states and from the model's own states along a rollout (exact dynamics for 16 fs from the model's state) | Any time in the rollout |
| Short path | How fast does the rollout leave the true path? | Error against the exact path from the same start, compared with exact dynamics from a slightly shifted start | About the first 1 ps |
| Long averages | Is the liquid it produces the right liquid? | Energy error per step, temperature of each element, shape of the molecules, structure (how often two atoms are found at each distance), movement (mean square displacement of oxygens, diffusion coefficient) | The whole run |

Choices made:
- The main set-up is "energy correction only": stable for 100 ps, no random numbers, and the correction is one number per box per step, so it cannot hide errors in positions, in the direction of momenta, or in how energy is shared between atoms. A thermostat acts on every atom with random numbers and is expected to hide such errors, so temperature under a thermostat is not used as a measure of progress.
- Reference: exact dynamics for 100 ps from the same 28 starts (eval/exact_reference_water.py, new). Scores: eval/rollout_scores.py (new).
- Then check that the scores separate a good model from a worse one, by rolling out weaker models we already have (10% of the labels, fewer epochs) in the same set-up.
- Proposed, not yet agreed: the one-step error along the energy-corrected rollout (existing code, eval/rollout_exact.py). In E22 this comparison was confounded by cooling after 2 ps; in the corrected rollout the liquid stays at 303 K.

## Plan 2: E27 third arm (running)

Plain pretraining on the 896 labelled frames only, then 2,000 labelled epochs, 3 seeds. If it beats the 2,400-epoch baseline on every seed, the gain in long runs comes from the pretraining objective and not from the extra frames.

## Plan 3: feature-space against real-space rollout (card E29, proposed)

One model with three parts (encoder, a step network with message passing, output layers), trained with labels and no pretraining, rolled out in two ways: encode, step, decode, repeat; or encode, step, step, step, decode. Small molecules whose neighbour list never changes (C2H4O and two CH2O2 isomers). When trained for several steps the loss is computed at the end. Error against exact (PET-MAD) trajectories after 1, 4, 16, 64 steps.

## Plan 4: direction for a paper (proposed by Lauren, sharpened by Claude)

A model that steps in feature space and takes the controls of a simulation as actions (thermostat target, electric field, box size, pulling or bias force), and is used to choose a control schedule, which is then checked in the exact simulator. Argument:
- Recent learned large-step integrators all work in real space and apply the thermostat outside the model: FlashMD (arXiv 2505.19350), learning the action (2508.01068), Hamiltonian flow maps (2601.22123), Langevin flow maps (2610.05998), Kolmogorov-Chapman maps (2611.12880). From a handful of searches, none takes a control as an input, none plans, none steps in feature space.
- In feature space there are no momenta to rescale, so a thermostat cannot be applied from outside; the control has to be an input of the step network.
- Pretrained (JEPA) features hold slow information and drop momentum; the response of a system to a control is slow information.
- Exact atom positions cannot be predicted beyond about 1 ps; averages under a control can.
Proposed smallest first test: water with the exact force field, thermostat target changed during the run on random schedules, the E29 three-part model with the control as an input to the step network; compare predicted with exact curves of averaged properties on schedules not seen in training, against FlashMD with an outside thermostat and against a simple fit.
Known risks: for a uniform liquid the response to temperature may be so simple that a two-number fit matches it; the direction replaces the pretraining question instead of answering it.

## Plan 5: pretraining (track B)

Replace linear read-outs by three functional tests of an encoder: is the frozen encoder sufficient for the one-step task with adequate output layers; is it easy to step; does it stay accurate when stepped repeatedly. Cards not yet written.

## What the review should look for

- Physics or reasoning that is wrong in the statements above.
- Measures a rollout evaluation of a learned integrator should have and this plan lacks.
- Comparisons that are confounded or cannot decide what they claim to decide.
- A cheaper or more decisive ordering of the work.
- Whether the literature claims in plan 4 hold up (prior work on control inputs, planning, or feature-space stepping for molecular dynamics).
