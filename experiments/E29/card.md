# E29: does stepping in feature space accumulate less error than stepping in real space? (small molecules, fixed neighbours)

Status: proposed (23 Nov 2026). Card first; no code is written until it is agreed, and nothing runs until Lauren gives the go-ahead in writing. Decisions 1 to 7 and the two rules below were agreed by Lauren on 23 Nov.

**Why this exists:** the end goal is a model that steps forward in feature space, because that might accumulate less error than decoding to positions and momenta and encoding again at every step. No paper we found shows this; the one direct comparison (LE-PDE, fluid simulations) found real-space stepping slightly better. This experiment tests it with pretraining kept out entirely.

**Question:** for one trained model, is the error against the exact trajectory after several steps smaller when the steps are taken in feature space than when the state is decoded and encoded again at every step?

**The model: three parts, trained with labels only (no pretraining)**
- Encoder: FlashMD's encoder, positions and momenta to features per atom.
- Step: a new small network with message passing between atoms, features now to features one step later.
- Decoder: FlashMD's output layers, features to positions and momenta.

**The two ways of running the same model (agreed rule 1)**
- Real space: encode, step, decode, then encode again, step, decode, and so on.
- Feature space: encode once, step, step, ..., decode once.

**Training on several steps (agreed rule 2):** the model is run for all the steps in its own way of running, and the loss compares only the final decoded state with the true state.

**Why small molecules:** in liquid water almost every atom gains or loses a neighbour within one 16 fs step (97% of atoms; measured 23 Nov, results/eval/neighbour_change_water.json), and a step in feature space cannot update the list of neighbours because it has no new positions. In a molecule where every pair of atoms always stays within the 4.5 A cutoff, the list never changes. That keeps this problem out of the first test. Five of our 210 organic molecules qualify (checked on every stored frame, results/eval/organic_molecule_sizes.json): C2H4O (7 atoms), CH2O2 (5 atoms, two entries), CH3 (4 atoms), CHN (3 atoms). Proposed: C2H4O and the two CH2O2 entries.

## Stages

| Stage | What | Passes when |
|---|---|---|
| 1 | Reference trajectories: for each molecule, constant-energy runs with the simulator that made the organic data (PET-MAD), stored every 8 fs, long enough for 64 consecutive steps from many starts; split into training, validation and test runs | Rerunning a stored 8 fs step from our organic file reproduces the stored future (as checked on 30 Oct); total energy is conserved within each run |
| 2 | One-step training of the three-part model and of unchanged FlashMD on the same data | The three-part model's one-step test error is within the seed spread of FlashMD's. If not, the step network is changed before going on |
| 3 | The four arms below | |

## Arms (stage 3; 3 seeds each; same data, same model size, same total optimiser steps)

| Arm | Trained on | Loss | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|---|---|
| R1 | 1 step | after 1 step | all training runs | all | S | F1 | test runs of the same molecules, never seen |
| F1 | 1 step (identical to R1 for one step, so R1 and F1 are the same trained model) | after 1 step | same | same | S | R1 | same |
| R4 | 4 steps in real space | final state only | same | same | S | F4 | same |
| F4 | 4 steps in feature space | final state only | same | same | S | R4 | same |

Each trained model is rolled out both ways at test time, so there are six rollouts: the 1-step model both ways, R4 both ways, F4 both ways.

What does each compared arm get that the other does not? R4 against F4: only whether the state is decoded and encoded again between steps during training. Real-space against feature-space rollout of one model: only that, at test time. The 1-step model against the 4-step models: the number of steps trained on.

**Measure:** error of positions and of momenta (mass-scaled, as everywhere) against the exact trajectory after 1, 4, 16 and 64 steps, from the same test starts, mean over starts, 3 seeds.

**Rule:** one way of running beats the other at a given number of steps when every seed of it beats every seed of the other.

**Predictions (confidence, Claude's; Lauren's to add):**
- The 1-step model rolled out in feature space is worse than in real space at 4 steps and beyond: 75%. (Its step network has never been given its own output. Lauren predicted this on 18 Nov.)
- F4 in feature space is no worse than R4 in real space at 4 steps: 55%.
- F4 in feature space is better than R4 in real space at 16 and 64 steps (seed rule): 30%.

**Controls:**
- Must work: at 1 step, both ways of running any model give the same error (they are the same computation).
- Must fail: a model whose step network is replaced by copying (features unchanged) must be clearly worse than every trained arm at 4 steps.
- Reference: exact dynamics from starts perturbed by the size of the model's one-step error gives the error growth that chaos alone would produce.
- Noise: seed spread of the one-step model before any difference is read.

**Assumptions:**
1. Three small molecules are a much easier problem than a liquid. A result here shows whether stepping in feature space can work when neighbours are fixed, not that it works for liquids. That is a later experiment.
2. The step network adds parameters to FlashMD; stage 2 checks that one-step accuracy is unchanged, not that the two models are the same size.
3. Training on 4 steps is one choice; other numbers of steps are not tested here.
4. A loss on the final state only is the agreed rule; a loss at every step is not tested here.

**Cost:** to be estimated after stage 1 is designed in detail. The molecules have 5 to 7 atoms, so training runs are far cheaper than any so far.

**Code needed after the card is agreed** (each piece reviewed by an independent agent before it runs): a script that generates and checks the reference trajectories; the step network and the two ways of running inside the FlashMD model; training on several steps with the loss on the final state; a rollout and scoring script.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
