# E18: what do the pretrained encoders hold, on organic and on water? (linear probes, like for like)

Status: agreed (17 Nov 2026). Measurement only, no training of encoders.

**Why this exists:** E08 measured how much features change, not what they describe. The theory that organic and water encoders satisfy the regulariser in different ways (organic by describing what each atom is, water by describing geometry) was only rejected in its narrowest form. Probes were run on organic encoders on 10 Nov, never on water with the same list of quantities.

**Question:** For the same list of quantities, how well can each be read out of each encoder's frozen features by a linear map, on organic and on water?

**Prediction (confidence):**
- Pretrained encoders (plain, no predictor) read out current geometry (bond length, nearest atom beyond two bonds, local density) better on water than on organic, within each element: 55% confident.
- On both systems, pretrained encoders read out fast quantities (current momentum, displacement over the stride) worse than the encoder trained with labels: 80% confident.
- The regulariser-only water encoder reads out fast quantities better than plain: 70% confident.

**Quantities (per atom; all are numbers that do not depend on how the frame is rotated):**

| Kind | Quantity |
|---|---|
| What the atom is | element; number of bonded neighbours |
| Current geometry | mean bond length to bonded neighbours; distance to the nearest atom more than two bonds away (for a water hydrogen, an atom of another molecule); number of atoms within 3 Å |
| Current motion (encoder inputs) | size of the momentum; momentum along the direction to the nearest neighbour |
| The frame | kinetic temperature of the whole frame |
| What the task needs | size of the displacement over the stride; displacement along the direction to the nearest neighbour; size of the momentum change over the stride |

**Method:** ridge regression from the 640 features to each quantity, fitted separately for each element (so "which element" cannot inflate the score), on 70% of the simulation runs and scored on the other 30% (no frame or run is in both). In both systems the held-out runs are of systems the probe has seen (the same liquid; the same organic molecules, other runs), so the two are like for like and the organic scores do not include transfer to new molecules. Score: share of the variance explained (R²), per element and averaged over atoms. Element itself is scored by a probe across all atoms.

**Encoders:**
- Organic: untrained; trained with labels (10% baseline); plain, normalised target and relative loss pretrained (E01).
- Water: untrained; trained with labels (10% baseline); plain, normalised target and relative loss (E04); no predictor (E11); regulariser only (E06).
- Organic no predictor is added when E15 has produced it.

**Fixed:** validation frames only (never used for any training): organic data/organic8/val.xyz every 5th frame (groups of 50); water data/water/pairs/16fs/val.xyz every 2nd frame. The same probe settings for every encoder.

**Controls:**
- Must work: current momentum size is an input to the encoder, so the untrained encoder and the encoder trained with labels should read it out clearly above zero.
- Must fail: a random number per atom, used as the quantity, scores about zero for every encoder.
- Leak check: every score is also computed with atoms split at random (frames shared between fitting and scoring); the gap to the run-level split shows what a leak would add.
- A quantity that is the same for more than 98% of an element's atoms (for example the number of bonded neighbours of water hydrogens) is not scored for that element.

**Assumptions:**
1. A linear probe shows what is easy to read out, not everything the features hold.
2. A group is one run: 50 consecutive frames in the organic file, 80 in the water file (40 pairs with their reversed copies); the script stops if the file is not a whole number of groups. Organic keeps forward and reversed frames (every 5th frame); water with every 2nd frame keeps forward frames only.
3. Frame temperature on water is fitted on about 10 runs, so its score is noisy.
4. The regularisation strength of each probe is chosen from five values on an inner split of its own training atoms (the held-out runs play no part). Every probe is fitted on at most 2,000 atoms (the same cap for organic and water), because a probe fitted on fewer rows overfits more and would score lower for that reason alone; the number of training atoms per element is recorded, and the random-number control is reported per element.

**Measure and rule:** the table of scores; differences read only if larger than 0.05 in R² (no seeds here; one encoder each).

**Runs:** 2 runs of eval/probe_compare.py, minutes each on one GPU.

## Result
Run 17 Nov (results/eval/probe_organic.json, probe_water.json). Share of variation explained on held-out runs (1 = read out perfectly, 0 = not at all), averaged over elements:

| Encoder | Element | Bond length | Nearest atom beyond two bonds | Atoms within 3 Å | Momentum size | Displacement size | Momentum change size |
|---|---|---|---|---|---|---|---|
| Organic, untrained | 1.00 | 0.56 | 0.36 | 0.90 | 0.76 | 0.38 | 0.18 |
| Organic, trained with labels | 1.00 | 0.99 | 0.66 | 0.92 | 1.00 | 0.84 | 0.73 |
| Organic, plain | 0.95 | 0.57 | 0.24 | 0.79 | -0.06 | -0.05 | -0.03 |
| Organic, normalised target | 1.00 | 0.58 | 0.20 | 0.78 | -0.02 | -0.02 | -0.03 |
| Organic, relative loss | 0.86 | 0.60 | 0.38 | 0.80 | -0.09 | -0.08 | -0.08 |
| Water, untrained | 1.00 | 0.02 | 0.51 | 0.54 | 0.82 | 0.21 | 0.15 |
| Water, trained with labels | 1.00 | 0.92 | 0.79 | 0.57 | 1.00 | 0.51 | 0.35 |
| Water, plain | 0.99 | 0.00 | 0.09 | 0.29 | 0.00 | 0.00 | 0.00 |
| Water, normalised target | 1.00 | 0.05 | 0.31 | 0.25 | 0.13 | 0.08 | 0.00 |
| Water, relative loss | 0.94 | 0.00 | 0.33 | 0.31 | 0.00 | 0.00 | 0.00 |
| Water, no predictor | 0.20 | 0.00 | 0.05 | 0.17 | 0.00 | 0.00 | 0.00 |
| Water, regulariser only | 0.01 | 0.00 | 0.10 | 0.02 | 0.23 | 0.06 | 0.01 |

Bonded neighbours are not scored for water (every hydrogen has one, every oxygen two). Frame temperature scores about 0 for every water encoder including the trained one: water runs differ little in temperature, so there is nothing to read out; not interpreted.

Predictions:
1. "Pretrained encoders read out current geometry better on water than on organic": failed. Organic pretrained encoders keep geometry (bond length 0.57 to 0.60, atoms within 3 Å about 0.8); water pretrained encoders keep almost none (bond length 0.00 to 0.05).
2. "Pretrained encoders read out fast quantities worse than the encoder trained with labels": held, more strongly than expected. In both systems pretraining with a prediction term removes momentum from the features entirely (0.00 against 0.76 to 0.82 untrained), although momentum is an input to the encoder.
3. "Regulariser only reads out fast quantities better than plain on water": held (momentum 0.23 against 0.00).

The water encoder that gave the largest gain (no predictor, 27%) is the one from which the least can be read: not even element (0.20), and nothing about motion or bond lengths.

## Assumptions check
- Controls passed: a random number per atom scores 0.00 for every encoder; momentum size, an encoder input, reads out at 0.76 to 1.00 from untrained and trained encoders.
- A linear probe shows what is easy to read out. Information may still be present in a form a linear map cannot use; but the labelled training starts its heads from these features, and the heads are small networks.

## Next
Taken with E13 and the pending E16: the water gain is not explained by useful information in the features. Candidate explanation to test: pretraining changes the starting weights in a way that helps optimisation (for example their scale or how the layers are conditioned), not what the features describe. E16 (equal training steps) is the first check.
