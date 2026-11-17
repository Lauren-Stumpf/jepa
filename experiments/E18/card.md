# E18: what do the pretrained encoders hold, on organic and on water? (linear probes, like for like)

Status: agreed (17 Nov 2026). Measurement only, no training of encoders.

**Why this exists:** E08 measured how much features change, not what they describe. The theory that organic and water encoders satisfy the regulariser in different ways (organic by describing what each atom is, water by describing geometry) was only rejected in its narrowest form. Probes were run on organic encoders on 10 Nov, never on water with the same list of quantities.

**Question:** For the same list of quantities, how well can each be read out of each encoder's frozen features by a linear map, on organic and on water?

**Prediction (confidence):**
- Pretrained encoders (plain, no predictor) read out current geometry (bond length, nearest non-bonded distance, local density) better on water than on organic, within each element: 55% confident.
- On both systems, pretrained encoders read out fast quantities (current momentum, displacement over the stride) worse than the encoder trained with labels: 80% confident.
- The regulariser-only water encoder reads out fast quantities better than plain: 70% confident.

**Quantities (per atom; all are numbers that do not depend on how the frame is rotated):**

| Kind | Quantity |
|---|---|
| What the atom is | element; number of bonded neighbours |
| Current geometry | mean bond length to bonded neighbours; distance to the nearest non-bonded atom; number of atoms within 3 Å |
| Current motion (encoder inputs) | size of the momentum; momentum along the direction to the nearest neighbour |
| The frame | kinetic temperature of the whole frame |
| What the task needs | size of the displacement over the stride; displacement along the direction to the nearest neighbour; size of the momentum change over the stride |

**Method:** ridge regression from the 640 features to each quantity, fitted separately for each element (so "which element" cannot inflate the score), on 70% of the molecules or runs and scored on the other 30% (no frame, molecule or run is in both). Score: share of the variance explained (R²), per element and averaged over atoms. Element itself is scored by a probe across all atoms.

**Encoders:**
- Organic: untrained; trained with labels (10% baseline); plain, normalised target and relative loss pretrained (E01).
- Water: untrained; trained with labels (10% baseline); plain, normalised target and relative loss (E04); no predictor (E11); regulariser only (E06).
- Organic no predictor is added when E15 has produced it.

**Fixed:** validation frames only (never used for any training): organic data/organic8/val.xyz every 5th frame; water data/water/pairs/16fs/val.xyz every 2nd frame. The same probe settings for every encoder.

**Controls:**
- Must work: current momentum size is an input to the encoder, so the untrained encoder and the encoder trained with labels should read it out clearly above zero.
- Must fail: a probe from features with the atoms shuffled within each element (features paired with the wrong atoms) scores about zero.

**Assumptions:**
1. A linear probe shows what is easy to read out, not everything the features hold.
2. The water validation file is grouped by run in blocks of 80 frames (40 pairs with their reversed copies); if it is not, the split is still by frame, never by atom.

**Measure and rule:** the table of scores; differences read only if larger than 0.05 in R² (no seeds here; one encoder each).

**Runs:** 2 runs of eval/probe_compare.py, minutes each on one GPU.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
