# E19: do the pretrained features depend on direction, and can directions be read out of them?

Status: agreed (17 Nov 2026). Measurement only.

**Why this exists:** E18 found that from the water encoders that give a gain almost nothing can be read out: not momentum, not bond lengths, for the best one not even the element. But every quantity in E18 was one that does not change when the frame is rotated. The features are known to be full of something (about 507 independent directions of variation, steady over 16 fs, different between atoms). One candidate E18 could not see: which way things point in the frame of the simulation box, for example the direction from an atom to its neighbours. Water molecules turn slowly compared with 16 fs, both frames of a pretraining pair get the same rotation, and the task predicts arrows (displacement, momentum), so direction would be steady, allowed and useful.

**Questions:**
1. How much do an encoder's features change when the whole frame is rotated?
2. Can directions (to the nearest neighbour, of the momentum, of the displacement) be read out of the features by a linear map?

**Prediction (confidence):**
A network that is not built to be rotation-symmetric changes under rotation even when untrained, so everything is read against the untrained encoder and the encoder trained with labels.
- The direction to the nearest neighbour reads out of the water pretrained features (plain, no predictor), on rotated frames, at least as well as out of the untrained encoder's, and well above zero (over 0.3): 45% confident.
- The water pretrained encoders change under rotation at least as much as the encoder trained with labels: 55% confident.
- If the direction to the nearest neighbour scores near zero on rotated frames for the water pretrained encoders, the "direction" explanation of the water gain is dropped.

**Measured:** as described in eval/probe_direction.py.
- Change under a random rotation, relative to the spread across atoms (all atoms, and within each element). Scale: 0 = ignores direction, about 2 = unrelated to the unrotated features.
- Ridge probes for five vector quantities, scored as the mean R² of the three components, with the E18 split, row cap and choice of strength. Each is scored on the held-out frames as stored and on the same frames after a random rotation (the probe is fitted on stored frames only). The two informative ones are the direction to the nearest neighbour and the momentum change over the stride; the momentum is an input, and the displacement is close to momentum times time, so it mostly re-reads the input.

**Encoders and frames:** the same as E18 (organic: untrained, trained with labels, plain, normalised target, relative loss; water: those plus no predictor and regulariser only; validation frames).

**Controls:**
- Must work: the same frames encoded twice give identical features (the script stops otherwise).
- Must work: the momentum vector is an encoder input; the untrained and label-trained encoders should read it out clearly above zero.
- Must fail: a random unit vector per atom scores about zero.
- Reference: the encoder trained with labels must carry direction to predict arrows; its values show what "carries direction" looks like.

**Assumptions:**
1. A linear probe for the three components of a direction is a strict test: direction could be held in a form a linear map cannot read. The rotation measurement does not have this limit.
2. A high score on stored frames alone can come from the probe memorising what each atom's surroundings look like in the stored orientation; the score on rotated frames is the one to read.
3. A large change under rotation shows dependence on direction, not that the dependence is useful.

**Runs:** 2 runs of eval/probe_direction.py, minutes each on one GPU.

## Result
Run 17 Nov (results/eval/direction_water.json, direction_organic.json). Scores on rotated held-out frames; scores on stored frames were the same to 0.01 for every entry, so the probes read a real code for direction, not a memorised orientation.

| Encoder | Change under rotation (0 to about 2) | Direction to nearest neighbour | Momentum change | Momentum (input) |
|---|---|---|---|---|
| Water, untrained | 0.42 | 0.37 | 0.40 | 1.00 |
| Water, trained with labels | 1.09 | 0.84 | 0.91 | 1.00 |
| Water, plain (gain 12%) | 1.97 | 0.64 | 0.00 | 0.00 |
| Water, normalised target (17%) | 1.96 | 0.70 | 0.00 | 0.00 |
| Water, relative loss (28%) | 1.98 | 0.70 | 0.08 | 0.13 |
| Water, no predictor (27%) | 1.98 | 0.64 | 0.00 | 0.00 |
| Water, regulariser only (1%) | 1.99 | 0.33 | 0.03 | 0.20 |
| Organic, untrained | 0.76 | 0.52 | 0.34 | 1.00 |
| Organic, trained with labels | 0.75 | 0.71 | 0.93 | 1.00 |
| Organic, plain | 1.61 | 0.48 | 0.00 | 0.00 |
| Organic, normalised target | 1.77 | 0.52 | 0.02 | 0.38 |
| Organic, relative loss | 1.20 | 0.46 | 0.00 | 0.00 |

Both predictions held; the "direction" explanation survives its test.
- Every pretrained encoder's features change almost completely when the frame is rotated (water 1.96 to 1.99, where about 2 means unrelated), far more than the untrained (0.42) or label-trained (1.09) encoder. The features are tied to orientation in the box. This is why E18, which only asked for quantities unchanged by rotation, read almost nothing out of them.
- On water, the encoders pretrained with a "now against later" term hold the direction to the nearest neighbour much better than the untrained encoder (0.64 to 0.70 against 0.37). The regulariser-only encoder, the one water pretraining that gave no gain, does not (0.33).
- On organic, pretraining does not raise it above the untrained encoder (0.46 to 0.52 against 0.52).
So the pattern "pretraining adds readable direction" matches the pattern "pretraining helps": water yes, organic no, regulariser-only no.

## Follow-up, 17 Nov: short against long pretraining (same frames, E13 encoders)
Tested on: water validation frames (the same liquid the encoders were pretrained on, other runs). Bond direction score for hydrogens, rotated frames (first of three probe repeats; the others were still running when this was written):

| Encoder | Its effect on test error (E13) | Hydrogen bond direction | Change under rotation |
|---|---|---|---|
| Untrained | | 0.49 | 0.42 |
| Plain, 20 epochs | 26% worse than baseline | 0.90 | 1.81 |
| Plain, 200 epochs | 14% better | 0.89 | 1.97 |
| No predictor, 20 epochs | 4% better (seeds overlap baseline) | 0.99 | 1.95 |
| No predictor, 200 epochs | 23% better | 0.86 | 1.98 |
| Trained with labels | | 0.99 | 1.09 |

The prediction "long pretrainings score higher than short ones" failed. Readable bond direction is already there after the short pretrainings, which give no gain (plain: a loss). So readable bond direction does not distinguish the pretrainings that help from those that do not. The "direction explains the water gain" idea is not supported.

Crowding around a hydrogen (validation frames): water nearest atom 0.96 Å, second 1.53 Å, third 1.90 Å, 38 atoms within 4.5 Å; organic nearest 1.09 Å, second 1.91 Å, third 2.12 Å, 11 atoms within 4.5 Å. Organic hydrogens also have non-bonded atoms close by; the clear difference is the number of atoms in range (about 3.5 times more in water).

## Assumptions check
- Controls passed: repeat encodings identical; random unit vector scores 0.00; momentum, an input, reads out at 1.00 from untrained and label-trained encoders.
- This is an association across encoders (one encoder each), not a demonstrated cause. Among the four water encoders with a gain, the score (0.64 to 0.70) does not order them by gain (12 to 28%).
- The momentum change, the other informative quantity, cannot be read out of any pretrained encoder (0.00): pretraining removes momentum, so whatever helps is geometric direction, not motion.

## Next
1. E16 (equal training steps) decides whether there is a real gain to explain.
2. If there is: pretraining with a separate rotation for each frame of a pair (forbids direction features), with the probes repeated on that encoder to see what it learned instead.
