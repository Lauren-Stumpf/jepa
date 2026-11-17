# E19: do the pretrained features depend on direction, and can directions be read out of them?

Status: agreed (17 Nov 2026). Measurement only.

**Why this exists:** E18 found that from the water encoders that give a gain almost nothing can be read out: not momentum, not bond lengths, for the best one not even the element. But every quantity in E18 was one that does not change when the frame is rotated. The features are known to be full of something (about 507 independent directions of variation, steady over 16 fs, different between atoms). One candidate E18 could not see: which way things point in the frame of the simulation box, for example the direction from an atom to its neighbours. Water molecules turn slowly compared with 16 fs, both frames of a pretraining pair get the same rotation, and the task predicts arrows (displacement, momentum), so direction would be steady, allowed and useful.

**Questions:**
1. How much do an encoder's features change when the whole frame is rotated?
2. Can directions (to the nearest neighbour, of the momentum, of the displacement) be read out of the features by a linear map?

**Prediction (confidence):**
- Water pretrained encoders (plain, no predictor) change strongly under rotation (more than 0.5 of the spread between atoms): 60% confident.
- The direction to the nearest neighbour reads out of the water pretrained features better than out of the untrained encoder's: 45% confident.
- If both fail, the "direction" explanation of the water gain is dropped.

**Measured:** as described in eval/probe_direction.py: change under a random rotation, relative to the spread across atoms; ridge probes for five vector quantities, scored as the mean R² of the three components, with the E18 split, row cap and choice of strength.

**Encoders and frames:** the same as E18 (organic: untrained, trained with labels, plain, normalised target, relative loss; water: those plus no predictor and regulariser only; validation frames).

**Controls:**
- Must work: the same frames encoded twice give identical features (the script stops otherwise).
- Must work: the momentum vector is an encoder input; the untrained and label-trained encoders should read it out clearly above zero.
- Must fail: a random unit vector per atom scores about zero.
- Reference: the encoder trained with labels must carry direction to predict arrows; its values show what "carries direction" looks like.

**Assumptions:**
1. A linear probe for the three components of a direction is a strict test: direction could be held in a form a linear map cannot read. The rotation measurement does not have this limit.
2. A large change under rotation shows dependence on direction, not that the dependence is useful.

**Runs:** 2 runs of eval/probe_direction.py, minutes each on one GPU.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
