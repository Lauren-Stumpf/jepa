# E17: does water pretraining help when all the frames are labelled?

Status: agreed (17 Nov 2026).

**Why this exists:** every water gain so far is with 896 labelled frames. One open explanation is that pretraining only helps because labels are few. With all 8,960 frames labelled, pretraining and the baseline see the same frames, and labels are ten times more plentiful.

**Question:** With all 8,960 water frames labelled, are the pretrained arms still better than the baseline?

**Prediction (confidence):**
- The gain shrinks to under a third of its 10% value (plain under 4%, no predictor under 9%): 65% confident.
- No pretrained arm passes the seed rule against the baseline: 50% confident.

**Varied:** the starting encoder: random (baseline), plain pretrained (E04's pretraining), no-predictor pretrained (E11's pretraining). Both pretrainings used all 8,960 frames, the same frames the baseline now trains on.

**Matched:** labelled training on all 8,960 frames, 100 epochs (56,000 steps), batch 16, seeds 0, 1, 2, every other setting as the 10% runs.

**Controls:** reference: the existing full-data water baseline at 300 epochs (0.0038, one seed). The 100-epoch baseline here should land above it.

**Assumptions:**
1. 100 epochs, not 300, to keep each run to about 10 hours; all three arms use the same length.
2. The pretrained arms have 22,400 extra pretraining steps on top of 56,000 labelled steps (40% more). E16 measures how much extra steps matter at 10%; noted here, not corrected.
3. One pretraining seed per variant.

**Measure and rule:** test momentum error, mean of three seeds. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** 9 runs of about 10 hours, 46 GB cards.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
