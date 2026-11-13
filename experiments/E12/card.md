# E12: does organic pretraining help when its pairs are 100 fs apart instead of 8 fs? (hypothesis H2)

Status: agreed (13 Nov 2026).

**Question:** Organic pretraining at 8 fs did not help (E01). Over 8 fs, organic features barely change (copying error 0.4% of the spread), so there may be nothing to predict. If pretraining pairs are far enough apart that features must change, does pretraining help the same 8 fs labelled task?

**Hypothesis H2:** the organic pretraining task was too easy to solve by describing what each atom is; a longer gap forces the encoder to describe motion and geometry, which the labelled task needs.

**Prediction (confidence):**
- If H2 is true: fine-tunes from the 100 fs pretraining beat E01 plain (test momentum 0.0234) and the baseline (0.0226).
- If H2 is false: they land at or above E01 plain.
- My confidence that H2 is true: 30%.

**Varied:** only the gap of the pretraining pairs: 8 fs (E01 plain) against 100 fs (this experiment). The pairs come from the same organic training frames: each frame paired with the frame two places later in the same constant-energy run (data/make_gap_pairs.py), which is 100 fs later.

**Matched:** everything else as E01 plain pretraining: plain predictor, weights 25/25/1, 20 epochs, batch 64 pairs, seed 0, all 168 training molecules; fine-tuning exactly as E01 (8 fs, 10% of the labels, 100 epochs, seeds 0, 1, 2).

**Controls:**
- Reference arms already run: baseline (E01, test momentum 0.0226), plain at 8 fs (E01, 0.0234).
- Must work: in pretraining, copying error relative to the feature spread is clearly larger than the 0.4% of E01 (otherwise the longer gap did not make the task harder).
- Must work: the pair file's exclusion counts are small and explained (pairs need both frames in the same run).

**Assumptions:**
1. Start frames in a run are 100 fs apart (notes/decisions.md); the 50-frame runs are confirmed by constant energy (E10).
2. About 4% fewer pretraining pairs than at 8 fs (the last two frames of each 50-frame run have no partner 100 fs later); counted in the data manifest.
3. 100 fs is one choice; organic motion at 100 fs includes several bond vibrations, so the future is still determined (constant-energy runs) but much less like the present.

**Measure and rule:** test momentum error, mean of three seeds, against both references. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** data preparation (minutes), 1 pretraining (about 7 hours on a 24 GB card), 3 fine-tunes (about 5.5 hours each). See `runs.txt`.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
