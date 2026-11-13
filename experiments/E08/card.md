# E08: where do encoder features get their spread, and how much do they change over the gap? (hypothesis H2)

Status: agreed (13 Nov 2026). Measurement only, no training.

**Question:** Do organic and water encoders satisfy the regulariser in different ways? The regulariser demands that each of the 640 features varies across the atoms of a batch and that no two features duplicate each other. It does not say what the variation should describe.

**Hypothesis H2:** on organic molecules the encoder can meet the regulariser by describing what each atom is (element and chemical surroundings), which does not change over 8 fs, so the future features equal the present ones and there is nothing to predict. On water, with two elements in one chemistry, it has to describe the current geometry, which does change.

**Prediction (confidence):**
- If H2 is true: organic pretrained encoders have a "moment" share of the spread below 5% and change by less than 1% over 8 fs; water pretrained encoders have a "moment" share above 20% and change by more than 10% over 16 fs.
- If H2 is false: the two systems give similar shares for their pretrained encoders.
- My confidence that H2 is true: 45%.

**Measured:**
1. Spread split: each feature vector = element average + atom's own average over time (minus element average) + the rest (this moment). Shares of the total sum of squares; they add to exactly 1.
2. Change over the gap: mean squared change of an atom's features between a frame and 8 fs (and, for water, 16 fs) later, divided by the spread across atoms within one frame; also within each element.

**Encoders:**
- Organic: untrained; trained baseline (flashmd_organic8_scratch_10pct_s0); the five E01 pretrainings.
- Water: untrained; trained baseline (flashmd_water16_scratch_10pct_s0); the five E04 pretrainings; the regulariser-only pretraining (E06).

**Fixed:** frames no encoder was trained on. Organic: data/organic8/val.xyz, every 10th frame, which are forward frames only (1,050 frames, 50 per molecule, 21 molecules). Water: the raw runs of the validation split, a start every 100 fs (28 runs, 560 frames).

**Controls:**
- Must work: the three shares add to 1.000 for every encoder.
- Must work: the untrained water encoder's change at 16 fs is close to the 19% measured on 11 Nov (eval/feature_change_vs_jump.py, other runs and spacing, so within a few points).

**Assumptions:**
1. "Which atom" is defined within one molecule (organic) or one run (water). In water, molecules move around during a 2 ps run, so an atom's own average drifts towards its element's average; this is part of what H2 says about water, not a flaw, but the water atom share depends on run length.
2. Both systems are measured on forward frames only: water from raw runs, organic skipping the stored time-reversed copies (which have flipped momenta and would add spread that is not motion).

**Measure and rule:** the shares and changes per encoder, compared between organic and water pretrained encoders, with the untrained and trained baselines as references. No seeds: one encoder each.

**Runs:** 2 runs of eval/feature_decomposition.py (organic 7 encoders, water 8 encoders), minutes each on one GPU.

## Result
Run 13 Nov (results/eval/decomp_organic.json, decomp_water.json). Shares of the feature spread, and the change over the pretraining gap (8 fs organic, 16 fs water) as a fraction of the spread across atoms:

| Encoder | Organic: element / atom / moment | Organic change, 8 fs | Water: element / atom / moment | Water change, 16 fs |
|---|---|---|---|---|
| Untrained | 0.13 / 0.67 / 0.21 | 3.4% | 0.16 / 0.30 / 0.54 | 20.5% |
| Trained baseline | 0.12 / 0.53 / 0.35 | 35% | 0.30 / 0.29 / 0.41 | 55% |
| Plain | 0.02 / 0.26 / 0.73 | 0.46% | 0.00 / 0.47 / 0.52 | 1.6% |
| Normalised target | 0.01 / 0.17 / 0.82 | 1.0% | 0.01 / 0.43 / 0.57 | 2.1% |
| Residual | 0.00 / 0.21 / 0.79 | 0.38% | 0.00 / 0.44 / 0.56 | 1.5% |
| Projector | 0.05 / 0.44 / 0.52 | 0.45% | 0.13 / 0.63 / 0.24 | 0.9% |
| Relative loss | 0.01 / 0.44 / 0.55 | 3.9% | 0.00 / 0.37 / 0.63 | 31% |
| Regulariser only (water) | | | 0.00 / 0.09 / 0.91 | 159% |

The prediction failed: H2 as written is not supported.
- Organic pretrained encoders do not describe "which atom": their moment share is 52 to 82%, not below 5%.
- Water pretrained encoders do not change more over the gap: 0.9 to 2.1% at 16 fs, not above 10%.
- The two systems look alike. In both, pretraining (with a prediction term) produces features that differ a lot between moments (frames 100 fs or more apart, other runs, other temperatures) but barely move over the pretraining gap: slow features.
- The exceptions point the same way. Relative loss, the only variant whose predictor beat copying, has fast features for hydrogen (89% change on water, 53% on organic) and slow ones for heavy atoms. The regulariser-only encoder, with no term pulling now and later together, has the fastest features of all (159%) and gave no gain (E06). This supports H5 on water.

## Assumptions check
- Controls passed: the shares add to 1.000 for every encoder; the untrained water encoder changes 20.5% over 16 fs (19% measured on 11 Nov with other runs).
- The "moment" part mixes fast motion with slow differences between runs (temperature, conformation), because frames are 100 fs or more apart. The change over the gap is the measure of fast motion; together they say "varies between moments, slowly".

## Next
H2 in its "organic features describe atom identity" form is dropped. The 100 fs organic pretraining (E12) still tests whether a longer gap forces faster, more useful features. H5 (E11) tests whether slowness alone gives the water gain. The organic-against-water difference has to come from elsewhere: H1 (E07), or from what slow features are worth for each labelled task.
