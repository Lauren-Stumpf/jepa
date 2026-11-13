# E07: does pretraining help on molecules it has seen and hurt on new ones? (hypothesis H1)

Status: agreed (13 Nov 2026). No training: the existing E01 models are evaluated on new frames.

**Question:** On organic molecules (E01) every pretrained variant was worse than the baseline on the test set. The test set is made of molecules that appear nowhere in training. Is pretraining still worse on new frames of the molecules it was trained on?

**Hypothesis H1:** pretraining teaches the encoder the particular molecules it saw. The pretraining loss is computed only on the 168 training molecules, and nothing in it rewards features that work on a molecule it never saw. Water's test set is the same liquid as its training set, which would explain why pretraining helps there.

**Prediction (confidence):**
- If H1 is true: on unused frames of the training molecules, the pretrained arms are at least as good as the baseline, while on the test set they are worse (as already measured).
- If H1 is false: the pretrained arms are worse than the baseline on unused frames too, by about as much as on the test set (plain 4% worse).
- My confidence that H1 is true: 55%.

**Varied:** the frames the models are evaluated on. Four sets:

| Set | Molecule seen in pretraining? | Frame in the 10% labelled subset? | Direction |
|---|---|---|---|
| Labelled frames | yes | yes | forward |
| Unused, forward | yes | no | forward |
| Unused, reversed | yes | no | time-reversed copy |
| Test set (already measured) | no | no | both, alternating |

- The organic files store each start frame (forward, even position) followed by its time-reversed copy (positions 8 fs later, momenta flipped; checked on 13 Nov). The 10% subset is every 10th frame, so it holds forward frames only.
- Labelled: positions 0, 40, 80, ... in data/organic8/train.xyz. Unused forward: positions 2, 42, 82, ... Unused reversed: positions 5, 45, 85, ... (added before any result was read: the reversed set was the first choice, the forward set was added on 13 Nov when the layout was found). 2,100 frames each, spread over all 168 training molecules.
- The primary comparison is unused forward against the test set; unused reversed shows whether direction matters.

**Matched:** the evaluation command is the one used for the test set (eval/eval_onestep.py, mass-scaled targets, 2,100 frames, the organic config's masses).

**Fixed:** the 18 E01 models, unchanged: baseline, plain, normalised target, residual, projector and relative loss, seeds 0, 1, 2.

**Controls:**
- Must work (frame selection): the frame numbers the evaluation records are all divisible by 10 for labelled frames, all end in 2 for unused forward frames and all end in 5 for unused reversed frames. Checked from each result file before reading any error.
- Must work (subset): data/organic8/train_10pct.xyz has 8,400 frames and its second frame equals frame 10 of train.xyz.
- Expected, not a pass/fail test: error on labelled frames at or below error on unused frames, and both below the test set. Labelled and unused frames are neighbours in the same runs, so the first two may be nearly equal.
- Reference: the straight-line guess, printed by the evaluation for each set.

**Assumptions and confounds:**
1. Unused frames come from the same simulation runs as labelled frames, so they are close to frames both arms trained on with labels. This lowers both arms' error.
2. Pretraining used all of train.xyz, so the pretrained arms saw the unused frames (without labels) and the baseline did not. This favours the pretrained arms on unused frames. It is part of what H1 claims ("pretraining helps on what it saw"), but this experiment cannot separate "seen molecule" from "seen frame".
3. The sets hold different molecules: 168 training molecules against 21 test molecules. Comparing the percentage against the baseline on the same set removes overall difficulty, but a gap that depends on the kind of molecule would still differ between sets for that reason alone.

**Measure and rule:**
- Primary measure: momentum error per arm, mean of three seeds, on unused frames, as a percentage against the baseline; compared with the same percentage on the test set.
- A difference counts if every seed of one arm beats every seed of the other (proposed 13 Nov; the older rule, a difference larger than three times the larger seed spread, is reported too).

**Runs:** 54 evaluations (18 models × 3 sets), about 1 minute each on one GPU. Script: scripts/eval_seen_organic.sh.

## Result
Run 13 Nov (results/eval/errors_by_frame_type_*.json). Momentum error, mean of three seeds, percentage against the baseline on the same frames:

| Arm | Labelled frames | Unused, forward | Unused, reversed | Test (new molecules) |
|---|---|---|---|---|
| Baseline | 0.0175 | 0.0201 | 0.0197 | 0.0226 |
| Plain | -4.0% | +0.3% | -0.3% | +3.7% |
| Relative loss | -4.6% | +0.8% | +1.0% | +4.2% |
| Normalised target | +4.0% | +3.4% | +4.4% | +6.6% |
| Residual | +4.3% | +4.3% | +5.9% | +6.8% |
| Projector | +17.2% | +17.2% | +15.4% | +19.0% |

By the seed rule (every seed of one arm beats every seed of the other): on the test set every variant is worse than the baseline; on unused frames of the training molecules plain and relative loss overlap the baseline (no difference), and the other three are still worse.

H1 is partly supported and does not explain the organic-against-water difference:
- New molecules do cost pretraining something: each variant's deficit is 2 to 4 points larger on new molecules than on unused frames of seen molecules.
- But on seen molecules, organic pretraining gives no gain at best (plain, relative loss) and is still worse for the other variants. On water, the same variants give 12 to 28% gains. So "the water test is the same system" accounts for a few points, not for the 15 to 30 point difference between the systems.
- Forward and reversed unused frames give the same picture, so direction does not matter.

## Assumptions check
- Frame selection control passed: all 54 result files have 2,100 frames with the expected frame numbers.
- Error order as expected: labelled below unused below test for every arm.
- Assumption 2 (pretraining saw the unused frames) favours the pretrained arms on unused frames; even with that advantage they gain nothing.

## Next
H1 is not the main cause. Remaining candidates: what slow features are worth for each labelled task (E11 on water; E12 forces faster features on organic), and the other differences between the systems (labelled set size: 8,400 organic pairs against 896 water pairs; one liquid against many molecules).
