# E07: does pretraining help on molecules it has seen and hurt on new ones? (hypothesis H1)

Status: agreed (13 Nov 2026). No training: the existing E01 models are evaluated on new frames.

**Question:** On organic molecules (E01) every pretrained variant was worse than the baseline on the test set. The test set is made of molecules that appear nowhere in training. Is pretraining still worse on new frames of the molecules it was trained on?

**Hypothesis H1:** pretraining teaches the encoder the particular molecules it saw. The pretraining loss is computed only on the 168 training molecules, and nothing in it rewards features that work on a molecule it never saw. Water's test set is the same liquid as its training set, which would explain why pretraining helps there.

**Prediction (confidence):**
- If H1 is true: on unused frames of the training molecules, the pretrained arms are at least as good as the baseline, while on the test set they are worse (as already measured).
- If H1 is false: the pretrained arms are worse than the baseline on unused frames too, by about as much as on the test set (plain 4% worse).
- My confidence that H1 is true: 55%.

**Varied:** the frames the models are evaluated on. Three sets:

| Set | Molecule seen in pretraining? | Frame in the 10% labelled subset? |
|---|---|---|
| Labelled frames | yes | yes |
| Unused frames | yes | no |
| Test set (already measured) | no | no |

- Labelled frames: positions 0, 40, 80, ... in data/organic8/train.xyz (every frame at a position divisible by 10 is in the 10% subset).
- Unused frames: positions 5, 45, 85, ... (never in the subset). 2,100 frames each, spread over all 168 training molecules (the file is grouped by molecule, 500 frames each).

**Matched:** the evaluation command is the one used for the test set (eval/eval_onestep.py, mass-scaled targets, 2,100 frames, the organic config's masses).

**Fixed:** the 18 E01 models, unchanged: baseline, plain, normalised target, residual, projector and relative loss, seeds 0, 1, 2.

**Controls:**
- Must work (frame selection): the frame numbers the evaluation records are all divisible by 10 for labelled frames and all end in 5 for unused frames. Checked from each result file before reading any error.
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

**Runs:** 36 evaluations (18 models × 2 sets), about 1 minute each on one GPU. Script: scripts/eval_seen_organic.sh.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
