# E27: water, longer runs: does the pretraining lead grow until it pays back its cost?

Status: first two arms done (launched 21 Nov, read 23 Nov 2026); third arm launched 26 Nov.

**Question:** at 134,400 total optimiser steps, does the pretrained model beat a baseline given the same total steps?

**Why:** at equal labelled steps the plain pretrained model is ahead by an amount equal to 1.2 to 1.3 times the labelled steps. 22,400 pretraining steps bought back about 1,200 labelled steps in the 100-epoch run and about 18,100 in the 1,000-epoch run (E16), where the two arms came out level. If the lead keeps growing in proportion, pretraining wins in a longer run. If it is a fixed lead, it never does.

**Tested on:** water test frames (the same liquid as training, runs never seen), 10% of the labels (896 frames).

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|
| Baseline, 2,400 epochs | 896 | 896 | 134,400 | plain | water test frames |
| Plain pretrained (the E04 encoder), then 2,000 epochs | 8,960 without labels, then 896 | 896 | 22,400 + 112,000 = 134,400 | baseline | water test frames |
| Added 26 Nov: plain pretrained on the 896 labelled frames only (the E13 encoder, 200 epochs), then 2,000 epochs | 896 without labels, then the same 896 | 896 | 22,400 + 112,000 = 134,400 | baseline, and the arm above | water test frames |

What does each arm get that the other does not? The pretrained arm spends its first 22,400 steps on pretraining over 8,960 frames, and has two warm-up and decay cycles of the learning rate; the baseline spends those steps on labels and has one cycle. E13 showed pretraining on the 896 labelled frames alone gives the same result, so the extra frames are not expected to matter.

**Third arm, added 26 Nov after the first result:** the line above was wrong for long runs (see the assumptions check). The third arm is pretrained on the 896 labelled frames only, so it sees exactly the frames the baseline sees. Against the baseline it differs in the kind of its first 22,400 steps (pretraining in place of labels) and in having two warm-up and decay cycles of the learning rate. Against the second arm it differs in one thing: the frames pretraining saw (896 against 8,960; the same number of pretraining steps, so 200 passes over each frame against 20). Launched 26 Nov on Lauren's instruction ("go ahead on that experiment").

**Prediction for the third arm (confidence, Claude's; Lauren's to add):** it beats the baseline on every seed: 55%. It is worse than the second arm on every seed: 60%. Reason for the first: the baseline's problem in long runs is memorising 896 frames, and a pretrained start changed how closely the model fits its training frames; that could come from the start and not from the extra frames. Reason for the second: in E15 (organic), 200 passes of pretraining over the labelled frames made the result 8 to 9% worse.

**Prediction (confidence, Claude's):** the two arms overlap (no winner by the seed rule): 55%. Pretrained wins: 25%. Baseline wins: 20%. A proportional lead of 1.3 predicts the pretrained arm about 5% better; a fixed lead predicts it about 2% worse.

**Controls:** the 2,400-epoch baseline must be better than the 1,400-epoch baseline (0.0061, E16). If not, the baseline has stopped improving and the comparison is between finished models.

**Assumptions:** one pretraining seed. A learning rate that decays over the run means a longer run is not the first part of a shorter one.

**Measure and rule:** test momentum error, 3 seeds per arm; one arm wins when every seed of it beats every seed of the other.

**Runs:** experiments/E27/runs.txt, 6 runs of about 22 to 26 hours on 46 GB cards (torrnode15), and 3 more for the third arm (about 21 hours each). The pretraining for the third arm already exists (E13).

## Result
First two arms, test momentum error, read 23 Nov. Tested on water test frames (the same liquid as training, runs never seen), 10% of the labels, 134,400 total optimiser steps each, 3 seeds.

| Arm | Seeds | Mean | Error on its own training frames at the end | Validation error at the end | Saved epoch |
|---|---|---|---|---|---|
| Baseline, 2,400 epochs | 0.0072, 0.0055, 0.0069 | 0.0065 | 0.0022, 0.0036, 0.0022 | 0.0072, 0.0055, 0.0070 | 2355, 2390, 2321 |
| Plain pretrained, then 2,000 epochs | 0.0046, 0.0045, 0.0046 | 0.0046 | 0.0040, 0.0039, 0.0039 | 0.0046, 0.0045, 0.0046 | 1998, 1997, (seed 2 not read) |

- Every pretrained seed beats every baseline seed: 30% better at equal total steps. This is the first comparison at equal total steps that pretraining wins.
- The control failed: the 2,400-epoch baseline (0.0065) is not better than the 1,400-epoch baseline (0.0061, E16).
- The reason is visible in the training errors. Two baseline seeds fit their 896 training frames three times better than they fit new frames (0.0022 against 0.0072). The pretrained models fit the training frames less closely (0.0039) and new frames better (0.0046). In long runs the baseline memorises its training frames and the pretrained model does not.
- Error by element (seed 0; hydrogen, oxygen): baseline 0.0083, 0.0043; pretrained 0.0051, 0.0033.

Third arm (pretraining on the 896 labelled frames only): launched 26 Nov; result to be added.

## Assumptions check
- "The extra frames are not expected to matter" (written above from E13, which used 100-epoch runs) does not carry over to long runs. The pretrained arm saw 8,960 frames without labels and the baseline 896. Once the baseline is limited by data, those extra frames could be the whole effect. The third arm tests this.
- A longer run is not the first part of a shorter one (the learning rate decays over each run's own length), so the 2,400-epoch and 1,400-epoch baselines are different trainings, not one curve.
- One pretraining seed.

## Next
Wait for the third arm. If it also beats the baseline, the gain comes from the pretraining objective and not from the extra frames. If it does not, the gain was the extra frames, which a baseline could also use with labels.
