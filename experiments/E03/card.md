# E03: freezing the encoder at the start of fine-tuning (organic, 10% of the labels)

Status: done. Card written before running, in conversation on 11 November; copied here afterwards.

**Question:** Does protecting the pretrained encoder at the start of fine-tuning let pretraining help?

**Varied:** starting encoder (random, or plain pretrained), within each procedure:
- A: encoder frozen for all 100 epochs, only the heads train. One seed.
- B: heads only for 10 epochs, then everything for 90, starting from stage 1's last epoch. Three seeds.

The existing E01 runs (everything trains from step 1) are procedure C.

**Fixed:** 8,400 labelled pairs, 100 epochs in total, E01 settings.

**Rule (set in advance):** a difference counts if larger than about 0.0006 in momentum error (about three seed spreads).

## Result
| Procedure | Random start | Plain pretrained | Difference |
|---|---|---|---|
| A (test momentum) | 0.2448 | 0.2425 | none: neither learns the task |
| B (test momentum) | 0.0238 | 0.0264 | pretrained 11% worse |
| C (test momentum, E01) | 0.0226 | 0.0234 | pretrained 4% worse |

Under B, the pretrained arm fits the training pairs better (0.0159 against 0.0187) and new molecules worse.

## Assumptions check
- The heads alone cannot do the task from either encoder, so procedure A measures nothing useful about feature quality with these small heads.
