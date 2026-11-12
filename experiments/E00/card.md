# E00: FlashMD baselines trained on all the labels

Status: done. Card written afterwards, from the record in notes/decisions.md.

**Question:** Does our from-scratch FlashMD reproduction work on organic molecules (8 fs) and on our own water (1, 4, 16 fs)?

**Varied:** system and jump length; for organic, the number of epochs (100 against 300).

**Fixed:** FlashMD recipe as in the paper (batch 16, learning rate 0.0003, random rotations, mass-scaled loss), all training pairs.

**Controls:**
- Must work: none.
- Must fail: the straight-line guess.
- Outside reference: the FlashMD authors' released model on our organic test set.

## Result
Test errors (position | momentum, mass-scaled):

| Run | Position | Momentum |
|---|---|---|
| Organic, 100 epochs | 0.0061 | 0.0138 |
| Organic, 300 epochs | 0.0062 | 0.0141 |
| Authors' released model (organic) | 0.0082 | 0.0177 |
| Straight-line guess (organic) | 0.154 | 0.333 |
| Water 1 fs | 0.0001 | 0.0004 |
| Water 4 fs | 0.0003 | 0.0011 |
| Water 16 fs | 0.0024 | 0.0038 |

Relative to the straight-line guess, the water errors are 0.8% at 4 fs and 1.1% at 16 fs.

## Assumptions check
- Our organic test set is 21 molecules never seen in training; the authors' model was trained on them.
- 300 epochs gave no gain on test over 100.
