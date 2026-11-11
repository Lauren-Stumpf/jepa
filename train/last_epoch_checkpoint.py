"""
Copy a finished metatrain FlashMD checkpoint so that fine-tuning from it starts from the
LAST epoch's weights instead of the best-validation epoch's.

Why: metatrain's fine-tuning loads `best_model_state_dict` (the epoch with the lowest
validation error), not the final weights. For the heads-first procedure (stage 1: heads
only for 10 epochs; stage 2: everything for 90 epochs) stage 2 must start from the end
of stage 1, so both arms get exactly 10 epochs of heads-only training whatever epoch
happened to score best.

The copy keeps everything else in the checkpoint and carries over the fine-tuning
setting stored with the best weights (metatrain re-applies it when loading).

  python train/last_epoch_checkpoint.py results/runs/<stage1>/model.ckpt results/runs/<stage1>/model_last.ckpt
"""

import os
import sys

import metatomic.torch  # noqa: F401  (registers the metadata class stored in metatrain checkpoints)
import torch

src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu", weights_only=False)
best, last = ckpt["best_model_state_dict"], ckpt["model_state_dict"]
new = dict(last)
if "finetune_config" in best and "finetune_config" not in new:
    new["finetune_config"] = best["finetune_config"]
missing = [k for k in best if k not in new]
assert not missing, f"keys in the best weights but not the last: {missing[:5]}"
changed = sum(1 for k, v in last.items()
              if torch.is_tensor(v) and not (torch.equal(v, best[k]) if not v.is_floating_point() else torch.allclose(v, best[k], rtol=0, atol=0, equal_nan=True)))
print(f"{src}: best epoch {ckpt.get('best_epoch')}, last epoch {ckpt.get('epoch')}; "
      f"{changed} weight tensors differ between them; writing the last epoch's weights as the start point")
ckpt["best_model_state_dict"] = new
ckpt["best_epoch"] = ckpt.get("epoch")
torch.save(ckpt, dst + ".tmp")   # then rename: a crash mid-write never leaves a truncated dst
os.replace(dst + ".tmp", dst)
print(f"-> {dst}")
