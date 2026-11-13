"""
Fix a crash in metatrain 2026.4.1 that happens after every training run has saved and exported
its final model.

After training, metatrain looks through the whole config for a key named like "batch_size" to
use in its own final evaluation (metatrain/cli/train.py, _get_batch_size_from_hypers). It calls
key.lower() on every key, but our configs give the atomic masses as a dictionary keyed by
atomic number (1, 6, 7, 8), which are numbers, so it stops with
    AttributeError: 'int' object has no attribute 'lower'
The process then exits with an error, and Weights and Biases marks the run "failed" although
training finished and model.ckpt and model.pt were written. The fix turns each key into text
before comparing. It changes nothing that happens before the final model is saved.

Safe to run more than once: it does nothing if the fix is already there, and stops if the
code does not look exactly as expected (for example after a metatrain upgrade).

  python scripts/patch_metatrain.py          # in the training environment (source env_train.sh)
"""

import os
import shutil
import sys

import metatrain

OLD = '            key.lower().replace("_", "").replace("-", "").replace(" ", "")\n'
NEW = '            str(key).lower().replace("_", "").replace("-", "").replace(" ", "")\n'


def main():
    path = os.path.join(os.path.dirname(metatrain.__file__), "cli", "train.py")
    text = open(path).read()
    if text.count(NEW) == 1 and text.count(OLD) == 0:
        print(f"already fixed: {path}")
        return
    if text.count(OLD) != 1:
        sys.exit(f"unexpected code in {path}: found the line to fix {text.count(OLD)} times, expected 1; nothing changed")
    tmp = path + ".patch_tmp"
    with open(tmp, "w") as f:
        f.write(text.replace(OLD, NEW))
    shutil.copymode(path, tmp)   # keep the original file's permissions
    os.replace(tmp, path)
    print(f"fixed: {path}")


if __name__ == "__main__":
    main()
