#!/usr/bin/env bash
# Separate environment for training with metatrain's FlashMD.
# Why separate: the reproduction environment (/storage3/lauren/code/flashmd/.venv) holds
# flashmd 0.1.2 and PET-MAD, which pin metatrain 2025.10; that version has no FlashMD
# trainer. Upgrading it in place would risk the PET-MAD 1.0.0 setup that regenerated the
# Zenodo data exactly, so training gets its own environment.
#   bash scripts/make_train_venv.sh      (once)
#   source env_train.sh                  (every login where you train or evaluate our models)
set -euo pipefail
VENV=/storage3/lauren/code/jepa/.venv-train
uv venv "$VENV" --python 3.11
source "$VENV/bin/activate"
uv pip install "metatrain==2026.4.1" wandb plotly pandas pyyaml scipy ase
python - <<'PY'
import torch, metatrain
print("metatrain", metatrain.__version__, "| torch", torch.__version__, "| CUDA build", torch.version.cuda)
import metatrain.experimental.flashmd  # the trainer we need
from metatrain.utils.augmentation import O3Augmenter  # used by checks/check_augmentation.py
ok = torch.cuda.is_available()
print("GPU visible:", ok)
if ok:
    x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item()
    print("GPU matmul OK on", torch.cuda.get_device_name(0))
else:
    raise SystemExit("torch cannot see a GPU: the driver may be older than this torch build; "
                     "paste `nvidia-smi | head -5` and the torch line above")
PY
echo "training environment ready: source env_train.sh"
