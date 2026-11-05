# Per-login environment for training/evaluating our models (metatrain FlashMD). Usage: source env_train.sh
# The reproduction environment (flashmd 0.1.2, PET-MAD, i-PI) stays in env.sh.
source /storage3/lauren/code/jepa/.venv-train/bin/activate
export PATH="$PATH:/storage3/lauren/code/i-pi/bin"
export HF_HOME="${HF_HOME:-/scratch/local/ssd/lauren/hf_cache}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
