#!/bin/bash
#SBATCH --job-name=photobench
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/photobench-%x-%j.out
#SBATCH --error=logs/photobench-%x-%j.err
set -euo pipefail
# Usage: sbatch --job-name=album2 scripts/eval_photobench_a40.sbatch 2
ALBUM="${1:-1}"
cd "${SLURM_SUBMIT_DIR:?}"
export PHOTORECALL_KEEP_MODELS=1
export PYTHONUNBUFFERED=1 PHOTORECALL_DISABLE_OLLAMA=1
.venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; aborting')
print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__, flush=True)
PY
.venv/bin/python scripts/eval_retrieval.py "photobench_album${ALBUM}" --stage recall --index visual
