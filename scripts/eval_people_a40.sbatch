#!/bin/bash
#SBATCH --job-name=people
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/people-%x-%j.out
#SBATCH --error=logs/people-%x-%j.err
set -euo pipefail
# Usage: sbatch --job-name=people1 scripts/eval_people_a40.sbatch 1 浩哥 晨阳 ...
# Runs the normal people baseline, then the ORACLE naming upper bound (answer leakage).
cd "${SLURM_SUBMIT_DIR:?}"
ALBUM="$1"; shift
export PHOTORECALL_KEEP_MODELS=1
export PYTHONUNBUFFERED=1 PHOTORECALL_DISABLE_OLLAMA=1
.venv/bin/python scripts/eval_retrieval.py "photobench_album${ALBUM}" --stage recall \
  --index visual people --config '{"people_enabled":true}'
.venv/bin/python scripts/eval_retrieval.py "photobench_album${ALBUM}" --stage recall \
  --index visual people --oracle-people "$@"
