#!/bin/bash
#SBATCH --job-name=photorecall-eval
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=06:00:00
#SBATCH --output=logs/flickr-%j.out
#SBATCH --error=logs/flickr-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PHOTORECALL_KEEP_MODELS=1
export PYTHONUNBUFFERED=1
# Do not contact or unload an unrelated service on a shared compute node.
export PHOTORECALL_DISABLE_OLLAMA=1
.venv/bin/python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable; refusing to run GPU benchmark on CPU')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
# Verify both actual models with one isolated public image before a long index.
.venv/bin/python - <<'PYSMOKE'
import argparse, json, shutil
from pathlib import Path
from scripts import eval_retrieval as e
suite = e.suites("flickr1k")[0]
query = suite["queries"][0]
folder = e.RUNS / "a40_smoke_images"
folder.mkdir(parents=True, exist_ok=True)
name = query["targets"][0]
if not (folder / name).exists():
    shutil.copy2(Path(suite["images"]) / name, folder / name)
args = argparse.Namespace(stage="recall", index=["visual"], mode="find_one", depth=200,
                          limit=1, offset=0, timeout=120, config=None)
e.evaluate({"name": "a40_smoke", "images": folder, "queries": [query]}, args)
summary_file = sorted(e.RESULTS.glob("a40_smoke-recall-*/summary.json"))[-1]
summary = json.loads(summary_file.read_text())
channels = summary["channel_query_count"]
if summary["queries_with_warnings"] or not all(any(k.startswith(prefix) for k in channels) for prefix in ("primary：", "siglip：")):
    raise SystemExit("Smoke test reported warnings or missing visual channels; aborting baseline")
PYSMOKE
.venv/bin/python scripts/eval_retrieval.py flickr1k --stage recall --limit 200
