"""Explicit, revision-pinned model download into the Desktop project."""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="准备新版整图检索与多条件精排模型")
parser.add_argument("--download", action="store_true")
args = parser.parse_args()
if not args.download:
    parser.error("需加 --download 才会联网下载")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
models = [("tencent/WeMM-Embedding-2B", "bbd6cd4bf52cfc6716f752a2df80b2706720bd95", "wemm-2b"),
          ("Alibaba-NLP/core-reranker-2b", "64e1a36cd6ecace459a4c7cd8f7d0ef5d6fee30b", "core-reranker-2b")]
for repo, revision, name in models:
    directory = root / "data/models" / name
    snapshot_download(repo_id=repo, revision=revision, local_dir=directory, max_workers=2,
        allow_patterns=["*.json", "*.jinja", "*.py", "*.safetensors", "*.txt", "LICENSE", "README.md"])
    (directory / "photorecall-source.json").write_text(json.dumps({"repo": repo, "revision": revision}, indent=2))
    print(repo + " 已准备")
