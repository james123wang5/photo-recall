"""Explicit model preparation. Never imported by the server."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="下载 SigLIP 2 到本项目；需先安装 requirements-ai.txt")
parser.add_argument("--download", action="store_true", help="明确允许下载模型文件")
args = parser.parse_args()
if not args.download:
    parser.error("需要 --download 才会联网下载模型。")
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parents[1]
snapshot_download(repo_id="google/siglip2-base-patch16-224", revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                  local_dir=root / "data" / "models" / "siglip2",
                  ignore_patterns=["*.bin", "*.h5", "*.msgpack", "onnx/*", "*.onnx"])
print("模型已准备。回到网页的『模型与设置』，启用画面搜索，再更新照片索引。")
