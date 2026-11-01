"""Explicit opt-in download of the small official object detector."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true")
args = parser.parse_args()
if not args.download:
    parser.error("模型下载需要显式 --download")
root = Path(__file__).resolve().parents[1]
path = root / "data/models/yolo26n.pt"
url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
sha = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".download")
try:
    with urllib.request.urlopen(url, timeout=120) as response:
        temporary.write_bytes(response.read())
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual != sha:
        raise RuntimeError("目标检测模型完整性校验失败")
    temporary.replace(path)
    (root / "data/yolo-download.json").write_text(json.dumps({"url": url, "size": path.stat().st_size, "sha256": actual}))
    print("本地目标检测模型已准备，大小约 5.5MB")
finally:
    temporary.unlink(missing_ok=True)
