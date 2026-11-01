import argparse
import json
import os
import threading
import urllib.request
import webbrowser
from pathlib import Path

parser = argparse.ArgumentParser(description="PhotoRecall 本地照片搜索")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--open", action="store_true", help="启动后打开浏览器")
args = parser.parse_args()
url = f"http://127.0.0.1:{args.port}"

try:
    with urllib.request.urlopen(url + "/api/state", timeout=1) as response:
        existing = json.load(response)
    if existing.get("version") in {"0.1.0", "0.2.0", "0.3.0", "0.4.0", "0.5.0", "0.6.0"} and "libraries" in existing:
        print("PhotoRecall 已经在运行：" + url)
        if args.open:
            webbrowser.open(url)
        raise SystemExit(0)
except (OSError, ValueError):
    pass

if args.open:
    def open_when_ready():
        import time
        for _ in range(60):
            try:
                with urllib.request.urlopen(url + "/api/state", timeout=1):
                    webbrowser.open(url)
                return
            except OSError:
                time.sleep(.25)
    threading.Thread(target=open_when_ready, daemon=True).start()
print("PhotoRecall 本地网页：" + url, flush=True)
print("关闭此终端窗口或按 Control + C 可停止服务。", flush=True)
import uvicorn
runtime = Path(__file__).resolve().parent / "data" / "server.json"
runtime.parent.mkdir(exist_ok=True)
runtime.write_text(json.dumps({"pid": os.getpid(), "url": url, "project_root": str(Path(__file__).resolve().parent)}))
try:
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, log_level="warning")
finally:
    try:
        if json.loads(runtime.read_text()).get("pid") == os.getpid():
            runtime.unlink()
    except (OSError, ValueError):
        pass
