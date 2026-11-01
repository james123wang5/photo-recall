import json
import os
import signal
import subprocess
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runtime = root / "data" / "server.json"
if not runtime.exists():
    print("没有记录到正在运行的 PhotoRecall。若在终端启动，可按 Control + C 停止。")
    raise SystemExit(0)
try:
    state = json.loads(runtime.read_text())
    pid = int(state["pid"])
    command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], text=True, capture_output=True).stdout.strip()
    cwd = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], text=True, capture_output=True).stdout
    relative_entry = bool(re.search(r"(?:^|\s)run\.py(?:\s|$)", command)) and ("n" + str(root) + "\n") in cwd
    recorded_root = Path(state.get("project_root", root))
    recorded_entry = str(recorded_root / "run.py") in command
    if str(root / "run.py") not in command and not recorded_entry and not relative_entry:
        print("原服务已退出；没有停止其他程序。")
        runtime.unlink(missing_ok=True)
        raise SystemExit(0)
    os.kill(pid, signal.SIGTERM)
    print("PhotoRecall 已收到停止指令。")
except (OSError, ValueError, KeyError) as exc:
    print("未能停止服务：" + str(exc))
