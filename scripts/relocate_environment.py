"""Repair virtualenv launchers after moving this folder, without reinstalling packages."""
from pathlib import Path
import re
import shlex

root = Path(__file__).resolve().parents[1]
venv = root / ".venv"
activation = venv / "bin/activate"
old = None
if activation.exists():
    match = re.search(r"^\s*export VIRTUAL_ENV=(.+)$", activation.read_text(), re.M)
    if match:
        values = shlex.split(match.group(1))
        if values:
            old = values[0]

updated = 0
for path in (venv / "bin").iterdir():
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 262144:
        continue
    try:
        text = path.read_text()
    except (UnicodeError, OSError):
        continue
    original = text
    first, separator, rest = text.partition("\n")
    shell_wrapper = first == "#!/bin/sh" and rest.startswith("'''exec' ") and "/.venv/bin/python" in rest.split("\n", 1)[0]
    if (first.startswith("#!") and "/.venv/bin/python" in first) or shell_wrapper:
        if shell_wrapper:
            rest = "\n".join(rest.split("\n")[2:])
        # A shell/Python trampoline supports spaces and Chinese directory names.
        text = '#!/bin/sh\n\'\'\'exec\' "$(dirname "$0")/python" "$0" "$@"\n\' \'\'\'\n' + rest
    elif old and old != str(venv) and old in text:
        if path.name.startswith("activate"):
            # The old path is replaced as a shell token, not an unquoted string.
            text = text.replace(shlex.quote(old), shlex.quote(str(venv)))
            if old in text:
                text = text.replace(old, str(venv))
    if text != original:
        path.write_text(text)
        updated += 1

cfg = venv / "pyvenv.cfg"
if cfg.exists() and old and old != str(venv):
    text = cfg.read_text().replace(old, str(venv))
    cfg.write_text(text)
if updated:
    print(f"已适配新位置的运行环境（{updated} 个启动文件）")
