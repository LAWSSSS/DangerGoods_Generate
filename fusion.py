#!/usr/bin/env python3
"""Run the controlled v2 workflow from a checkout, without installing the project."""

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def _runtime_ready() -> bool:
    if sys.version_info < (3, 11):
        return False
    for name in ("numpy", "cv2", "jsonschema", "yaml", "httpx", "PIL", "requests"):
        try:
            __import__(name)
        except ImportError:
            return False
    return True


if not _runtime_ready():
    if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])
    sys.stderr.write(
        "当前解释器无法运行受控融合入口，需要 Python 3.11+ 以及 requirements.txt 里的依赖。\n"
        f"当前: {sys.executable} ({sys.version.split()[0]})\n"
        "在仓库根目录执行: uv pip install --python .venv/bin/python -r requirements.txt\n"
    )
    raise SystemExit(1)

sys.path.insert(0, str(ROOT / "src"))

from fusion_engine.controlled.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
