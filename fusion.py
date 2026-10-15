#!/usr/bin/env python3
"""Run the controlled v2 workflow from a checkout, without installing the project."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fusion_engine.controlled.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
