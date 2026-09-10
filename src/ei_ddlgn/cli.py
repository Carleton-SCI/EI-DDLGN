"""Installed command-line entry point for static PBS analysis."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "analyze_pbs.py"
    namespace = runpy.run_path(str(script))
    return int(namespace["main"]())
