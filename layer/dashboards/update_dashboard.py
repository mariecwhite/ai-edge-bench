#!/usr/bin/env python3
"""Convenience entrypoint forwarding to `layer/scripts/update_dashboard.py`."""

from pathlib import Path
import runpy

if __name__ == "__main__":
  runpy.run_path(
      str(Path(__file__).resolve().parents[1] / "scripts" / "update_dashboard.py"),
      run_name="__main__",
  )
