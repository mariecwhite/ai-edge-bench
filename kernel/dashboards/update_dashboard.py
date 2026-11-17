#!/usr/bin/env python3
"""Update the Level 1 Kernel Benchmark Suite dashboard (kernel/dashboards/index.html).

Delegates to kernel/scripts/update_dashboard.py so the updater can be invoked
from either `kernel/scripts/update_dashboard.py` or `kernel/dashboards/update_dashboard.py`.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts.update_dashboard import main  # noqa: E402

if __name__ == "__main__":
  sys.exit(main())
