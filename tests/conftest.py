"""tests/conftest.py

Shared pytest configuration.

Adds the project root to sys.path so that ``from app.X import ...`` and
``from pipeline.X import ...`` work without a package install.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root is one level above this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
