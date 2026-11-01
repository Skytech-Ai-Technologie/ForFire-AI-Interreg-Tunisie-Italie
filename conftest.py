# SPDX-License-Identifier: Apache-2.0
"""Makes the src-layout package importable for pytest without installation.

For development, prefer an editable install (``pip install -e .``); this
fallback exists so the suite runs on a fresh clone.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
