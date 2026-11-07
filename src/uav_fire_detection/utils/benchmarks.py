# SPDX-License-Identifier: Apache-2.0
"""Reference benchmark tables (CSV) loader.

The benchmark tables in ``benchmarks/*.csv`` hold the reference figures for
the pipeline (detection metrics, latency profile, quantization comparison,
operational analyses). Scripts load them at runtime so reports always carry
the same reference values as the tables.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional


def load_csv(path) -> List[Dict[str, str]]:
    """Read a benchmark CSV table into a list of row dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("benchmark table not found: {}".format(p))
    with open(p, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def find_row(rows: List[Dict[str, str]], **filters) -> Optional[Dict[str, str]]:
    """Return the first row whose fields match all ``filters``
    (string-compared, case-insensitive), or None."""
    for row in rows:
        if all(
            str(row.get(key, "")).strip().lower() == str(value).strip().lower()
            for key, value in filters.items()
        ):
            return row
    return None
