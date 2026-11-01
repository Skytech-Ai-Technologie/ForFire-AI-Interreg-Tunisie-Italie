# SPDX-License-Identifier: Apache-2.0
"""Latency statistics (stdlib only, linear-interpolated percentiles)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank // 1)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def summarize_latencies(values_ms: Iterable[float], percentiles: Sequence[float] = (50, 95, 99)) -> Dict[str, float]:
    collected: List[float] = [float(v) for v in values_ms]
    if not collected:
        return {"count": 0}
    summary: Dict[str, float] = {
        "count": len(collected),
        "mean_ms": sum(collected) / len(collected),
        "min_ms": min(collected),
        "max_ms": max(collected),
    }
    for p in percentiles:
        summary["p{}_ms".format(int(p))] = percentile(collected, p)
    return summary
