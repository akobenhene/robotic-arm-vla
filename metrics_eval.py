"""Evaluation metrics helpers: Wilson CI, timing, steps-to-success."""

from __future__ import annotations

import math
from typing import Any, Sequence


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial success rate (as percentages)."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = max(0.0, (center - margin) * 100.0)
    hi = min(100.0, (center + margin) * 100.0)
    return (round(lo, 2), round(hi, 2))


def summarize_episodes(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode rows into portfolio-ready KPI block."""
    n = len(episodes)
    n_success = sum(1 for e in episodes if e.get("success"))
    rate = (100.0 * n_success / n) if n else 0.0
    lo, hi = wilson_interval(n_success, n)

    success_steps = [int(e["steps"]) for e in episodes if e.get("success")]
    all_steps = [int(e["steps"]) for e in episodes]
    latencies = [float(e["seconds"]) for e in episodes if "seconds" in e]

    def _mean(xs: list[float] | list[int]) -> float | None:
        return round(float(sum(xs) / len(xs)), 3) if xs else None

    return {
        "n_episodes": n,
        "n_success": n_success,
        "success_rate_pct": round(rate, 2),
        "wilson_ci95_pct": [lo, hi],
        "mean_steps_all": _mean(all_steps),
        "mean_steps_to_success": _mean(success_steps),
        "mean_episode_seconds": _mean(latencies),
        "mean_ms_per_step": (
            round(1000.0 * sum(latencies) / max(sum(all_steps), 1), 2) if latencies and all_steps else None
        ),
    }
