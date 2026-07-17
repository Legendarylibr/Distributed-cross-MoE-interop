"""Evaluation metrics helpers (SPEC §7 / docs/evaluation.md)."""

from __future__ import annotations

import numpy as np

from cei.simulate import SimResult


def gini(values: list[float] | np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size == 0 or np.allclose(x, 0):
        return 0.0
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1) / n)


def regret(utility_a: float, utility_b: float) -> float:
    """Positive means A is worse than B."""
    return utility_b - utility_a


def format_ablation_table(results: dict[str, dict[str, float]]) -> str:
    headers = ["mode", "utility_mean", "reward_mean", "latency_p50", "fallback_rate", "remote_plan_rate"]
    lines = [" | ".join(headers), " | ".join("---" for _ in headers)]
    for mode in ("local", "random", "heuristic", "learned"):
        row = results.get(mode)
        if not row:
            continue
        lines.append(
            " | ".join(
                [
                    mode,
                    f"{row['utility_mean']:.4f}",
                    f"{row['reward_mean']:.4f}",
                    f"{row['latency_p50']:.2f}",
                    f"{row['fallback_rate']:.3f}",
                    f"{row['remote_plan_rate']:.3f}",
                ]
            )
        )
    return "\n".join(lines)


def windowed_means(result: SimResult, windows: int = 5) -> list[float]:
    u = result.utilities
    if not u:
        return []
    n = len(u)
    size = max(1, n // windows)
    means = []
    for i in range(0, n, size):
        chunk = u[i : i + size]
        if chunk:
            means.append(float(np.mean(chunk)))
    return means
