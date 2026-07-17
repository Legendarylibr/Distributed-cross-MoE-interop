"""Combination learner: contextual bandit over plan arms."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from cei.types import CombinationPlan, Outcome


@dataclass
class ContextualBanditLearner:
    """Linear contextual bandit: score = <θ_a, φ> with per-arm ridge regression."""

    ctx_dim: int
    lambda_reg: float = 1.0
    batch_size: int = 64
    lambda_bal: float = 0.01
    lambda_stick: float = 0.01
    version: int = 0
    _A: dict[str, np.ndarray] = field(default_factory=dict)
    _b: dict[str, np.ndarray] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _pending: list[Outcome] = field(default_factory=list)
    _prev_arm: str | None = None
    _expert_uses: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def _ensure_arm(self, arm: str) -> None:
        if arm not in self._A:
            self._A[arm] = self.lambda_reg * np.eye(self.ctx_dim)
            self._b[arm] = np.zeros(self.ctx_dim)

    def estimate_utility(self, phi: np.ndarray, plan: CombinationPlan) -> float:
        arm = plan.arm_key()
        self._ensure_arm(arm)
        A_inv = np.linalg.pinv(self._A[arm])
        theta = A_inv @ self._b[arm]
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape[0] != self.ctx_dim:
            phi = _pad_or_trim(phi, self.ctx_dim)
        mean = float(theta @ phi)
        # UCB-style bonus for exploration in scoring
        bonus = 0.1 * float(np.sqrt(phi @ A_inv @ phi))
        stick = 0.0
        if self._prev_arm is not None and arm != self._prev_arm:
            stick = self.lambda_stick
        bal = self.lambda_bal * self._balance_penalty(plan)
        return mean + bonus - stick - bal

    def _balance_penalty(self, plan: CombinationPlan) -> float:
        if not self._expert_uses:
            return 0.0
        uses = []
        for step in plan.steps:
            for ref in step.expert_refs:
                uses.append(self._expert_uses.get(ref.key(), 0))
        if not uses:
            return 0.0
        total = sum(self._expert_uses.values()) + 1e-6
        # Penalize heavily used experts
        return float(np.mean(uses) / total) * 10.0

    def report(self, outcome: Outcome) -> None:
        self._pending.append(outcome)
        if outcome.plan is not None:
            for step in outcome.plan.steps:
                for ref in step.expert_refs:
                    self._expert_uses[ref.key()] += 1
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        for outcome in self._pending:
            plan = outcome.plan
            if plan is None or outcome.context_embedding is None:
                continue
            arm = plan.arm_key()
            self._ensure_arm(arm)
            phi = _pad_or_trim(np.asarray(outcome.context_embedding, dtype=np.float64), self.ctx_dim)
            r = outcome.reward
            self._A[arm] += np.outer(phi, phi)
            self._b[arm] += r * phi
            self._counts[arm] += 1
            self._prev_arm = arm
        self._pending.clear()
        self.version += 1

    def policy_snapshot(self) -> dict:
        """Export arm thetas for router-side caching (hot-path scoring without RPC)."""
        arms = []
        for arm, A in self._A.items():
            A_inv = np.linalg.pinv(A)
            theta = A_inv @ self._b[arm]
            arms.append(
                {
                    "arm_key": arm,
                    "theta": theta.astype(np.float64),
                    "count": int(self._counts.get(arm, 0)),
                }
            )
        return {"version": self.version, "ctx_dim": self.ctx_dim, "arms": arms}

    def score_from_snapshot(
        self,
        phi: np.ndarray,
        plan: CombinationPlan,
        snapshot: dict,
        fingerprint_sims: dict[str, float] | None = None,
        alpha: float = 0.5,
    ) -> float | None:
        return score_plan_from_snapshot(phi, plan, snapshot, fingerprint_sims, alpha)

    def cold_start_utility(
        self,
        phi: np.ndarray,
        plan: CombinationPlan,
        fingerprint_sims: dict[str, float],
        alpha: float = 0.5,
        snapshot: dict | None = None,
    ) -> float:
        """Blend learned estimate with fingerprint similarity prior."""
        if snapshot is not None:
            cached = score_plan_from_snapshot(phi, plan, snapshot, fingerprint_sims, alpha)
            if cached is not None:
                return cached
        learned = self.estimate_utility(phi, plan)
        if plan.local_only_equivalent:
            return learned
        sims = []
        for step in plan.steps:
            for ref in step.expert_refs:
                sims.append(fingerprint_sims.get(ref.key(), 0.0))
        prior = alpha * (float(np.mean(sims)) if sims else 0.0)
        arm = plan.arm_key()
        n = self._counts.get(arm, 0)
        w = n / (n + 5.0)
        return w * learned + (1.0 - w) * (prior + 0.1)


def score_plan_from_snapshot(
    phi: np.ndarray,
    plan: CombinationPlan,
    snapshot: dict,
    fingerprint_sims: dict[str, float] | None = None,
    alpha: float = 0.5,
) -> float | None:
    """Score using a policy snapshot. Returns None if arm unknown."""
    arm = plan.arm_key()
    by_key = {a["arm_key"]: a for a in snapshot.get("arms", [])}
    if arm not in by_key:
        return None
    entry = by_key[arm]
    theta = np.asarray(entry["theta"], dtype=np.float64)
    phi = _pad_or_trim(np.asarray(phi, dtype=np.float64), int(snapshot["ctx_dim"]))
    learned = float(theta @ phi)
    if plan.local_only_equivalent:
        return learned
    sims = [
        (fingerprint_sims or {}).get(ref.key(), 0.0)
        for step in plan.steps
        for ref in step.expert_refs
    ]
    prior = alpha * (float(np.mean(sims)) if sims else 0.0)
    n = int(entry.get("count", 0))
    w = n / (n + 5.0)
    return w * learned + (1.0 - w) * (prior + 0.1)


def _pad_or_trim(phi: np.ndarray, dim: int) -> np.ndarray:
    phi = phi.reshape(-1)
    if phi.size == dim:
        return phi
    out = np.zeros(dim, dtype=np.float64)
    n = min(dim, phi.size)
    out[:n] = phi[:n]
    return out
