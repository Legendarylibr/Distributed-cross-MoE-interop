"""MoE host: local routing + marketplace composition + outcome reporting."""

from __future__ import annotations

import uuid
import zlib
from dataclasses import dataclass, field

import numpy as np

from cei.learner import ContextualBanditLearner
from cei.node import ExpertNode
from cei.router import CombinationRouter
from cei.types import (
    ActivationBatch,
    Budget,
    CombinationPlan,
    CombinationStep,
    ExpertRef,
    FallbackEvent,
    Outcome,
)


@dataclass
class MoEHost:
    model_id: str
    node: ExpertNode
    router: CombinationRouter | None
    learner: ContextualBanditLearner | None
    num_layers: int
    dim: int
    top_k: int = 2
    domain: str = "general"
    # Per-layer router matrices: (num_experts, dim)
    gate_weights: dict[int, np.ndarray] = field(default_factory=dict)
    layer_expert_ids: dict[int, list[int]] = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    lambda_lat: float = 0.002
    lambda_cap: float = 0.01
    epsilon: float = 0.05

    def local_topk(self, layer_id: int, h: np.ndarray) -> list[tuple[ExpertRef, float]]:
        ids = self.layer_expert_ids[layer_id]
        W = self.gate_weights[layer_id]
        # h: (dim,) — score experts
        logits = W @ h
        k = min(self.top_k, len(ids))
        idx = np.argpartition(-logits, kth=k - 1)[:k]
        idx = idx[np.argsort(-logits[idx])]
        scores = logits[idx]
        # softmax weights
        e = np.exp(scores - scores.max())
        w = e / e.sum()
        return [
            (ExpertRef(self.model_id, layer_id, int(ids[j])), float(w[t]))
            for t, j in enumerate(idx)
        ]

    def all_layer_topk_for_propose(self, h: np.ndarray) -> dict[int, list[tuple[ExpertRef, float]]]:
        return {ell: self.local_topk(ell, h) for ell in range(self.num_layers)}

    def execute_layer(
        self,
        layer_id: int,
        h: np.ndarray,
        plan: CombinationPlan | None,
        budget: Budget,
        nodes: dict,
        require_leases: bool,
    ) -> tuple[np.ndarray, float, list[FallbackEvent]]:
        """Returns (y, latency_ms, fallbacks). `nodes` may be ExpertNode or NodeClient."""
        fallbacks: list[FallbackEvent] = []
        latency = 0.0

        step = None
        if plan is not None:
            for s in plan.steps:
                if s.layer_id == layer_id:
                    step = s
                    break

        if step is None:
            y, lat = self._local_moe(layer_id, h)
            return y, lat, fallbacks

        try:
            y, lat = self._execute_step(step, h, nodes, require_leases, budget, lease_ttl_ms=5000)
            latency += lat
            return y, latency, fallbacks
        except Exception as exc:  # noqa: BLE001 — map to fallback reasons
            reason = str(exc) if str(exc) else type(exc).__name__
            fallbacks.append(FallbackEvent(layer_id=layer_id, reason=reason))
            if budget.strict_local_fallback:
                # caller may abort remaining remote; still return local for this layer
                pass
            y, lat = self._local_moe(layer_id, h)
            return y, lat + latency, fallbacks

    def _local_moe(self, layer_id: int, h: np.ndarray) -> tuple[np.ndarray, float]:
        pairs = self.local_topk(layer_id, h)
        y = np.zeros_like(h)
        for ref, w in pairs:
            y = y + w * self.node.local_forward(ref, h)
        return y, self.node.base_latency_ms

    def _execute_step(
        self,
        step: CombinationStep,
        h: np.ndarray,
        nodes: dict,
        require_leases: bool,
        budget: Budget,
        lease_ttl_ms: int = 5000,
    ) -> tuple[np.ndarray, float]:
        y = np.zeros_like(h)
        total_lat = 0.0
        for ref, weight, lease_id, adapter_id in _zip_step(step):
            if ref.model_id == self.model_id:
                y = y + weight * self.node.local_forward(ref, h)
                total_lat += self.node.base_latency_ms
            else:
                remote = nodes.get(ref.model_id)
                if remote is None:
                    raise RuntimeError("UNAVAILABLE")
                # Lease if needed
                lid = lease_id
                if require_leases and not lid:
                    lease = remote.lease_capacity(ref, tokens_or_qps=1.0, ttl_ms=lease_ttl_ms)
                    lid = lease.lease_id
                act, lat = remote.forward_expert(
                    expert_ref=ref,
                    activation=ActivationBatch(tensor=h.copy()),
                    lease_id=lid,
                    adapter_id=adapter_id or None,
                    request_id=str(uuid.uuid4()),
                    require_lease=require_leases,
                )
                # Latency budget check
                if lat > budget.max_remote_latency_ms and not budget.allow_soft_latency:
                    if lid:
                        remote.release_capacity(lid)
                    raise TimeoutError("DEADLINE_EXCEEDED")
                y = y + weight * act.tensor
                total_lat += lat
                if lid:
                    remote.release_capacity(lid)
        return y, total_lat

    def _make_local_only_plan(
        self, local_topk: dict[int, list[tuple[ExpertRef, float]]], budget: Budget
    ) -> CombinationPlan:
        from cei.types import CombinationOp

        steps = []
        for ell in range(self.num_layers):
            pairs = local_topk.get(ell, [])
            if not pairs:
                continue
            refs = [r for r, _ in pairs]
            weights = [w for _, w in pairs]
            s = sum(weights) + 1e-12
            weights = [w / s for w in weights]
            steps.append(
                CombinationStep(
                    layer_id=ell,
                    expert_refs=refs,
                    weights=weights,
                    op=CombinationOp.REPLACE,
                )
            )
        return CombinationPlan(
            plan_id=str(uuid.uuid4()),
            host_model_id=self.model_id,
            steps=steps,
            budget=budget,
            local_only_equivalent=True,
        )

    def _sample_plan(self, plans: list[CombinationPlan], mode: str) -> CombinationPlan:
        if not plans:
            raise ValueError("no plans")
        if mode == "random":
            remotes = [p for p in plans if not p.local_only_equivalent]
            return remotes[int(self.rng.integers(0, len(remotes)))] if remotes else plans[0]
        if mode == "heuristic":
            return max(plans, key=lambda p: p.score)
        # learned: epsilon-greedy
        if self.rng.random() < self.epsilon:
            return plans[int(self.rng.integers(0, len(plans)))]
        return max(plans, key=lambda p: p.score)

    def forward(
        self,
        x: np.ndarray,
        domain_vec: np.ndarray,
        nodes: dict,
        budget: Budget | None = None,
        use_marketplace: bool = True,
        force_local: bool = False,
        mode: str = "learned",  # learned | random | heuristic | local
    ) -> tuple[np.ndarray, Outcome]:
        if self.router is None or self.learner is None:
            raise RuntimeError("in-process forward requires router and learner")
        budget = budget or Budget()

        h = x.astype(np.float64).copy()
        phi = _context_embedding(h, domain_vec, self.model_id)

        plan: CombinationPlan | None = None
        if force_local or mode == "local" or not use_marketplace:
            local_topk = self.all_layer_topk_for_propose(h)
            plan = self._make_local_only_plan(local_topk, budget)
        else:
            local_topk = self.all_layer_topk_for_propose(h)
            plans = self.router.propose(
                host_model_id=self.model_id,
                context_embedding=phi,
                local_topk=local_topk,
                host_dim=self.dim,
                budget=budget,
            )
            plan = self._sample_plan(plans, mode)

        assert plan is not None
        return self._run_plan(h, domain_vec, phi, plan, nodes, budget, mode, use_marketplace, force_local)

    def forward_distributed(
        self,
        x: np.ndarray,
        domain_vec: np.ndarray,
        peers: dict,
        router_client: object,
        learner_client: object,
        budget: Budget | None = None,
        use_marketplace: bool = True,
        force_local: bool = False,
        mode: str = "learned",
    ) -> tuple[np.ndarray, Outcome]:
        """Marketplace via gRPC Router/Learner; remotes via peer NodeClients."""
        budget = budget or Budget(allow_soft_latency=True, require_leases=True)
        h = x.astype(np.float64).copy()
        phi = _context_embedding(h, domain_vec, self.model_id)
        local_topk = self.all_layer_topk_for_propose(h)

        if force_local or mode == "local" or not use_marketplace:
            plan = self._make_local_only_plan(local_topk, budget)
        else:
            plans = router_client.propose(  # type: ignore[attr-defined]
                host_model_id=self.model_id,
                context_embedding=phi,
                local_topk=local_topk,
                host_dim=self.dim,
                budget=budget,
            )
            plan = self._sample_plan(plans, mode)

        hidden, outcome = self._run_plan(
            h, domain_vec, phi, plan, peers, budget, mode, use_marketplace, force_local, pred_cap=0.0
        )
        if mode == "learned" and use_marketplace and not force_local:
            learner_client.report(outcome)  # type: ignore[attr-defined]
        return hidden, outcome

    def _run_plan(
        self,
        h: np.ndarray,
        domain_vec: np.ndarray,
        phi: np.ndarray,
        plan: CombinationPlan,
        nodes: dict,
        budget: Budget,
        mode: str,
        use_marketplace: bool,
        force_local: bool,
        pred_cap: float | None = None,
    ) -> tuple[np.ndarray, Outcome]:
        total_lat = 0.0
        all_fallbacks: list[FallbackEvent] = []
        abort_remote = False

        for ell in range(self.num_layers):
            active_plan = None if abort_remote else plan
            y, lat, fb = self.execute_layer(
                layer_id=ell,
                h=h,
                plan=active_plan,
                budget=budget,
                nodes=nodes,
                require_leases=budget.require_leases,
            )
            total_lat += lat
            all_fallbacks.extend(fb)
            if fb and budget.strict_local_fallback:
                abort_remote = True
            h = h + y
            h = np.tanh(h)

        utility = float(np.dot(h / (np.linalg.norm(h) + 1e-12), domain_vec))
        if pred_cap is None and self.router is not None:
            cap_pen = self.router._pred_cap(plan, self.model_id)  # noqa: SLF001
        else:
            cap_pen = 0.0 if pred_cap is None else pred_cap
        reward = utility - self.lambda_lat * total_lat - self.lambda_cap * cap_pen

        outcome = Outcome(
            plan_id=plan.plan_id,
            host_model_id=self.model_id,
            reward=reward,
            utility=utility,
            latency_ms=total_lat,
            capacity_penalty=cap_pen,
            tokens=1,
            fallbacks=all_fallbacks,
            partial=bool(all_fallbacks),
            context_embedding=phi,
            plan=plan,
        )
        if (
            mode == "learned"
            and use_marketplace
            and not force_local
            and self.learner is not None
            and pred_cap is None
        ):
            self.learner.report(outcome)
        return h, outcome


def _zip_step(step: CombinationStep):
    refs = step.expert_refs
    weights = step.weights or [1.0 / len(refs)] * len(refs)
    leases = step.lease_ids + [""] * max(0, len(refs) - len(step.lease_ids))
    adapters = step.adapter_ids + [""] * max(0, len(refs) - len(step.adapter_ids))
    for i, ref in enumerate(refs):
        yield ref, weights[i], leases[i] if i < len(leases) else "", adapters[i] if i < len(adapters) else ""


def _context_embedding(h: np.ndarray, domain_vec: np.ndarray, model_id: str) -> np.ndarray:
    # Fixed-size context: concat hidden summary + domain + model hash.
    # crc32 rather than hash(): str hashes are salted per process (PYTHONHASHSEED),
    # which would make context features — and thus learner behavior — non-reproducible.
    mid = (zlib.crc32(model_id.encode()) % 997) / 997.0
    return np.concatenate([h[:16] if h.size >= 16 else np.pad(h, (0, 16 - h.size)), domain_vec[:16], [mid]])
