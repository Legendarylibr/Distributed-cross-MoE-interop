"""Combination router: candidate generation + scoring."""

from __future__ import annotations

import itertools
import os
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from cei.learner import ContextualBanditLearner, score_plan_from_snapshot
from cei.registry import ExpertRegistry
from cei.types import (
    Budget,
    CombinationOp,
    CombinationPlan,
    CombinationStep,
    ExpertDescriptor,
    ExpertRef,
)


@dataclass
class CombinationRouter:
    registry: ExpertRegistry
    learner: ContextualBanditLearner
    m: int = 2
    n: int = 8
    k_nn: int = 32
    top_nn_per_layer: int = 4
    max_candidates: int = 64
    lambda_lat: float = 0.01
    lambda_cap: float = 0.01
    epsilon: float = 0.05
    gumbel_tau: float = 1.0
    sample_mode: str = "epsilon"  # or "gumbel"
    version: str = "0.1.0"
    # exact_layer | any_with_adapter | free
    layer_compat: str = field(
        default_factory=lambda: os.environ.get("CEI_LAYER_COMPAT", "exact_layer")
    )
    policy_cache_ttl_s: float = 2.0
    _policy_snapshot: dict | None = field(default=None, repr=False)
    _policy_fetched_at: float = 0.0

    def _refresh_policy_cache(self) -> dict | None:
        now = time.time()
        if (
            self._policy_snapshot is not None
            and now - self._policy_fetched_at < self.policy_cache_ttl_s
        ):
            return self._policy_snapshot
        snap = None
        if hasattr(self.learner, "policy_snapshot"):
            snap = self.learner.policy_snapshot()
        elif hasattr(self.learner, "get_policy_snapshot"):
            snap = self.learner.get_policy_snapshot()
        if snap is not None:
            self._policy_snapshot = snap
            self._policy_fetched_at = now
        return self._policy_snapshot

    def _layer_ok(self, host_layer: int, desc: ExpertDescriptor) -> bool:
        mode = self.layer_compat
        if mode == "free":
            return True
        if mode == "exact_layer":
            return desc.expert_ref.layer_id == host_layer
        if mode == "any_with_adapter":
            return desc.expert_ref.layer_id == host_layer or bool(desc.adapter_id)
        return desc.expert_ref.layer_id == host_layer

    def _score_plan(
        self,
        context_embedding: np.ndarray,
        plan: CombinationPlan,
        fp_sims: dict[str, float],
    ) -> float:
        snap = self._refresh_policy_cache()
        if snap is not None:
            cached = score_plan_from_snapshot(context_embedding, plan, snap, fp_sims)
            if cached is not None:
                return cached
        return self.learner.cold_start_utility(
            context_embedding, plan, fp_sims, snapshot=None
        )

    def propose(
        self,
        host_model_id: str,
        context_embedding: np.ndarray,
        local_topk: dict[int, list[tuple[ExpertRef, float]]],
        host_dim: int,
        budget: Budget | None = None,
        layer_hints: list[int] | None = None,
        n: int | None = None,
        include_local_only: bool = True,
        principal: str | None = None,
    ) -> list[CombinationPlan]:
        budget = budget or Budget()
        n = n if n is not None else self.n
        layers = self._select_layers(layer_hints, list(local_topk.keys()))

        local_only = self._local_only_plan(host_model_id, local_topk, layers, budget)
        candidates: list[CombinationPlan] = []
        if include_local_only:
            candidates.append(local_only)

        # Build NN pools per layer from local expert fingerprints
        pools: dict[int, list[tuple[ExpertRef, float]]] = {}
        for ell in layers:
            local_refs = [r for r, _ in local_topk.get(ell, [])]
            if not local_refs:
                continue
            # Use first local expert fingerprint as query
            desc = self.registry.get(local_refs[0])
            if desc is None:
                fp = context_embedding
            else:
                fp = desc.fingerprint
            nn = self.registry.describe_nn(
                fingerprint=fp,
                k=self.k_nn,
                host_dim_in=host_dim,
                host_dim_out=host_dim,
                principal=principal,
                exclude_model=None,  # include remotes; filter host locals below
            )
            remote = [
                (d.expert_ref, sim)
                for d, sim, routable in nn
                if routable
                and d.expert_ref.model_id != host_model_id
                and self._layer_ok(ell, d)
            ]
            pools[ell] = remote[: self.top_nn_per_layer]

        # Enumerate subsets of layers up to m (cap raw candidates before scoring)
        for r in range(1, min(self.m, len(layers)) + 1):
            if len(candidates) >= self.max_candidates:
                break
            for subset in itertools.combinations(layers, r):
                if len(candidates) >= self.max_candidates:
                    break
                for choices in itertools.product(
                    *[
                        pools.get(ell, [])[: self.top_nn_per_layer] or [(None, 0.0)]
                        for ell in subset
                    ]
                ):
                    if len(candidates) >= self.max_candidates:
                        break
                    if any(c[0] is None for c in choices):
                        continue
                    for op in (CombinationOp.REPLACE, CombinationOp.AUGMENT):
                        if len(candidates) >= self.max_candidates:
                            break
                        plan = self._build_plan(
                            host_model_id=host_model_id,
                            local_topk=local_topk,
                            subset=list(subset),
                            remotes=[c[0] for c in choices],
                            op=op,
                            budget=budget,
                        )
                        if self._feasible(plan, host_model_id, budget):
                            candidates.append(plan)

        # Score
        fp_sims: dict[str, float] = {}
        for ell, pool in pools.items():
            for ref, sim in pool:
                fp_sims[ref.key()] = sim

        scored: list[tuple[float, CombinationPlan]] = []
        for plan in candidates:
            û = self._score_plan(context_embedding, plan, fp_sims)
            cost = self.lambda_lat * self._pred_lat(plan, host_model_id) + self.lambda_cap * self._pred_cap(
                plan, host_model_id
            )
            plan.score = û - cost
            scored.append((plan.score, plan))

        # Unique by arm_key keeping best score
        best: dict[str, CombinationPlan] = {}
        for score, plan in scored:
            k = plan.arm_key()
            if k not in best or score > best[k].score:
                plan.score = score
                best[k] = plan

        ranked = sorted(best.values(), key=lambda p: p.score, reverse=True)
        # Spec: MUST include a local-only plan when requested
        if include_local_only:
            local_plans = [p for p in ranked if p.local_only_equivalent]
            others = [p for p in ranked if not p.local_only_equivalent]
            if not local_plans:
                local_plans = [local_only]
            return local_plans[:1] + others[: max(0, n - 1)]
        return ranked[:n]

    def sample(self, plans: list[CombinationPlan], rng: np.random.Generator) -> CombinationPlan:
        if not plans:
            raise ValueError("no plans")
        if len(plans) == 1:
            return plans[0]
        if self.sample_mode == "gumbel":
            scores = np.array([p.score for p in plans], dtype=np.float64)
            gumbel = rng.gumbel(size=len(plans))
            idx = int(np.argmax(scores + gumbel / self.gumbel_tau))
            return plans[idx]
        # epsilon-greedy
        if rng.random() < self.epsilon:
            return plans[int(rng.integers(0, len(plans)))]
        return max(plans, key=lambda p: p.score)

    def _select_layers(self, hints: list[int] | None, available: list[int]) -> list[int]:
        available = sorted(available)
        if hints:
            return [h for h in hints if h in available] or available
        if not available:
            return []
        # Prefer mid-depth layers
        mid = available[len(available) // 2]
        # Return up to 3 mid-ish layers for search
        idxs = sorted(set([max(0, available.index(mid) - 1), available.index(mid), min(len(available) - 1, available.index(mid) + 1)]))
        return [available[i] for i in idxs]

    def _local_only_plan(
        self,
        host_model_id: str,
        local_topk: dict[int, list[tuple[ExpertRef, float]]],
        layers: list[int],
        budget: Budget,
    ) -> CombinationPlan:
        steps = []
        for ell in layers:
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
            host_model_id=host_model_id,
            steps=steps,
            budget=budget,
            score=0.0,
            local_only_equivalent=True,
        )

    def _build_plan(
        self,
        host_model_id: str,
        local_topk: dict[int, list[tuple[ExpertRef, float]]],
        subset: list[int],
        remotes: list[ExpertRef],
        op: CombinationOp,
        budget: Budget,
    ) -> CombinationPlan:
        steps: list[CombinationStep] = []
        for ell, remote in zip(subset, remotes):
            local_pairs = local_topk.get(ell, [])
            local_refs = [r for r, _ in local_pairs]
            local_w = [w for _, w in local_pairs]
            if op == CombinationOp.REPLACE:
                # swap lowest-weight local with remote
                if local_refs:
                    idx_min = int(np.argmin(local_w)) if local_w else 0
                    refs = list(local_refs)
                    weights = list(local_w) if local_w else [1.0] * len(refs)
                    refs[idx_min] = remote
                    weights[idx_min] = max(weights) if weights else 1.0
                else:
                    refs = [remote]
                    weights = [1.0]
            else:  # AUGMENT / insert
                refs = list(local_refs) + [remote]
                weights = list(local_w) + [max(local_w) if local_w else 1.0]
            s = sum(weights) + 1e-12
            weights = [w / s for w in weights]
            adapter_ids: list[str] = []
            for ref in refs:
                if ref.model_id == host_model_id:
                    adapter_ids.append("")
                else:
                    d = self.registry.get(ref)
                    adapter_ids.append((d.adapter_id or "") if d else "")
            steps.append(
                CombinationStep(
                    layer_id=ell,
                    expert_refs=refs,
                    weights=weights,
                    op=op,
                    adapter_ids=adapter_ids,
                )
            )
        return CombinationPlan(
            plan_id=str(uuid.uuid4()),
            host_model_id=host_model_id,
            steps=steps,
            budget=budget,
            local_only_equivalent=False,
        )

    def _feasible(self, plan: CombinationPlan, host_model_id: str, budget: Budget) -> bool:
        remotes = plan.remote_refs(host_model_id)
        if len(remotes) > budget.max_remote_experts:
            return False
        pred = self._pred_lat(plan, host_model_id)
        if not budget.allow_soft_latency and pred > budget.max_remote_latency_ms:
            return False
        return True

    def _pred_lat(self, plan: CombinationPlan, host_model_id: str) -> float:
        total = 0.0
        for ref in plan.remote_refs(host_model_id):
            d = self.registry.get(ref)
            total += (d.p50_latency_ms if d else 10.0) + 5.0  # +RTT estimate
        return total

    def _pred_cap(self, plan: CombinationPlan, host_model_id: str) -> float:
        pen = 0.0
        tau = 0.8
        for ref in plan.remote_refs(host_model_id):
            load = self.registry.load(ref)
            cap = self.registry.capacity(ref) or 1.0
            pen += max(0.0, load / cap - tau)
        return pen
