"""Fleet construction and multi-domain simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cei.host import MoEHost
from cei.learner import ContextualBanditLearner
from cei.node import ExpertNode, fingerprint_from_weights, make_expert_module
from cei.registry import ExpertRegistry
from cei.router import CombinationRouter
from cei.types import Budget, DType, ExpertDescriptor, ExpertRef, Outcome


DOMAINS = ("code", "math", "general")


@dataclass
class Fleet:
    registry: ExpertRegistry
    learner: ContextualBanditLearner
    router: CombinationRouter
    hosts: dict[str, MoEHost]
    nodes: dict[str, ExpertNode]
    domain_vecs: dict[str, np.ndarray]
    dim: int
    ctx_dim: int


@dataclass
class SimResult:
    mode: str
    utilities: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    fallback_rates: list[float] = field(default_factory=list)
    remote_plan_rates: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        def mean(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        return {
            "mode": self.mode,  # type: ignore[dict-item]
            "utility_mean": mean(self.utilities),
            "reward_mean": mean(self.rewards),
            "latency_p50": float(np.percentile(self.latencies, 50)) if self.latencies else 0.0,
            "latency_p99": float(np.percentile(self.latencies, 99)) if self.latencies else 0.0,
            "fallback_rate": mean(self.fallback_rates),
            "remote_plan_rate": mean(self.remote_plan_rates),
        }


def build_fleet(
    dim: int = 32,
    num_layers: int = 4,
    experts_per_layer: int = 4,
    seed: int = 0,
) -> Fleet:
    rng = np.random.default_rng(seed)
    domain_vecs = {}
    for i, d in enumerate(DOMAINS):
        v = rng.normal(size=(dim,))
        # Make domains roughly orthogonal
        for prev in domain_vecs.values():
            v = v - np.dot(v, prev) * prev
        domain_vecs[d] = v / (np.linalg.norm(v) + 1e-12)

    registry = ExpertRegistry(allow_all=True, auto_promote=True)
    ctx_dim = 16 + 16 + 1
    learner = ContextualBanditLearner(ctx_dim=ctx_dim, batch_size=32)
    router = CombinationRouter(registry=registry, learner=learner, m=2, n=8)
    router.lambda_lat = 0.002
    router.lambda_cap = 0.01

    hosts: dict[str, MoEHost] = {}
    nodes: dict[str, ExpertNode] = {}

    for domain in DOMAINS:
        model_id = f"moe-{domain}"
        node = ExpertNode(
            node_id=f"node-{domain}",
            model_id=model_id,
            base_latency_ms=1.0,
            remote_extra_latency_ms=4.0,
            acl_open=True,
        )
        layer_expert_ids: dict[int, list[int]] = {}
        gate_weights: dict[int, np.ndarray] = {}

        for ell in range(num_layers):
            ids = list(range(experts_per_layer))
            layer_expert_ids[ell] = ids
            # Gate prefers experts whose specialty matches domain
            W = rng.normal(scale=0.1, size=(experts_per_layer, dim))
            for k in ids:
                ref = ExpertRef(model_id, ell, k)
                # Expert k=0 is strongly domain-specialized; others weaker / noisy
                if k == 0:
                    spec = domain_vecs[domain].copy()
                elif k == 1:
                    # Secondary: mix with another domain (cross-useful)
                    other = DOMAINS[(DOMAINS.index(domain) + 1) % len(DOMAINS)]
                    spec = 0.6 * domain_vecs[domain] + 0.4 * domain_vecs[other]
                    spec = spec / (np.linalg.norm(spec) + 1e-12)
                else:
                    spec = rng.normal(size=(dim,))
                    spec = spec / (np.linalg.norm(spec) + 1e-12)
                module = make_expert_module(ref, dim, domain, rng, specialty=spec)
                fp = fingerprint_from_weights(module)
                # Bias fingerprint toward specialty so NN finds cross-domain useful experts
                fp = 0.5 * fp + 0.5 * np.pad(spec, (0, max(0, len(fp) - len(spec))))[: len(fp)]
                fp = fp / (np.linalg.norm(fp) + 1e-12)
                desc = ExpertDescriptor(
                    expert_ref=ref,
                    version="1.0.0",
                    dim_in=dim,
                    dim_out=dim,
                    dtype=DType.F32,
                    fingerprint=fp,
                    cost_flops=dim * dim,
                    p50_latency_ms=3.0,
                    capacity_qps=1000.0,
                    domain_tags=[domain],
                    node_id=node.node_id,
                )
                node.add_expert(module, desc)
                registry.register(desc)
                # Align gate row with specialty for local routing
                W[k] = spec + 0.05 * rng.normal(size=(dim,))
            gate_weights[ell] = W

        # Heartbeat all
        registry.heartbeat(node.node_id, None)

        host = MoEHost(
            model_id=model_id,
            node=node,
            router=router,
            learner=learner,
            num_layers=num_layers,
            dim=dim,
            top_k=2,
            domain=domain,
            gate_weights=gate_weights,
            layer_expert_ids=layer_expert_ids,
            rng=np.random.default_rng(seed + hash(domain) % 10000),
        )
        hosts[model_id] = host
        nodes[model_id] = node

    return Fleet(
        registry=registry,
        learner=learner,
        router=router,
        hosts=hosts,
        nodes=nodes,
        domain_vecs=domain_vecs,
        dim=dim,
        ctx_dim=ctx_dim,
    )


def sample_task(
    fleet: Fleet,
    rng: np.random.Generator,
    cross_domain_prob: float = 0.4,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Return (host_model_id, x, domain_vec) where domain may differ from host."""
    host_domain = rng.choice(DOMAINS)
    host_id = f"moe-{host_domain}"
    if rng.random() < cross_domain_prob:
        task_domain = rng.choice([d for d in DOMAINS if d != host_domain])
    else:
        task_domain = host_domain
    # Input biased toward task domain
    noise = rng.normal(scale=0.3, size=(fleet.dim,))
    x = fleet.domain_vecs[task_domain] + noise
    x = x / (np.linalg.norm(x) + 1e-12)
    return host_id, x, fleet.domain_vecs[task_domain]


def run_simulation(
    steps: int = 500,
    seed: int = 0,
    mode: str = "learned",
    cross_domain_prob: float = 0.45,
    budget: Budget | None = None,
    fleet: Fleet | None = None,
) -> tuple[Fleet, SimResult]:
    rng = np.random.default_rng(seed)
    fleet = fleet or build_fleet(seed=seed)
    budget = budget or Budget(
        max_remote_latency_ms=40.0,
        max_remote_experts=4,
        require_leases=True,
        allow_soft_latency=True,
    )
    result = SimResult(mode=mode)

    for t in range(steps):
        # Keep registry alive
        if t % 50 == 0:
            for node in fleet.nodes.values():
                fleet.registry.heartbeat(node.node_id, None)

        host_id, x, domain_vec = sample_task(fleet, rng, cross_domain_prob)
        host = fleet.hosts[host_id]
        use_market = mode != "local"
        _, outcome = host.forward(
            x=x,
            domain_vec=domain_vec,
            nodes=fleet.nodes,
            budget=budget,
            use_marketplace=use_market,
            force_local=(mode == "local"),
            mode=mode if mode != "local" else "local",
        )
        result.utilities.append(outcome.utility)
        result.rewards.append(outcome.reward)
        result.latencies.append(outcome.latency_ms)
        result.fallback_rates.append(1.0 if outcome.fallbacks else 0.0)
        remote = (
            outcome.plan is not None
            and not outcome.plan.local_only_equivalent
            and len(outcome.plan.remote_refs(host_id)) > 0
        )
        result.remote_plan_rates.append(1.0 if remote else 0.0)

    fleet.learner.flush()
    return fleet, result


def run_ablations(
    steps: int = 400,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Run A0–A3 ablations with independent fleets (fair cold start each)."""
    modes = ("local", "random", "heuristic", "learned")
    out: dict[str, dict[str, float]] = {}
    for i, mode in enumerate(modes):
        _, result = run_simulation(steps=steps, seed=seed + i * 17, mode=mode)
        summary = result.summary()
        # Cast mode properly for JSON-ish dict
        summary["mode"] = mode
        out[mode] = summary
    return out
