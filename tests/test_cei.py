"""Unit tests for CEI reference implementation."""

from __future__ import annotations

import numpy as np
import pytest

from cei.learner import ContextualBanditLearner
from cei.node import ExpertNode, fingerprint_from_weights, make_expert_module
from cei.registry import ExpertRegistry
from cei.simulate import build_fleet, run_ablations, run_simulation
from cei.types import (
    ActivationBatch,
    Budget,
    CombinationOp,
    CombinationPlan,
    CombinationStep,
    DType,
    ExpertDescriptor,
    ExpertRef,
    Outcome,
)


def test_registry_nn_and_heartbeat():
    rng = np.random.default_rng(0)
    reg = ExpertRegistry(heartbeat_ttl_ms=60_000, allow_all=True, auto_promote=True)
    ref = ExpertRef("m", 0, 0)
    mod = make_expert_module(ref, 16, "code", rng)
    fp = fingerprint_from_weights(mod, dim_fp=32)
    desc = ExpertDescriptor(
        expert_ref=ref,
        version="1.0.0",
        dim_in=16,
        dim_out=16,
        dtype=DType.F32,
        fingerprint=fp,
        cost_flops=100,
        p50_latency_ms=2.0,
        capacity_qps=100,
        domain_tags=["code"],
        node_id="n1",
    )
    reg.register(desc)
    reg.heartbeat("n1", [ref])
    hits = reg.describe_nn(fp, k=5, host_dim_in=16, host_dim_out=16)
    assert hits and hits[0][0].expert_ref == ref
    assert hits[0][2] is True


def test_stale_version_rejected():
    rng = np.random.default_rng(1)
    reg = ExpertRegistry(allow_all=True, auto_promote=True)
    ref = ExpertRef("m", 0, 0)
    mod = make_expert_module(ref, 8, "math", rng)
    fp = fingerprint_from_weights(mod, dim_fp=16)
    d1 = ExpertDescriptor(
        expert_ref=ref,
        version="2.0.0",
        dim_in=8,
        dim_out=8,
        dtype=DType.F32,
        fingerprint=fp,
        cost_flops=1,
        p50_latency_ms=1.0,
        capacity_qps=10,
        node_id="n",
    )
    reg.register(d1)
    d0 = ExpertDescriptor(
        expert_ref=ref,
        version="1.0.0",
        dim_in=8,
        dim_out=8,
        dtype=DType.F32,
        fingerprint=fp,
        cost_flops=1,
        p50_latency_ms=1.0,
        capacity_qps=10,
        node_id="n",
    )
    with pytest.raises(ValueError, match="STALE_VERSION"):
        reg.register(d0)


def test_forward_expert_lease_and_acl():
    rng = np.random.default_rng(2)
    ref = ExpertRef("moe-code", 1, 0)
    mod = make_expert_module(ref, 16, "code", rng)
    node = ExpertNode(node_id="n", model_id="moe-code", acl_allow={"alice"})
    fp = fingerprint_from_weights(mod, 32)
    desc = ExpertDescriptor(
        expert_ref=ref,
        version="1.0.0",
        dim_in=16,
        dim_out=16,
        dtype=DType.F32,
        fingerprint=fp,
        cost_flops=1,
        p50_latency_ms=1.0,
        capacity_qps=100,
    )
    node.add_expert(mod, desc)
    with pytest.raises(PermissionError):
        node.lease_capacity(ref, 1.0, 1000, principal="bob")
    lease = node.lease_capacity(ref, 1.0, 5000, principal="alice")
    h = rng.normal(size=(16,))
    out, lat = node.forward_expert(
        ref,
        ActivationBatch(tensor=h),
        lease_id=lease.lease_id,
        request_id="r1",
        principal="alice",
        require_lease=True,
    )
    assert out.tensor.shape == (16,)
    assert lat > 0


def test_router_includes_local_only():
    fleet = build_fleet(dim=16, num_layers=3, experts_per_layer=3, seed=3)
    host = fleet.hosts["moe-code"]
    h = np.ones(16) / 4
    local = host.all_layer_topk_for_propose(h)
    phi = np.zeros(fleet.ctx_dim)
    plans = fleet.router.propose(
        host_model_id="moe-code",
        context_embedding=phi,
        local_topk=local,
        host_dim=16,
        budget=Budget(allow_soft_latency=True, max_remote_latency_ms=100),
    )
    assert any(p.local_only_equivalent for p in plans)
    assert len(plans) >= 1


def test_learner_updates():
    learner = ContextualBanditLearner(ctx_dim=8, batch_size=2)
    phi = np.ones(8)
    plan = CombinationPlan(
        plan_id="p",
        host_model_id="m",
        steps=[
            CombinationStep(
                layer_id=0,
                expert_refs=[ExpertRef("m", 0, 0)],
                weights=[1.0],
                op=CombinationOp.REPLACE,
            )
        ],
        budget=Budget(),
        local_only_equivalent=True,
    )
    for r in (0.5, 1.0):
        learner.report(
            Outcome(
                plan_id="p",
                host_model_id="m",
                reward=r,
                utility=r,
                latency_ms=1.0,
                capacity_penalty=0.0,
                tokens=1,
                context_embedding=phi,
                plan=plan,
            )
        )
    assert learner.version >= 1
    score = learner.estimate_utility(phi, plan)
    assert np.isfinite(score)


def test_simulation_runs():
    _, result = run_simulation(steps=40, seed=5, mode="learned")
    assert len(result.utilities) == 40
    assert np.isfinite(result.summary()["utility_mean"])


def test_ablations_all_modes():
    results = run_ablations(steps=60, seed=7)
    for mode in ("local", "random", "heuristic", "learned"):
        assert mode in results
        assert "utility_mean" in results[mode]
