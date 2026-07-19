"""Lease binding/expiry, registry ownership, heartbeat scoping, plan TTL."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cei.node import ExpertNode, make_expert_module
from cei.registry import ExpertRegistry
from cei.simulate import build_fleet
from cei.types import (
    ActivationBatch,
    Budget,
    CombinationOp,
    CombinationPlan,
    CombinationStep,
    DType,
    ExpertDescriptor,
    ExpertRef,
)


def _desc(ref: ExpertRef, node_id: str = "n1", dim: int = 8) -> ExpertDescriptor:
    return ExpertDescriptor(
        expert_ref=ref,
        version="1.0.0",
        dim_in=dim,
        dim_out=dim,
        dtype=DType.F32,
        fingerprint=np.ones(16) / 4.0,
        cost_flops=1,
        p50_latency_ms=1.0,
        capacity_qps=100.0,
        node_id=node_id,
    )


def _node_with_expert(dim: int = 8, **node_kwargs) -> tuple[ExpertNode, ExpertRef]:
    rng = np.random.default_rng(0)
    ref = ExpertRef("moe-x", 0, 0)
    mod = make_expert_module(ref, dim, "code", rng)
    node = ExpertNode(node_id="n", model_id="moe-x", **node_kwargs)
    node.add_expert(mod, _desc(ref, node_id="n", dim=dim))
    return node, ref


# --- lease binding -----------------------------------------------------------


def test_lease_bound_to_expert():
    node, ref = _node_with_expert(acl_open=True)
    other = ExpertRef("moe-x", 0, 1)
    rng = np.random.default_rng(1)
    node.add_expert(make_expert_module(other, 8, "code", rng), _desc(other, node_id="n"))
    lease = node.lease_capacity(ref, 1.0, 5000, principal="alice")
    with pytest.raises(PermissionError, match="LEASE_MISMATCH"):
        node.forward_expert(
            other,
            ActivationBatch(tensor=np.ones(8)),
            lease_id=lease.lease_id,
            principal="alice",
            require_lease=True,
        )


def test_lease_bound_to_principal():
    node, ref = _node_with_expert(acl_open=False, acl_allow={"alice", "bob"})
    lease = node.lease_capacity(ref, 1.0, 5000, principal="alice")
    with pytest.raises(PermissionError, match="LEASE_MISMATCH"):
        node.forward_expert(
            ref,
            ActivationBatch(tensor=np.ones(8)),
            lease_id=lease.lease_id,
            principal="bob",
            require_lease=True,
        )
    out, _ = node.forward_expert(
        ref,
        ActivationBatch(tensor=np.ones(8)),
        lease_id=lease.lease_id,
        principal="alice",
        require_lease=True,
    )
    assert out.tensor.shape == (8,)


def test_expired_lease_rejected_and_purged():
    node, ref = _node_with_expert(acl_open=True)
    lease = node.lease_capacity(ref, 1.0, 5000, principal="alice")
    node.leases[lease.lease_id].deadline_ms = 0.0  # force expiry
    with pytest.raises(RuntimeError, match="CAPACITY_EXHAUSTED"):
        node.forward_expert(
            ref,
            ActivationBatch(tensor=np.ones(8)),
            lease_id=lease.lease_id,
            principal="alice",
            require_lease=True,
        )
    # New lease acquisition purges the expired one.
    node.lease_capacity(ref, 1.0, 5000, principal="alice")
    assert lease.lease_id not in node.leases


def test_release_requires_grantee_principal():
    node, ref = _node_with_expert(acl_open=False, acl_allow={"alice", "mallory"})
    lease = node.lease_capacity(ref, 1.0, 5000, principal="alice")
    with pytest.raises(PermissionError):
        node.release_capacity(lease.lease_id, principal="mallory")
    assert lease.lease_id in node.leases
    node.release_capacity(lease.lease_id, principal="alice")
    assert lease.lease_id not in node.leases


def test_lease_invalid_ttl_and_qps_rejected():
    node, ref = _node_with_expert(acl_open=True)
    with pytest.raises(ValueError, match="INVALID_TTL"):
        node.lease_capacity(ref, 1.0, 0, principal="a")
    with pytest.raises(ValueError, match="INVALID_QPS"):
        node.lease_capacity(ref, float("nan"), 5000, principal="a")


def test_lease_table_bounded():
    node, ref = _node_with_expert(acl_open=True)
    node.max_active_leases = 5
    for _ in range(5):
        node.lease_capacity(ref, 1.0, 60_000, principal="a")
    with pytest.raises(RuntimeError, match="CAPACITY_EXHAUSTED"):
        node.lease_capacity(ref, 1.0, 60_000, principal="a")


# --- registry ownership ------------------------------------------------------


def test_owner_enforced_on_reregister():
    reg = ExpertRegistry(allow_all=True, auto_promote=True, enforce_ownership=True)
    ref = ExpertRef("m", 0, 0)
    reg.register(_desc(ref), principal="node-a")
    with pytest.raises(PermissionError, match="NOT_OWNER"):
        reg.register(_desc(ref), force=True, principal="node-b")
    # Owner may refresh.
    reg.register(_desc(ref), force=True, principal="node-a")


def test_owner_enforced_on_deregister():
    reg = ExpertRegistry(allow_all=True, auto_promote=True, enforce_ownership=True)
    ref = ExpertRef("m", 0, 0)
    reg.register(_desc(ref), principal="node-a")
    with pytest.raises(PermissionError, match="NOT_OWNER"):
        reg.deregister([ref], principal="node-b")
    reg.deregister([ref], principal="node-a")
    assert reg.get(ref) is None


def test_heartbeat_scoped_to_owning_node():
    reg = ExpertRegistry(allow_all=True, auto_promote=True, heartbeat_ttl_ms=1_000_000)
    ref = ExpertRef("m", 0, 0)
    reg.register(_desc(ref, node_id="n1"))
    key = ref.key()
    # A different node cannot refresh or mutate metrics for n1's expert.
    reg.heartbeat("evil-node", [ref], capacity_qps={key: 1.0}, load_qps={key: 999.0})
    assert reg.capacity(ref) == 100.0
    assert reg.load(ref) == 0.0
    # The owning node can.
    reg.heartbeat("n1", [ref], capacity_qps={key: 55.0}, load_qps={key: 5.0})
    assert reg.capacity(ref) == 55.0
    assert reg.load(ref) == 5.0


# --- plan TTL ---------------------------------------------------------------


def test_plan_expiry_predicate():
    plan = CombinationPlan(
        plan_id="p",
        host_model_id="m",
        steps=[],
        budget=Budget(),
        ttl_ms=1000,
        issued_unix_ms=1_000_000,
    )
    assert plan.expired(1_000_500) is False
    assert plan.expired(1_001_001) is True
    # Unknown issue time → never expires (backwards compatible).
    plan.issued_unix_ms = 0
    assert plan.expired(9e15) is False


def test_expired_plan_falls_back_to_local():
    fleet = build_fleet(dim=16, num_layers=3, experts_per_layer=3, seed=11)
    host = fleet.hosts["moe-code"]
    remote_ref = ExpertRef("moe-math", 1, 0)
    stale_plan = CombinationPlan(
        plan_id="stale",
        host_model_id="moe-code",
        steps=[
            CombinationStep(
                layer_id=1,
                expert_refs=[remote_ref],
                weights=[1.0],
                op=CombinationOp.REPLACE,
            )
        ],
        budget=Budget(allow_soft_latency=True),
        ttl_ms=1,
        issued_unix_ms=int(time.time() * 1000) - 60_000,  # long past TTL
    )
    h = np.ones(16) / 4.0
    y, lat, fallbacks = host.execute_layer(
        layer_id=1,
        h=h,
        plan=stale_plan,
        budget=stale_plan.budget,
        nodes=fleet.nodes,
        require_leases=False,
    )
    assert fallbacks and fallbacks[0].reason == "PLAN_EXPIRED"
    assert y.shape == (16,)


def test_fresh_plan_executes_remote():
    fleet = build_fleet(dim=16, num_layers=3, experts_per_layer=3, seed=11)
    host = fleet.hosts["moe-code"]
    remote_ref = ExpertRef("moe-math", 1, 0)
    plan = CombinationPlan(
        plan_id="fresh",
        host_model_id="moe-code",
        steps=[
            CombinationStep(
                layer_id=1,
                expert_refs=[remote_ref],
                weights=[1.0],
                op=CombinationOp.REPLACE,
            )
        ],
        budget=Budget(allow_soft_latency=True),
        ttl_ms=60_000,
        issued_unix_ms=int(time.time() * 1000),
    )
    h = np.ones(16) / 4.0
    _, _, fallbacks = host.execute_layer(
        layer_id=1,
        h=h,
        plan=plan,
        budget=plan.budget,
        nodes=fleet.nodes,
        require_leases=False,
    )
    assert fallbacks == []


def test_wire_plan_roundtrips_issue_time():
    from cei import wire

    plan = CombinationPlan(
        plan_id="p",
        host_model_id="m",
        steps=[],
        budget=Budget(),
        ttl_ms=1234,
        issued_unix_ms=987_654_321,
    )
    p2 = wire.plan_from_pb(wire.plan_to_pb(plan))
    assert p2.issued_unix_ms == 987_654_321
    assert p2.ttl_ms == 1234


# --- host releases leases on failure ------------------------------------------


def test_lease_released_when_forward_exceeds_budget():
    fleet = build_fleet(dim=16, num_layers=3, experts_per_layer=3, seed=3)
    host = fleet.hosts["moe-code"]
    remote_node = fleet.nodes["moe-math"]
    remote_ref = ExpertRef("moe-math", 1, 0)
    step = CombinationStep(
        layer_id=1,
        expert_refs=[remote_ref],
        weights=[1.0],
        op=CombinationOp.REPLACE,
    )
    # Impossible latency budget forces DEADLINE_EXCEEDED after the forward.
    budget = Budget(max_remote_latency_ms=0.0, allow_soft_latency=False)
    with pytest.raises(TimeoutError):
        host._execute_step(step, np.ones(16) / 4.0, fleet.nodes, True, budget)
    assert remote_node.leases == {}
