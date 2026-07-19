"""Thread-safety smoke tests: registry, node, and learner under contention."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cei.learner import ContextualBanditLearner
from cei.node import ExpertNode, make_expert_module
from cei.registry import ExpertRegistry
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


def _desc(ref: ExpertRef, node_id: str = "n1") -> ExpertDescriptor:
    return ExpertDescriptor(
        expert_ref=ref,
        version="1.0.0",
        dim_in=8,
        dim_out=8,
        dtype=DType.F32,
        fingerprint=np.ones(16) / 4.0,
        cost_flops=1,
        p50_latency_ms=1.0,
        capacity_qps=1e9,
        node_id=node_id,
    )


def test_registry_concurrent_register_heartbeat_query():
    reg = ExpertRegistry(allow_all=True, auto_promote=True, max_experts_per_model=4096)
    errors: list[Exception] = []

    def register(i: int) -> None:
        try:
            ref = ExpertRef("m", i % 8, i)
            reg.register(_desc(ref))
            reg.heartbeat("n1", [ref])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def query(_: int) -> None:
        try:
            reg.describe_nn(np.ones(16), k=8)
            reg.describe_explicit([ExpertRef("m", 0, 0)], principal=None)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as pool:
        for i in range(200):
            pool.submit(register, i)
            pool.submit(query, i)
    assert errors == []


def test_node_concurrent_lease_forward_release():
    rng = np.random.default_rng(0)
    ref = ExpertRef("m", 0, 0)
    node = ExpertNode(node_id="n", model_id="m", acl_open=True)
    node.add_expert(make_expert_module(ref, 8, "code", rng), _desc(ref, node_id="n"))
    errors: list[Exception] = []

    def cycle(i: int) -> None:
        try:
            lease = node.lease_capacity(ref, 1.0, 5000, principal="p")
            node.forward_expert(
                ref,
                ActivationBatch(tensor=np.ones(8)),
                lease_id=lease.lease_id,
                request_id=f"r{i}",
                principal="p",
                require_lease=True,
            )
            node.release_capacity(lease.lease_id, principal="p")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as pool:
        for i in range(300):
            pool.submit(cycle, i)
    assert errors == []
    assert node.leases == {}


def test_learner_concurrent_report_and_snapshot():
    learner = ContextualBanditLearner(ctx_dim=8, batch_size=4)
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
    errors: list[Exception] = []

    def report(i: int) -> None:
        try:
            learner.report(
                Outcome(
                    plan_id="p",
                    host_model_id="m",
                    reward=float(i % 3),
                    utility=1.0,
                    latency_ms=1.0,
                    capacity_penalty=0.0,
                    tokens=1,
                    context_embedding=np.ones(8),
                    plan=plan,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def snapshot(_: int) -> None:
        try:
            learner.policy_snapshot()
            learner.estimate_utility(np.ones(8), plan)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as pool:
        for i in range(200):
            pool.submit(report, i)
            pool.submit(snapshot, i)
    learner.flush()
    assert errors == []
    assert learner.version >= 1
