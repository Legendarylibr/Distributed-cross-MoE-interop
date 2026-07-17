"""Tests for architecture fixes: policy cache, layer compat, quotas, adapter hub."""

from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np
import pytest

from cei.adapters import AdapterHub
from cei.learner import ContextualBanditLearner, score_plan_from_snapshot
from cei.pb import cei_internal_pb2, cei_internal_pb2_grpc
from cei.registry import ExpertRegistry
from cei.router import CombinationRouter
from cei.server.adapter_hub_servicer import AdapterHubServicer
from cei.types import (
    Budget,
    CombinationOp,
    CombinationPlan,
    CombinationStep,
    DType,
    ExpertDescriptor,
    ExpertRef,
    Outcome,
)
from cei import wire
from cei.node import fingerprint_from_weights, make_expert_module


def test_registry_quota_exceeded():
    reg = ExpertRegistry(max_experts_per_model=2, allow_all=True, auto_promote=True)
    rng = np.random.default_rng(0)
    for k in range(2):
        ref = ExpertRef("m", 0, k)
        mod = make_expert_module(ref, 8, "code", rng)
        reg.register(
            ExpertDescriptor(
                expert_ref=ref,
                version="1.0.0",
                dim_in=8,
                dim_out=8,
                dtype=DType.F32,
                fingerprint=fingerprint_from_weights(mod, 16),
                cost_flops=1,
                p50_latency_ms=1.0,
                capacity_qps=10,
                node_id="n",
            )
        )
    ref = ExpertRef("m", 0, 99)
    mod = make_expert_module(ref, 8, "code", rng)
    with pytest.raises(ValueError, match="QUOTA_EXCEEDED"):
        reg.register(
            ExpertDescriptor(
                expert_ref=ref,
                version="1.0.0",
                dim_in=8,
                dim_out=8,
                dtype=DType.F32,
                fingerprint=fingerprint_from_weights(mod, 16),
                cost_flops=1,
                p50_latency_ms=1.0,
                capacity_qps=10,
                node_id="n",
            )
        )


def test_layer_compat_exact_filters_wrong_layer():
    reg = ExpertRegistry(allow_all=True, auto_promote=True)
    learner = ContextualBanditLearner(ctx_dim=8, batch_size=10)
    router = CombinationRouter(registry=reg, learner=learner, layer_compat="exact_layer", m=1)
    rng = np.random.default_rng(1)
    # Remote expert at layer 2
    ref = ExpertRef("moe-math", 2, 0)
    mod = make_expert_module(ref, 8, "math", rng)
    fp = fingerprint_from_weights(mod, 16)
    reg.register(
        ExpertDescriptor(
            expert_ref=ref,
            version="1.0.0",
            dim_in=8,
            dim_out=8,
            dtype=DType.F32,
            fingerprint=fp,
            cost_flops=1,
            p50_latency_ms=1.0,
            capacity_qps=10,
            domain_tags=["math"],
            node_id="n",
        )
    )
    reg.heartbeat("n", [ref])
    local_ref = ExpertRef("moe-code", 1, 0)
    local_topk = {1: [(local_ref, 1.0)]}
    # Query with matching fingerprint so NN would find remote if layer allowed
    plans = router.propose(
        host_model_id="moe-code",
        context_embedding=fp[:8] if len(fp) >= 8 else np.pad(fp, (0, 8 - len(fp))),
        local_topk=local_topk,
        host_dim=8,
        budget=Budget(allow_soft_latency=True, max_remote_latency_ms=100),
    )
    remotes = [p for p in plans if not p.local_only_equivalent]
    # exact_layer: host layer 1 vs remote layer 2 → no remote plans
    assert remotes == []


def test_policy_snapshot_roundtrip_scoring():
    learner = ContextualBanditLearner(ctx_dim=4, batch_size=1)
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
    phi = np.ones(4)
    learner.report(
        Outcome(
            plan_id="p",
            host_model_id="m",
            reward=1.5,
            utility=1.5,
            latency_ms=1.0,
            capacity_penalty=0.0,
            tokens=1,
            context_embedding=phi,
            plan=plan,
        )
    )
    snap = learner.policy_snapshot()
    assert snap["version"] >= 1
    score = score_plan_from_snapshot(phi, plan, snap)
    assert score is not None
    assert np.isfinite(score)


def test_adapter_hub_grpc():
    hub = AdapterHub()
    rng = np.random.default_rng(0)
    adapter = AdapterHub.identity("id-8", 8, rng)
    hub.register(adapter)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    cei_internal_pb2_grpc.add_AdapterHubServicer_to_server(AdapterHubServicer(hub), server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = cei_internal_pb2_grpc.AdapterHubStub(channel)
    resp = stub.GetAdapter(
        cei_internal_pb2.GetAdapterRequest(meta=wire.new_meta(), adapter_id="id-8")
    )
    assert not resp.error_code
    assert resp.adapter.adapter_id == "id-8"
    channel.close()
    server.stop(0)
