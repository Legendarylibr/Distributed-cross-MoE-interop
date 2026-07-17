"""Tests for wire mapping and in-process gRPC servicers."""

from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np
import pytest

from cei import wire
from cei.learner import ContextualBanditLearner
from cei.pb import cei_internal_pb2_grpc, cei_pb2, cei_pb2_grpc
from cei.server.learner_servicer import LearnerInternalServicer, LearnerServicer
from cei.server.registry_servicer import RegistryServicer
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


def test_wire_roundtrip_descriptor_and_activation():
    ref = ExpertRef("moe-code", 1, 0)
    fp = np.linspace(0, 1, 32)
    fp = fp / np.linalg.norm(fp)
    d = ExpertDescriptor(
        expert_ref=ref,
        version="1.0.0",
        dim_in=32,
        dim_out=32,
        dtype=DType.F32,
        fingerprint=fp,
        cost_flops=100,
        p50_latency_ms=2.0,
        capacity_qps=10.0,
        domain_tags=["code"],
        node_id="n",
    )
    d2 = wire.descriptor_from_pb(wire.descriptor_to_pb(d))
    assert d2.expert_ref == d.expert_ref
    assert np.allclose(d2.fingerprint, d.fingerprint)

    act = ActivationBatch(tensor=np.arange(16, dtype=np.float64))
    act2 = wire.activation_from_pb(wire.activation_to_pb(act))
    assert np.allclose(act.tensor, act2.tensor)


def test_wire_plan_roundtrip():
    plan = CombinationPlan(
        plan_id="p1",
        host_model_id="moe-code",
        steps=[
            CombinationStep(
                layer_id=1,
                expert_refs=[ExpertRef("moe-math", 1, 0)],
                weights=[1.0],
                op=CombinationOp.REPLACE,
            )
        ],
        budget=Budget(allow_soft_latency=True),
        score=0.5,
    )
    p2 = wire.plan_from_pb(wire.plan_to_pb(plan))
    assert p2.plan_id == plan.plan_id
    assert p2.steps[0].expert_refs[0].model_id == "moe-math"


def test_registry_servicer_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    cei_pb2_grpc.add_ExpertRegistryServicer_to_server(RegistryServicer(), server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = cei_pb2_grpc.ExpertRegistryStub(channel)
    fp = [0.1] * 16
    n = float(np.linalg.norm(fp))
    fp = [x / n for x in fp]
    resp = stub.RegisterExpert(
        cei_pb2.RegisterExpertRequest(
            meta=wire.new_meta(),
            descriptor=cei_pb2.ExpertDescriptor(
                expert_ref=cei_pb2.ExpertRef(model_id="m", layer_id=0, expert_id=0),
                version="1.0.0",
                dim_in=16,
                dim_out=16,
                dtype=cei_pb2.F32,
                fingerprint=fp,
                cost_flops=1,
                p50_latency_ms=1.0,
                capacity_qps=10,
                node_id="n1",
            ),
        )
    )
    assert resp.ok
    stub.Heartbeat(
        cei_pb2.HeartbeatRequest(meta=wire.new_meta(), node_id="n1")
    )
    desc = stub.DescribeExperts(
        cei_pb2.DescribeExpertsRequest(
            meta=wire.new_meta(),
            nn=cei_pb2.NNQuery(fingerprint=fp, k=5, host_dim_in=16, host_dim_out=16),
        )
    )
    assert len(desc.experts) == 1
    channel.close()
    server.stop(0)


def test_learner_estimate_utility_grpc():
    learner = ContextualBanditLearner(ctx_dim=8, batch_size=10)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    cei_pb2_grpc.add_CombinationLearnerServicer_to_server(LearnerServicer(learner), server)
    cei_internal_pb2_grpc.add_LearnerInternalServicer_to_server(
        LearnerInternalServicer(learner), server
    )
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    from cei.pb import cei_internal_pb2

    stub = cei_internal_pb2_grpc.LearnerInternalStub(channel)
    plan = CombinationPlan(
        plan_id="p",
        host_model_id="m",
        steps=[],
        budget=Budget(),
        local_only_equivalent=True,
    )
    resp = stub.EstimateUtility(
        cei_internal_pb2.EstimateUtilityRequest(
            meta=wire.new_meta(),
            context_embedding=[0.1] * 8,
            plan=wire.plan_to_pb(plan),
            fingerprint_sims={},
            alpha=0.5,
        )
    )
    assert np.isfinite(resp.utility)
    channel.close()
    server.stop(0)
