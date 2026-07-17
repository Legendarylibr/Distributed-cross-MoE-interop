"""Regression tests for review fixes: budget wire, capacity cache, adapters."""

from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np

from cei import wire
from cei.adapters import AdapterHub
from cei.client import RegistryClient
from cei.node import ExpertNode, fingerprint_from_weights, make_expert_module
from cei.pb import cei_pb2, cei_pb2_grpc
from cei.server.registry_servicer import RegistryServicer
from cei.types import ActivationBatch, Budget, DType, ExpertDescriptor, ExpertRef


def test_budget_optional_bools_roundtrip():
    b = Budget(require_leases=False, allow_soft_latency=False, strict_local_fallback=True)
    msg = wire.budget_to_pb(b)
    assert msg.HasField("require_leases")
    assert msg.require_leases is False
    assert msg.allow_soft_latency is False
    assert msg.strict_local_fallback is True
    b2 = wire.budget_from_pb(msg)
    assert b2.require_leases is False
    assert b2.allow_soft_latency is False
    assert b2.strict_local_fallback is True


def test_budget_defaults_when_unset():
    msg = cei_pb2.Budget(max_remote_latency_ms=10.0, max_remote_experts=2)
    b = wire.budget_from_pb(msg)
    assert b.require_leases is True  # SPEC default when unset
    assert b.allow_soft_latency is False


def test_registry_client_load_capacity_from_describe():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    reg = RegistryServicer()
    cei_pb2_grpc.add_ExpertRegistryServicer_to_server(reg, server)
    port = server.add_insecure_port("localhost:0")
    server.start()

    ref = ExpertRef("m", 0, 0)
    rng = np.random.default_rng(0)
    mod = make_expert_module(ref, 8, "code", rng)
    fp = fingerprint_from_weights(mod, 16)
    d = ExpertDescriptor(
        expert_ref=ref,
        version="1.0.0",
        dim_in=8,
        dim_out=8,
        dtype=DType.F32,
        fingerprint=fp,
        cost_flops=1,
        p50_latency_ms=1.0,
        capacity_qps=42.0,
        node_id="n1",
    )
    reg.registry.register(d)
    reg.registry.heartbeat(
        "n1", [ref], capacity_qps={ref.key(): 42.0}, load_qps={ref.key(): 7.5}
    )

    client = RegistryClient(f"localhost:{port}")
    client.connect()
    got = client.get(ref)
    assert got is not None
    assert client.capacity(ref) == 42.0
    assert client.load(ref) == 7.5
    client.close()
    server.stop(0)


def test_adapter_forward_on_node():
    rng = np.random.default_rng(1)
    hub = AdapterHub()
    adapter = AdapterHub.random_proj("a1", dim_host=8, dim_remote=4, rng=rng)
    hub.register(adapter)
    ref = ExpertRef("m", 0, 0)
    # Expert in remote dim
    mod = make_expert_module(ref, 4, "code", rng)
    node = ExpertNode(node_id="n", model_id="m", adapter_hub=hub, acl_open=True)
    fp = fingerprint_from_weights(mod, 16)
    node.add_expert(
        mod,
        ExpertDescriptor(
            expert_ref=ref,
            version="1.0.0",
            dim_in=4,
            dim_out=4,
            dtype=DType.F32,
            fingerprint=fp,
            cost_flops=1,
            p50_latency_ms=1.0,
            capacity_qps=10,
            adapter_id="a1",
        ),
    )
    h = rng.normal(size=(8,))
    out, lat = node.forward_expert(
        ref,
        ActivationBatch(tensor=h),
        adapter_id="a1",
        request_id="r1",
        require_lease=False,
    )
    assert out.tensor.shape == (8,)
    assert lat > 0
