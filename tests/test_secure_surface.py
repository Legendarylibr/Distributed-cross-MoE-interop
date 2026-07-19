"""Secure-profile gRPC surface: heartbeat ACL, RunStep ACL, propose auth."""

from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np
import pytest

from cei import security, wire
from cei.pb import cei_internal_pb2, cei_pb2, cei_pb2_grpc
from cei.server.node_servicer import HostServicer
from cei.server.registry_servicer import RegistryServicer
from cei.simulate import build_fleet


@pytest.fixture
def secure_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CEI_SECURITY_PROFILE", "secure")
    monkeypatch.setenv("CEI_AUTO_PROMOTE", "0")
    monkeypatch.setenv("CEI_REGISTRY_ALLOW_ALL", "0")
    monkeypatch.setenv("CEI_REGISTRY_PUBLISHERS", "node-code")
    monkeypatch.setenv("CEI_REGISTRY_CONSUMERS", "cei-router")
    monkeypatch.setenv("CEI_NODE_ACL_OPEN", "0")
    monkeypatch.setenv("CEI_NODE_ACL_ALLOW", "cei-driver")
    security.reset_config_cache()
    yield
    security.reset_config_cache()


class _FakeContext:
    """Minimal grpc.ServicerContext stand-in for direct servicer calls."""

    def __init__(self) -> None:
        self.code = None
        self.details = None

    def set_code(self, code) -> None:
        self.code = code

    def set_details(self, details) -> None:
        self.details = details

    def auth_context(self):
        return {}


def _register(stub_or_servicer, principal: str):
    fp = list(np.ones(16) / 4.0)
    req = cei_pb2.RegisterExpertRequest(
        meta=wire.new_meta(principal),
        descriptor=cei_pb2.ExpertDescriptor(
            expert_ref=cei_pb2.ExpertRef(model_id="m", layer_id=0, expert_id=0),
            version="1.0.0",
            dim_in=8,
            dim_out=8,
            dtype=cei_pb2.F32,
            fingerprint=fp,
            cost_flops=1,
            p50_latency_ms=1.0,
            capacity_qps=10,
            node_id="n1",
        ),
        promote=True,
    )
    return req


def test_heartbeat_requires_publisher(secure_env):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = RegistryServicer()
    cei_pb2_grpc.add_ExpertRegistryServicer_to_server(servicer, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = cei_pb2_grpc.ExpertRegistryStub(channel)
    stub.RegisterExpert(_register(stub, "node-code"))

    denied = stub.Heartbeat(
        cei_pb2.HeartbeatRequest(meta=wire.new_meta("attacker"), node_id="n1")
    )
    assert denied.ok is False
    ok = stub.Heartbeat(
        cei_pb2.HeartbeatRequest(meta=wire.new_meta("node-code"), node_id="n1")
    )
    assert ok.ok is True
    channel.close()
    server.stop(0)


def test_heartbeat_cannot_poison_capacity(secure_env):
    servicer = RegistryServicer()
    servicer.RegisterExpert(_register(None, "node-code"), _FakeContext())
    # Publisher-authenticated but wrong node: metrics must not change.
    resp = servicer.Heartbeat(
        cei_pb2.HeartbeatRequest(
            meta=wire.new_meta("node-code"),
            node_id="other-node",
            expert_refs=[cei_pb2.ExpertRef(model_id="m", layer_id=0, expert_id=0)],
            capacity_qps={"m:0:0": 1.0},
            load_qps={"m:0:0": 9999.0},
        ),
        _FakeContext(),
    )
    assert resp.ok is True
    from cei.types import ExpertRef

    assert servicer.registry.capacity(ExpertRef("m", 0, 0)) == 10.0
    assert servicer.registry.load(ExpertRef("m", 0, 0)) == 0.0


def test_deregister_requires_owner(secure_env):
    servicer = RegistryServicer()
    servicer.RegisterExpert(_register(None, "node-code"), _FakeContext())
    # node-code is the only configured publisher, so an attacker cannot even
    # reach ownership checks; ownership blocks a second publisher.
    resp = servicer.Deregister(
        cei_pb2.DeregisterRequest(
            meta=wire.new_meta("attacker"),
            expert_refs=[cei_pb2.ExpertRef(model_id="m", layer_id=0, expert_id=0)],
        ),
        _FakeContext(),
    )
    assert resp.ok is False


def test_runstep_gated_by_node_acl(secure_env):
    fleet = build_fleet(dim=16, num_layers=2, experts_per_layer=2, seed=0)
    host = fleet.hosts["moe-code"]
    host.node.acl_open = False
    host.node.acl_allow = {"cei-driver"}
    servicer = HostServicer(host, peer_addrs={}, router_client=None, learner_client=None)

    ctx = _FakeContext()
    denied = servicer.RunStep(
        cei_internal_pb2.RunStepRequest(
            meta=wire.new_meta("attacker"),
            x=[0.1] * 16,
            domain_vec=[0.1] * 16,
            mode="local",
        ),
        ctx,
    )
    assert denied.error_code == "ACL_DENIED"
    assert ctx.code == grpc.StatusCode.PERMISSION_DENIED

    allowed = servicer.RunStep(
        cei_internal_pb2.RunStepRequest(
            meta=wire.new_meta("cei-driver"),
            x=[0.1] * 16,
            domain_vec=[0.1] * 16,
            mode="local",
        ),
        _FakeContext(),
    )
    assert not allowed.error_code
    assert len(allowed.hidden) == 16
