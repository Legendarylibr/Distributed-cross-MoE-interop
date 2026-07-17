"""Security canaries for deny-by-default ACLs, promotion, attestation, adapters."""

from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np
import pytest

from cei import security, wire
from cei.adapters import AdapterHub
from cei.learner import ContextualBanditLearner
from cei.node import ExpertNode, fingerprint_from_weights, make_expert_module
from cei.pb import cei_internal_pb2, cei_internal_pb2_grpc, cei_pb2, cei_pb2_grpc
from cei.registry import ExpertRegistry
from cei.server.adapter_hub_servicer import AdapterHubServicer
from cei.server.learner_servicer import LearnerServicer
from cei.server.node_servicer import NodeServicer
from cei.server.registry_servicer import RegistryServicer
from cei.types import (
    ActivationBatch,
    DType,
    ExpertDescriptor,
    ExpertRef,
    Outcome,
)


@pytest.fixture
def secure_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CEI_SECURITY_PROFILE", "secure")
    monkeypatch.setenv("CEI_OUTCOME_HMAC_SECRET", "test-secret")
    monkeypatch.setenv("CEI_REQUIRE_OUTCOME_ATTESTATION", "1")
    monkeypatch.setenv("CEI_AUTO_PROMOTE", "0")
    monkeypatch.setenv("CEI_REGISTRY_ALLOW_ALL", "0")
    monkeypatch.setenv("CEI_REGISTRY_PUBLISHERS", "node-code")
    monkeypatch.setenv("CEI_REGISTRY_CONSUMERS", "cei-router,host-code")
    monkeypatch.setenv("CEI_ADAPTER_WRITERS", "node-code")
    monkeypatch.setenv("CEI_NODE_ACL_OPEN", "0")
    monkeypatch.setenv("CEI_NODE_ACL_ALLOW", "host-code")
    monkeypatch.setenv("CEI_PRIORITY_ADMINS", "ops-admin")
    monkeypatch.delenv("CEI_TRUST_META_PRINCIPAL", raising=False)
    security.reset_config_cache()
    yield
    security.reset_config_cache()


def _desc(ref: ExpertRef, fp: np.ndarray) -> ExpertDescriptor:
    return ExpertDescriptor(
        expert_ref=ref,
        version="1.0.0",
        dim_in=8,
        dim_out=8,
        dtype=DType.F32,
        fingerprint=fp,
        cost_flops=1,
        p50_latency_ms=1.0,
        capacity_qps=10,
        node_id="n1",
    )


def test_unpromoted_expert_not_routable():
    reg = ExpertRegistry(allow_all=True, auto_promote=False)
    rng = np.random.default_rng(0)
    ref = ExpertRef("m", 0, 0)
    mod = make_expert_module(ref, 8, "code", rng)
    fp = fingerprint_from_weights(mod, 16)
    reg.register(_desc(ref, fp), promote=False)
    reg.heartbeat("n1", [ref])
    hits = reg.describe_nn(fp, k=5, host_dim_in=8, host_dim_out=8)
    assert hits and hits[0][2] is False
    reg.promote([ref])
    hits2 = reg.describe_nn(fp, k=5, host_dim_in=8, host_dim_out=8)
    assert hits2[0][2] is True


def test_registry_acl_denies_unknown_principal():
    reg = ExpertRegistry(allow_all=False, auto_promote=True)
    rng = np.random.default_rng(1)
    ref = ExpertRef("m", 0, 0)
    mod = make_expert_module(ref, 8, "code", rng)
    fp = fingerprint_from_weights(mod, 16)
    reg.register(_desc(ref, fp), promote=True, principal="node-code")
    reg.heartbeat("n1", [ref])
    assert reg.describe_nn(fp, k=5, principal="attacker") == []
    assert reg.describe_nn(fp, k=5, principal="node-code")


def test_registry_publish_acl(secure_env):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    cei_pb2_grpc.add_ExpertRegistryServicer_to_server(RegistryServicer(), server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = cei_pb2_grpc.ExpertRegistryStub(channel)
    fp = [0.1] * 8
    n = float(np.linalg.norm(fp))
    fp = [x / n for x in fp]
    desc = cei_pb2.ExpertDescriptor(
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
    )
    denied = stub.RegisterExpert(
        cei_pb2.RegisterExpertRequest(meta=wire.new_meta("attacker"), descriptor=desc, promote=True)
    )
    assert denied.ok is False
    assert denied.error_code == "ACL_DENIED"
    ok = stub.RegisterExpert(
        cei_pb2.RegisterExpertRequest(
            meta=wire.new_meta("node-code"), descriptor=desc, promote=True
        )
    )
    assert ok.ok is True
    channel.close()
    server.stop(0)


def test_node_acl_and_priority_bypass(secure_env):
    rng = np.random.default_rng(2)
    ref = ExpertRef("moe-code", 0, 0)
    mod = make_expert_module(ref, 8, "code", rng)
    node = ExpertNode(
        node_id="n",
        model_id="moe-code",
        acl_open=False,
        acl_allow={"host-code", "ops-admin"},
        priority_admins={"ops-admin"},
    )
    node.add_expert(mod, _desc(ref, fingerprint_from_weights(mod, 16)))
    with pytest.raises(PermissionError):
        node.lease_capacity(ref, 1.0, 1000, principal="attacker")
    node._load_tokens = 10_000.0
    with pytest.raises(RuntimeError, match="CAPACITY"):
        node.lease_capacity(ref, 1.0, 1000, principal="host-code", priority=99)
    lease = node.lease_capacity(ref, 1.0, 1000, principal="ops-admin", priority=99)
    assert lease.lease_id


def test_outcome_attestation_required(secure_env):
    learner = ContextualBanditLearner(ctx_dim=4, batch_size=8)
    servicer = LearnerServicer(learner)
    outcome = Outcome(
        plan_id="p1",
        host_model_id="moe-code",
        reward=1.0,
        utility=1.0,
        latency_ms=1.0,
        capacity_penalty=0.0,
        tokens=1,
        context_embedding=np.zeros(4),
    )
    bad = wire.outcome_to_report_pb(outcome, attestation="")
    resp = servicer.ReportOutcome(bad, context=None)
    assert resp.ok is False
    good = wire.outcome_to_report_pb(outcome)  # auto-sign with secret
    assert good.attestation
    resp2 = servicer.ReportOutcome(good, context=None)
    assert resp2.ok is True


def test_adapter_writer_acl_and_digest(secure_env):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    hub = AdapterHubServicer()
    cei_internal_pb2_grpc.add_AdapterHubServicer_to_server(hub, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = cei_internal_pb2_grpc.AdapterHubStub(channel)
    rng = np.random.default_rng(3)
    adapter = AdapterHub.identity("id8", 8, rng)
    w_in = np.asarray(adapter.w_in, dtype=np.float64).tobytes()
    w_out = np.asarray(adapter.w_out, dtype=np.float64).tobytes()
    digest = security.adapter_digest(w_in, w_out)
    blob = cei_internal_pb2.AdapterBlob(
        adapter_id=adapter.adapter_id,
        dim_in_host=8,
        dim_in_remote=8,
        dim_out_remote=8,
        dim_out_host=8,
        w_in=w_in,
        w_out=w_out,
        w_in_shape=list(adapter.w_in.shape),
        w_out_shape=list(adapter.w_out.shape),
        content_digest=digest,
    )
    deny = stub.UpsertAdapter(
        cei_internal_pb2.UpsertAdapterRequest(meta=wire.new_meta("attacker"), adapter=blob)
    )
    assert deny.ok is False
    ok = stub.UpsertAdapter(
        cei_internal_pb2.UpsertAdapterRequest(meta=wire.new_meta("node-code"), adapter=blob)
    )
    assert ok.ok is True
    assert ok.content_digest == digest
    channel.close()
    server.stop(0)


def test_forward_export_audit_deny(secure_env):
    rng = np.random.default_rng(4)
    ref = ExpertRef("moe-code", 0, 0)
    mod = make_expert_module(ref, 8, "code", rng)
    node = ExpertNode(
        node_id="n",
        model_id="moe-code",
        acl_open=False,
        acl_allow={"host-code"},
    )
    node.add_expert(mod, _desc(ref, fingerprint_from_weights(mod, 16)))
    servicer = NodeServicer(node)
    resp = servicer.ForwardExpert(
        cei_pb2.ForwardExpertRequest(
            meta=wire.new_meta("attacker"),
            expert_ref=wire.expert_ref_to_pb(ref),
            activation=wire.activation_to_pb(ActivationBatch(tensor=np.ones(8))),
        ),
        context=None,
    )
    assert resp.error_code == "ACL_DENIED"
    export = servicer.ExportWeights(
        cei_pb2.ExportWeightsRequest(meta=wire.new_meta("host-code")),
        context=None,
    )
    assert export.error_code == "ACL_DENIED"
