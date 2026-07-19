"""Boundary validation: tensors, descriptors, adapters, registry inputs."""

from __future__ import annotations

from concurrent import futures

import grpc
import numpy as np
import pytest

from cei import security, wire
from cei.node import ExpertNode, make_expert_module
from cei.pb import cei_internal_pb2, cei_internal_pb2_grpc, cei_pb2
from cei.registry import ExpertRegistry, validate_descriptor
from cei.server.adapter_hub_servicer import AdapterHubServicer
from cei.types import ActivationBatch, DType, ExpertDescriptor, ExpertRef


def _desc(**overrides) -> ExpertDescriptor:
    base = dict(
        expert_ref=ExpertRef("m", 0, 0),
        version="1.0.0",
        dim_in=8,
        dim_out=8,
        dtype=DType.F32,
        fingerprint=np.ones(16) / 4.0,
        cost_flops=1,
        p50_latency_ms=1.0,
        capacity_qps=10.0,
        node_id="n1",
    )
    base.update(overrides)
    return ExpertDescriptor(**base)


# --- activation decoding -----------------------------------------------------


def test_activation_shape_mismatch_rejected():
    msg = cei_pb2.ActivationBatch(
        tensor=np.zeros(4, dtype=np.float64).tobytes(), shape=[16]
    )
    with pytest.raises(ValueError, match="shape_mismatch"):
        wire.activation_from_pb(msg)


def test_activation_negative_dim_rejected():
    msg = cei_pb2.ActivationBatch(
        tensor=np.zeros(4, dtype=np.float64).tobytes(), shape=[-1, 4]
    )
    with pytest.raises(ValueError, match="negative_dim"):
        wire.activation_from_pb(msg)


def test_activation_partial_element_rejected():
    msg = cei_pb2.ActivationBatch(tensor=b"\x00" * 12)  # not a multiple of 8
    with pytest.raises(ValueError, match="byte_length"):
        wire.activation_from_pb(msg)


def test_activation_nonfinite_rejected():
    msg = cei_pb2.ActivationBatch(
        tensor=np.array([1.0, float("inf")], dtype=np.float64).tobytes(), shape=[2]
    )
    with pytest.raises(ValueError, match="non_finite"):
        wire.activation_from_pb(msg)


def test_activation_excessive_rank_rejected():
    msg = cei_pb2.ActivationBatch(
        tensor=np.zeros(1, dtype=np.float64).tobytes(), shape=[1] * 12
    )
    with pytest.raises(ValueError, match="rank"):
        wire.activation_from_pb(msg)


def test_activation_valid_roundtrip():
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    act = wire.activation_from_pb(wire.activation_to_pb(ActivationBatch(tensor=arr)))
    assert np.allclose(act.tensor, arr)


# --- descriptor validation ---------------------------------------------------


def test_descriptor_bad_model_id():
    with pytest.raises(ValueError, match="model_id"):
        validate_descriptor(_desc(expert_ref=ExpertRef("bad model!", 0, 0)))


def test_descriptor_negative_ids():
    with pytest.raises(ValueError, match="ref_ids"):
        validate_descriptor(_desc(expert_ref=ExpertRef("m", -1, 0)))


def test_descriptor_zero_dims():
    with pytest.raises(ValueError, match="dims"):
        validate_descriptor(_desc(dim_in=0))


def test_descriptor_empty_fingerprint():
    with pytest.raises(ValueError, match="fingerprint"):
        validate_descriptor(_desc(fingerprint=np.array([])))


def test_descriptor_nonfinite_fingerprint():
    with pytest.raises(ValueError, match="fingerprint_nonfinite"):
        validate_descriptor(_desc(fingerprint=np.array([1.0, float("nan")])))


def test_descriptor_negative_capacity():
    with pytest.raises(ValueError, match="capacity"):
        validate_descriptor(_desc(capacity_qps=-1.0))


def test_descriptor_empty_version():
    with pytest.raises(ValueError, match="version"):
        validate_descriptor(_desc(version=""))


def test_registry_register_validates():
    reg = ExpertRegistry(allow_all=True, auto_promote=True)
    with pytest.raises(ValueError, match="INVALID_DESCRIPTOR"):
        reg.register(_desc(dim_in=-5))


def test_registry_heartbeat_ignores_bogus_metrics():
    reg = ExpertRegistry(allow_all=True, auto_promote=True)
    d = _desc()
    reg.register(d)
    key = d.expert_ref.key()
    reg.heartbeat("n1", [d.expert_ref], capacity_qps={key: float("nan")}, load_qps={key: -5.0})
    assert reg.capacity(d.expert_ref) == 10.0  # unchanged
    assert reg.load(d.expert_ref) == 0.0


def test_registry_nn_domain_filter_applies():
    reg = ExpertRegistry(allow_all=True, auto_promote=True)
    d_code = _desc(expert_ref=ExpertRef("m", 0, 0), domain_tags=["code"])
    d_math = _desc(expert_ref=ExpertRef("m", 0, 1), domain_tags=["math"])
    reg.register(d_code)
    reg.register(d_math)
    hits = reg.describe_nn(np.ones(16), k=10, domain_tags=["math"])
    assert [h[0].expert_ref.expert_id for h in hits] == [1]


def test_registry_nn_oversized_query_rejected():
    reg = ExpertRegistry(allow_all=True, auto_promote=True)
    with pytest.raises(ValueError, match="INVALID_QUERY"):
        reg.describe_nn(np.ones(10_000), k=5)


# --- node input dims ---------------------------------------------------------


def test_forward_rejects_wrong_input_dim():
    rng = np.random.default_rng(0)
    ref = ExpertRef("m", 0, 0)
    mod = make_expert_module(ref, 8, "code", rng)
    node = ExpertNode(node_id="n", model_id="m", acl_open=True)
    node.add_expert(mod, _desc())
    with pytest.raises(RuntimeError, match="INCOMPATIBLE_DIMS"):
        node.forward_expert(ref, ActivationBatch(tensor=np.ones(5)))


def test_forward_accepts_matching_dim():
    rng = np.random.default_rng(0)
    ref = ExpertRef("m", 0, 0)
    mod = make_expert_module(ref, 8, "code", rng)
    node = ExpertNode(node_id="n", model_id="m", acl_open=True)
    node.add_expert(mod, _desc())
    out, _ = node.forward_expert(ref, ActivationBatch(tensor=np.ones(8)))
    assert out.tensor.shape == (8,)


# --- adapter hub upload validation -------------------------------------------


@pytest.fixture
def lab_hub(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CEI_SECURITY_PROFILE", "lab")
    monkeypatch.setenv("CEI_REQUIRE_ADAPTER_DIGEST", "0")
    security.reset_config_cache()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = AdapterHubServicer()
    cei_internal_pb2_grpc.add_AdapterHubServicer_to_server(servicer, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = cei_internal_pb2_grpc.AdapterHubStub(channel)
    yield stub
    channel.close()
    server.stop(0)
    security.reset_config_cache()


def _blob(w_in: np.ndarray, w_out: np.ndarray, **overrides) -> cei_internal_pb2.AdapterBlob:
    base = dict(
        adapter_id="a1",
        dim_in_host=w_in.shape[0] if w_in.ndim == 2 else 0,
        dim_in_remote=w_in.shape[1] if w_in.ndim == 2 else 0,
        dim_out_remote=w_out.shape[0] if w_out.ndim == 2 else 0,
        dim_out_host=w_out.shape[1] if w_out.ndim == 2 else 0,
        w_in=w_in.astype(np.float64).tobytes(),
        w_out=w_out.astype(np.float64).tobytes(),
        w_in_shape=list(w_in.shape),
        w_out_shape=list(w_out.shape),
    )
    base.update(overrides)
    return cei_internal_pb2.AdapterBlob(**base)


def test_adapter_shape_mismatch_rejected(lab_hub):
    w = np.eye(4)
    blob = _blob(w, w, w_in_shape=[4, 8])  # lies about shape
    resp = lab_hub.UpsertAdapter(
        cei_internal_pb2.UpsertAdapterRequest(meta=wire.new_meta("dev"), adapter=blob)
    )
    assert resp.ok is False
    assert "MALFORMED_ADAPTER" in resp.error_code


def test_adapter_declared_dims_must_match(lab_hub):
    w = np.eye(4)
    blob = _blob(w, w, dim_in_host=8)  # declared dims disagree with matrix
    resp = lab_hub.UpsertAdapter(
        cei_internal_pb2.UpsertAdapterRequest(meta=wire.new_meta("dev"), adapter=blob)
    )
    assert resp.ok is False


def test_adapter_nonfinite_rejected(lab_hub):
    w = np.eye(4)
    bad = w.copy()
    bad[0, 0] = float("inf")
    blob = _blob(bad, w)
    resp = lab_hub.UpsertAdapter(
        cei_internal_pb2.UpsertAdapterRequest(meta=wire.new_meta("dev"), adapter=blob)
    )
    assert resp.ok is False
    assert "non_finite" in resp.error_code


def test_adapter_valid_upload_ok(lab_hub):
    w = np.eye(4)
    blob = _blob(w, w)
    resp = lab_hub.UpsertAdapter(
        cei_internal_pb2.UpsertAdapterRequest(meta=wire.new_meta("dev"), adapter=blob)
    )
    assert resp.ok is True


def test_adapter_digest_required_in_secure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CEI_SECURITY_PROFILE", "secure")
    monkeypatch.setenv("CEI_ADAPTER_WRITERS", "dev")
    monkeypatch.delenv("CEI_REQUIRE_ADAPTER_DIGEST", raising=False)
    security.reset_config_cache()
    try:
        servicer = AdapterHubServicer()
        w = np.eye(4)
        blob = _blob(w, w)  # no content_digest
        resp = servicer.UpsertAdapter(
            cei_internal_pb2.UpsertAdapterRequest(meta=wire.new_meta("dev"), adapter=blob),
            context=None,
        )
        assert resp.ok is False
        assert resp.error_code == "DIGEST_REQUIRED"
        blob2 = _blob(w, w, content_digest=security.adapter_digest(
            w.astype(np.float64).tobytes(), w.astype(np.float64).tobytes()
        ))
        resp2 = servicer.UpsertAdapter(
            cei_internal_pb2.UpsertAdapterRequest(meta=wire.new_meta("dev"), adapter=blob2),
            context=None,
        )
        assert resp2.ok is True
    finally:
        security.reset_config_cache()
