"""TLS / mTLS transport tests using ephemeral self-signed certificates."""

from __future__ import annotations

import shutil
import subprocess
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import pytest

from cei import security, tlsutil, wire
from cei.pb import cei_pb2, cei_pb2_grpc
from cei.server.registry_servicer import RegistryServicer

openssl = shutil.which("openssl")
pytestmark = pytest.mark.skipif(openssl is None, reason="openssl not available")


@pytest.fixture(scope="module")
def certs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("cei-certs")
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(out / "server.key"),
            "-out",
            str(out / "server.crt"),
            "-days",
            "1",
            "-subj",
            "/CN=cei.local",
            "-addext",
            "subjectAltName=DNS:cei.local,DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    shutil.copy(out / "server.crt", out / "ca.crt")
    return {"cert": out / "server.crt", "key": out / "server.key", "ca": out / "ca.crt"}


@pytest.fixture
def tls_env(certs, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CEI_TLS_CERT", str(certs["cert"]))
    monkeypatch.setenv("CEI_TLS_KEY", str(certs["key"]))
    monkeypatch.setenv("CEI_TLS_CA", str(certs["ca"]))
    monkeypatch.setenv("CEI_TLS_SERVER_NAME", "cei.local")
    security.reset_config_cache()
    yield certs
    security.reset_config_cache()


def _tls_registry_server() -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    cei_pb2_grpc.add_ExpertRegistryServicer_to_server(RegistryServicer(), server)
    creds = tlsutil.server_credentials()
    assert creds is not None
    port = server.add_secure_port("localhost:0", creds)
    server.start()
    return server, port


def test_mtls_roundtrip(tls_env):
    server, port = _tls_registry_server()
    try:
        channel = tlsutil.make_channel(f"localhost:{port}")
        stub = cei_pb2_grpc.ExpertRegistryStub(channel)
        fp = list(np.ones(16) / 4.0)
        resp = stub.RegisterExpert(
            cei_pb2.RegisterExpertRequest(
                meta=wire.new_meta("node-code"),
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
                promote=True,
            ),
            timeout=10,
        )
        assert resp.ok is True
        channel.close()
    finally:
        server.stop(0)


def test_plaintext_client_rejected_by_tls_server(tls_env):
    server, port = _tls_registry_server()
    try:
        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = cei_pb2_grpc.ExpertRegistryStub(channel)
        with pytest.raises(grpc.RpcError):
            stub.DescribeExperts(
                cei_pb2.DescribeExpertsRequest(
                    meta=wire.new_meta(),
                    explicit=cei_pb2.ExplicitRefs(expert_refs=[]),
                ),
                timeout=5,
            )
        channel.close()
    finally:
        server.stop(0)


def test_mtls_peer_identity_resolved(tls_env):
    """Server extracts principal from the client certificate CN."""
    server, port = _tls_registry_server()
    captured: dict[str, str | None] = {}

    class Probe(cei_pb2_grpc.ExpertRegistryServicer):
        def DescribeExperts(self, request, context):
            captured["peer"] = security.peer_identity_from_context(context)
            return cei_pb2.DescribeExpertsResponse()

    probe_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    cei_pb2_grpc.add_ExpertRegistryServicer_to_server(Probe(), probe_server)
    creds = tlsutil.server_credentials()
    probe_port = probe_server.add_secure_port("localhost:0", creds)
    probe_server.start()
    try:
        channel = tlsutil.make_channel(f"localhost:{probe_port}")
        stub = cei_pb2_grpc.ExpertRegistryStub(channel)
        stub.DescribeExperts(
            cei_pb2.DescribeExpertsRequest(
                meta=wire.new_meta(),
                explicit=cei_pb2.ExplicitRefs(expert_refs=[]),
            ),
            timeout=10,
        )
        # mTLS client presented server.crt (CN=cei.local); identity resolved.
        assert captured["peer"] in {"cei.local", "localhost"}
        channel.close()
    finally:
        probe_server.stop(0)
        server.stop(0)


def test_require_tls_asserts_without_certs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CEI_REQUIRE_TLS", "1")
    monkeypatch.delenv("CEI_TLS_CERT", raising=False)
    monkeypatch.delenv("CEI_TLS_KEY", raising=False)
    security.reset_config_cache()
    with pytest.raises(RuntimeError, match="CEI_REQUIRE_TLS"):
        security.assert_transport_ok()
    security.reset_config_cache()
