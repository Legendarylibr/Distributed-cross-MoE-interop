"""TLS helpers for CEI gRPC (dev: optional self-signed; prod: provide certs)."""

from __future__ import annotations

import os
from pathlib import Path

import grpc


def tls_enabled() -> bool:
    return bool(os.environ.get("CEI_TLS_CERT") and os.environ.get("CEI_TLS_KEY"))


def server_credentials() -> grpc.ServerCredentials | None:
    cert = os.environ.get("CEI_TLS_CERT")
    key = os.environ.get("CEI_TLS_KEY")
    if not cert or not key:
        return None
    cert_path, key_path = Path(cert), Path(key)
    private_key = key_path.read_bytes()
    certificate_chain = cert_path.read_bytes()
    # Optional client CA for mTLS
    ca = os.environ.get("CEI_TLS_CA")
    if ca:
        root = Path(ca).read_bytes()
        return grpc.ssl_server_credentials(
            [(private_key, certificate_chain)],
            root_certificates=root,
            require_client_auth=True,
        )
    return grpc.ssl_server_credentials([(private_key, certificate_chain)])


def make_channel(addr: str) -> grpc.Channel:
    """Insecure unless CEI_TLS_CERT (+ optional CEI_TLS_CA) is set."""
    cert = os.environ.get("CEI_TLS_CERT")
    if not cert:
        return grpc.insecure_channel(addr)
    root = Path(os.environ.get("CEI_TLS_CA", cert)).read_bytes()
    # Client key/cert for mTLS when CA requires it
    key = os.environ.get("CEI_TLS_KEY")
    if key and os.environ.get("CEI_TLS_CA"):
        creds = grpc.ssl_channel_credentials(
            root_certificates=root,
            private_key=Path(key).read_bytes(),
            certificate_chain=Path(cert).read_bytes(),
        )
    else:
        creds = grpc.ssl_channel_credentials(root_certificates=root)
    # Override target name for self-signed / IP hosts
    opts = (("grpc.ssl_target_name_override", os.environ.get("CEI_TLS_SERVER_NAME", "cei.local")),)
    return grpc.secure_channel(addr, creds, options=opts)


def add_secure_or_insecure_port(server: grpc.Server, bind: str) -> None:
    creds = server_credentials()
    if creds is None:
        server.add_insecure_port(bind)
    else:
        server.add_secure_port(bind, creds)
