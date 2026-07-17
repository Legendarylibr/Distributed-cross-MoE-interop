"""CEI security controls: identity, ACLs, attestation, audit, digests.

Defaults are deny-by-default for distributed (gRPC) roles. In-process
simulation may opt into open ACLs explicitly.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


_AUDIT_LOG = logging.getLogger("cei.audit")
if not _AUDIT_LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _AUDIT_LOG.addHandler(_handler)
    _AUDIT_LOG.setLevel(logging.INFO)
    _AUDIT_LOG.propagate = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {p.strip() for p in raw.split(",") if p.strip()}


@dataclass
class SecurityConfig:
    """Resolved security posture for a process."""

    profile: str = "secure"  # secure | lab
    require_tls: bool = False
    registry_allow_all: bool = False
    auto_promote: bool = False
    node_acl_open: bool = False
    node_acl_allow: set[str] = field(default_factory=set)
    registry_publishers: set[str] = field(default_factory=set)
    registry_consumers: set[str] = field(default_factory=set)
    adapter_writers: set[str] = field(default_factory=set)
    priority_admins: set[str] = field(default_factory=set)
    outcome_hmac_secret: str = ""
    require_outcome_attestation: bool = True
    trust_meta_principal: bool = True  # False when peer cert identity is required

    @classmethod
    def from_env(cls) -> SecurityConfig:
        profile = os.environ.get("CEI_SECURITY_PROFILE", "secure").strip().lower()
        if profile not in {"secure", "lab"}:
            profile = "secure"
        lab = profile == "lab"
        secret = os.environ.get("CEI_OUTCOME_HMAC_SECRET", "")
        if lab and not secret:
            secret = "cei-lab-insecure-hmac"
        # Attestation required only when explicitly enabled or secret is set in secure mode.
        require_attest = _env_bool(
            "CEI_REQUIRE_OUTCOME_ATTESTATION",
            default=(not lab and bool(os.environ.get("CEI_OUTCOME_HMAC_SECRET"))),
        )
        cfg = cls(
            profile=profile,
            require_tls=_env_bool("CEI_REQUIRE_TLS", default=False),
            registry_allow_all=_env_bool("CEI_REGISTRY_ALLOW_ALL", default=lab),
            auto_promote=_env_bool("CEI_AUTO_PROMOTE", default=lab),
            node_acl_open=_env_bool("CEI_NODE_ACL_OPEN", default=lab),
            node_acl_allow=_env_csv("CEI_NODE_ACL_ALLOW"),
            registry_publishers=_env_csv("CEI_REGISTRY_PUBLISHERS"),
            registry_consumers=_env_csv("CEI_REGISTRY_CONSUMERS"),
            adapter_writers=_env_csv("CEI_ADAPTER_WRITERS"),
            priority_admins=_env_csv("CEI_PRIORITY_ADMINS"),
            outcome_hmac_secret=secret,
            require_outcome_attestation=require_attest,
            trust_meta_principal=_env_bool("CEI_TRUST_META_PRINCIPAL", default=True),
        )
        if lab and not cfg.node_acl_allow:
            cfg.node_acl_open = True
        return cfg


_CFG: SecurityConfig | None = None
_CFG_LOCK = threading.Lock()


def get_config() -> SecurityConfig:
    global _CFG
    with _CFG_LOCK:
        if _CFG is None:
            _CFG = SecurityConfig.from_env()
        return _CFG


def reset_config_cache() -> None:
    """Test helper: reload config from environment."""
    global _CFG
    with _CFG_LOCK:
        _CFG = None


def assert_transport_ok() -> None:
    cfg = get_config()
    if not cfg.require_tls:
        return
    if not (os.environ.get("CEI_TLS_CERT") and os.environ.get("CEI_TLS_KEY")):
        raise RuntimeError("CEI_REQUIRE_TLS=1 but CEI_TLS_CERT/KEY not set")


def peer_identity_from_context(context: Any) -> str | None:
    """Extract principal from mTLS peer cert CN / URI SAN when available."""
    if context is None:
        return None
    try:
        auth = context.auth_context()
    except Exception:
        return None
    if not auth:
        return None
    # grpc auth_context keys vary; common: "x509_common_name", "x509_subject_alternative_name"
    for key in ("x509_common_name", "transport_security_type"):
        vals = auth.get(key)
        if key == "transport_security_type":
            continue
        if vals:
            try:
                raw = vals[0]
                return raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            except Exception:
                continue
    sans = auth.get("x509_subject_alternative_name")
    if sans:
        for item in sans:
            try:
                text = item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
            except Exception:
                continue
            if text.startswith("URI:spiffe://") or text.startswith("spiffe://"):
                return text.split("URI:")[-1]
            if text.startswith("DNS:"):
                return text[4:]
    return None


def resolve_principal(context: Any, meta_principal: str | None) -> str | None:
    """Prefer peer cert identity; optionally fall back to RequestMeta.principal_id."""
    cfg = get_config()
    peer = peer_identity_from_context(context)
    if peer:
        return peer
    if not cfg.trust_meta_principal:
        return None
    if meta_principal and meta_principal.strip():
        return meta_principal.strip()
    return None


def audit(event: str, **fields: Any) -> None:
    payload = {"ts_ms": int(time.time() * 1000), "event": event, **fields}
    _AUDIT_LOG.info(json.dumps(payload, sort_keys=True, default=str))


def adapter_digest(w_in: bytes, w_out: bytes) -> str:
    h = hashlib.sha256()
    h.update(w_in)
    h.update(w_out)
    return h.hexdigest()


def outcome_attestation_payload(
    *,
    plan_id: str,
    host_model_id: str,
    reward: float,
    utility: float,
    latency_ms: float,
    tokens: int,
) -> bytes:
    # Canonical, stable encoding for HMAC (no floats as binary — fixed decimals).
    body = (
        f"{plan_id}|{host_model_id}|{reward:.6f}|{utility:.6f}|{latency_ms:.6f}|{tokens}"
    )
    return body.encode("utf-8")


def sign_outcome(
    *,
    plan_id: str,
    host_model_id: str,
    reward: float,
    utility: float,
    latency_ms: float,
    tokens: int,
    secret: str | None = None,
) -> str:
    cfg = get_config()
    key = (secret if secret is not None else cfg.outcome_hmac_secret).encode("utf-8")
    if not key:
        return ""
    msg = outcome_attestation_payload(
        plan_id=plan_id,
        host_model_id=host_model_id,
        reward=reward,
        utility=utility,
        latency_ms=latency_ms,
        tokens=tokens,
    )
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_outcome_attestation(
    *,
    plan_id: str,
    host_model_id: str,
    reward: float,
    utility: float,
    latency_ms: float,
    tokens: int,
    attestation: str,
    secret: str | None = None,
) -> bool:
    expected = sign_outcome(
        plan_id=plan_id,
        host_model_id=host_model_id,
        reward=reward,
        utility=utility,
        latency_ms=latency_ms,
        tokens=tokens,
        secret=secret,
    )
    if not expected:
        return False
    if not attestation:
        return False
    return hmac.compare_digest(expected, attestation)


def can_publish(principal: str | None, publishers: Iterable[str] | None = None) -> bool:
    cfg = get_config()
    allow = set(publishers) if publishers is not None else set(cfg.registry_publishers)
    if principal is None:
        return False
    if not allow:
        # Unconfigured allowlist: lab is open to any principal; secure denies.
        return cfg.profile == "lab"
    return principal in allow


def can_write_adapter(principal: str | None) -> bool:
    cfg = get_config()
    if cfg.profile == "lab" and not cfg.adapter_writers:
        return True
    if principal is None:
        return False
    if not cfg.adapter_writers:
        return False
    return principal in cfg.adapter_writers


def can_use_priority(principal: str | None, priority: int) -> bool:
    if priority < 10:
        return True
    cfg = get_config()
    if principal is None:
        return False
    return principal in cfg.priority_admins
