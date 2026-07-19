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
import struct
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

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
    # Fleet-wide request-auth secret. When set, RequestMeta.auth_token is an
    # HMAC-SHA256 over (principal, request_id, ts) proving principal identity
    # on plaintext channels. mTLS peer identity always takes precedence.
    auth_secret: str = ""
    require_auth_token: bool = False
    auth_max_skew_ms: int = 120_000
    # Require adapter uploads to carry a content digest (integrity pinning).
    require_adapter_digest: bool = True

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
        auth_secret = os.environ.get("CEI_AUTH_SECRET", "")
        require_auth = _env_bool(
            "CEI_REQUIRE_AUTH_TOKEN",
            default=(not lab and bool(auth_secret)),
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
            auth_secret=auth_secret,
            require_auth_token=require_auth,
            auth_max_skew_ms=int(os.environ.get("CEI_AUTH_MAX_SKEW_MS", "120000")),
            require_adapter_digest=_env_bool("CEI_REQUIRE_ADAPTER_DIGEST", default=not lab),
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
            except Exception:  # noqa: S112 — malformed cert field, try next
                continue
    sans = auth.get("x509_subject_alternative_name")
    if sans:
        for item in sans:
            try:
                text = item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
            except Exception:  # noqa: S112 — malformed SAN entry, try next
                continue
            if text.startswith("URI:spiffe://") or text.startswith("spiffe://"):
                return text.split("URI:")[-1]
            if text.startswith("DNS:"):
                return text[4:]
    return None


def sign_meta(
    principal_id: str,
    request_id: str,
    ts_unix_ms: int,
    secret: str | None = None,
) -> str:
    """HMAC-SHA256 request-auth token binding principal, request id, and time."""
    key = (secret if secret is not None else get_config().auth_secret).encode("utf-8")
    if not key:
        return ""
    msg = f"v1|{principal_id}|{request_id}|{ts_unix_ms}".encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_meta(meta: Any, secret: str | None = None, now_ms: int | None = None) -> bool:
    """Verify RequestMeta.auth_token integrity and freshness."""
    cfg = get_config()
    key = secret if secret is not None else cfg.auth_secret
    if not key or meta is None:
        return False
    token = getattr(meta, "auth_token", "") or ""
    if not token:
        return False
    ts = int(getattr(meta, "ts_unix_ms", 0) or 0)
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if abs(now - ts) > cfg.auth_max_skew_ms:
        return False
    expected = sign_meta(
        getattr(meta, "principal_id", "") or "",
        getattr(meta, "request_id", "") or "",
        ts,
        secret=key,
    )
    return bool(expected) and hmac.compare_digest(expected, token)


def resolve_principal(context: Any, meta: Any) -> str | None:
    """Resolve the caller principal, strongest evidence first.

    1. mTLS peer certificate identity (CN / SPIFFE / DNS SAN).
    2. HMAC-authenticated RequestMeta (when CEI_AUTH_SECRET is configured).
    3. Unauthenticated RequestMeta.principal_id (only when the profile trusts
       it and auth tokens are not required).

    ``meta`` may be a RequestMeta protobuf or a bare principal string
    (in-process callers).
    """
    cfg = get_config()
    peer = peer_identity_from_context(context)
    if peer:
        return peer

    meta_principal: str | None
    if meta is None:
        meta_principal = None
    elif isinstance(meta, str):
        meta_principal = meta
    else:
        meta_principal = getattr(meta, "principal_id", None)

    if cfg.auth_secret and not isinstance(meta, (str, type(None))):
        if verify_meta(meta):
            return (meta_principal or "").strip() or None
        if cfg.require_auth_token:
            return None
    elif cfg.require_auth_token:
        # Tokens required but no verifiable meta available.
        return None

    if not cfg.trust_meta_principal:
        return None
    if meta_principal and meta_principal.strip():
        return meta_principal.strip()
    return None


class ReplayCache:
    """Thread-safe seen-id cache with TTL eviction (at-most-once semantics)."""

    def __init__(self, ttl_ms: float = 300_000.0, max_entries: int = 100_000) -> None:
        self.ttl_ms = ttl_ms
        self.max_entries = max_entries
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def check_and_add(self, key: str, now_ms: float | None = None) -> bool:
        """Return True if the key is fresh (first sighting); False on replay."""
        if not key:
            return False
        now = time.time() * 1000.0 if now_ms is None else now_ms
        with self._lock:
            expired = [k for k, t in self._seen.items() if now - t > self.ttl_ms]
            for k in expired:
                del self._seen[k]
            if key in self._seen:
                return False
            if len(self._seen) >= self.max_entries:
                # Shed oldest half to bound memory under flood.
                for k, _ in sorted(self._seen.items(), key=lambda kv: kv[1])[
                    : self.max_entries // 2
                ]:
                    del self._seen[k]
            self._seen[key] = now
            return True


def audit(event: str, **fields: Any) -> None:
    payload = {"ts_ms": int(time.time() * 1000), "event": event, **fields}
    _AUDIT_LOG.info(json.dumps(payload, sort_keys=True, default=str))


def adapter_digest(w_in: bytes, w_out: bytes) -> str:
    h = hashlib.sha256()
    h.update(w_in)
    h.update(w_out)
    return h.hexdigest()


def _f32(x: float) -> float:
    """Quantize to IEEE float32.

    Proto `float` fields are 32-bit: the signer sees the original Python
    float64 while the verifier sees the wire-truncated value. Quantizing both
    sides before canonical encoding keeps the HMAC stable across the wire.
    """
    return struct.unpack("<f", struct.pack("<f", x))[0]


def outcome_attestation_payload(
    *,
    plan_id: str,
    host_model_id: str,
    reward: float,
    utility: float,
    latency_ms: float,
    tokens: int,
    request_id: str = "",
) -> bytes:
    # Canonical, stable encoding for HMAC: float32-quantized, hex-exact floats.
    # request_id binds the signature to one report, blocking replay.
    body = (
        f"v2|{plan_id}|{host_model_id}|{_f32(reward).hex()}|{_f32(utility).hex()}"
        f"|{_f32(latency_ms).hex()}|{tokens}|{request_id}"
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
    request_id: str = "",
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
        request_id=request_id,
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
    request_id: str = "",
    secret: str | None = None,
) -> bool:
    expected = sign_outcome(
        plan_id=plan_id,
        host_model_id=host_model_id,
        reward=reward,
        utility=utility,
        latency_ms=latency_ms,
        tokens=tokens,
        request_id=request_id,
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
