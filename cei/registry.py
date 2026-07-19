"""Expert Registry: catalog, heartbeats, fingerprint nearest-neighbor.

Thread-safe: all mutating and query paths take an internal RLock so the
registry can back a multi-worker gRPC servicer.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from cei.types import ExpertDescriptor, ExpertRef

_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
_MAX_FINGERPRINT = 4096
_MAX_DOMAIN_TAGS = 32


def validate_descriptor(d: ExpertDescriptor) -> None:
    """Validate an untrusted descriptor at the trust boundary.

    Raises ValueError with a stable ``INVALID_DESCRIPTOR:<field>`` code.
    """
    ref = d.expert_ref
    if not ref.model_id or not _ID_RE.match(ref.model_id):
        raise ValueError("INVALID_DESCRIPTOR:model_id")
    if ref.layer_id < 0 or ref.expert_id < 0:
        raise ValueError("INVALID_DESCRIPTOR:ref_ids")
    if not d.version or len(d.version) > 64:
        raise ValueError("INVALID_DESCRIPTOR:version")
    if d.dim_in <= 0 or d.dim_out <= 0 or d.dim_in > 1_000_000 or d.dim_out > 1_000_000:
        raise ValueError("INVALID_DESCRIPTOR:dims")
    fp = np.asarray(d.fingerprint, dtype=np.float64).reshape(-1)
    if fp.size == 0 or fp.size > _MAX_FINGERPRINT:
        raise ValueError("INVALID_DESCRIPTOR:fingerprint")
    if not np.all(np.isfinite(fp)):
        raise ValueError("INVALID_DESCRIPTOR:fingerprint_nonfinite")
    if d.capacity_qps < 0 or not np.isfinite(d.capacity_qps):
        raise ValueError("INVALID_DESCRIPTOR:capacity")
    if d.p50_latency_ms < 0 or not np.isfinite(d.p50_latency_ms):
        raise ValueError("INVALID_DESCRIPTOR:latency")
    if d.cost_flops < 0:
        raise ValueError("INVALID_DESCRIPTOR:cost")
    if len(d.domain_tags) > _MAX_DOMAIN_TAGS:
        raise ValueError("INVALID_DESCRIPTOR:domain_tags")
    if d.node_id and not _ID_RE.match(d.node_id):
        raise ValueError("INVALID_DESCRIPTOR:node_id")


@dataclass
class _Entry:
    descriptor: ExpertDescriptor
    last_heartbeat_ms: float
    load_qps: float = 0.0
    capacity_qps: float = 0.0
    promoted: bool = False
    routable: bool = False
    owner_principal: str | None = None


@dataclass
class ExpertRegistry:
    heartbeat_ttl_ms: float = 15_000.0
    max_experts_per_model: int = 256
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _acl: dict[str, set[str]] = field(default_factory=dict)
    # Deny-by-default: principal must be ACL-granted unless allow_all.
    allow_all: bool = False
    # Principals that receive read ACL on every newly registered expert.
    consumer_principals: set[str] = field(default_factory=set)
    # If True, new registrations are promoted (routable after heartbeat).
    auto_promote: bool = False
    # If True, mutations must come from the registering owner principal.
    enforce_ownership: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def _model_expert_count(self, model_id: str) -> int:
        return sum(1 for e in self._entries.values() if e.descriptor.expert_ref.model_id == model_id)

    def register(
        self,
        descriptor: ExpertDescriptor,
        force: bool = False,
        *,
        promote: bool | None = None,
        principal: str | None = None,
    ) -> None:
        validate_descriptor(descriptor)
        key = descriptor.expert_ref.key()
        with self._lock:
            existing = self._entries.get(key)
            if existing and not force:
                if _semver_tuple(descriptor.version) < _semver_tuple(existing.descriptor.version):
                    raise ValueError(f"STALE_VERSION:{key}")
            if (
                existing is not None
                and self.enforce_ownership
                and existing.owner_principal
                and principal != existing.owner_principal
            ):
                raise PermissionError(f"NOT_OWNER:{key}")
            if existing is None:
                n = self._model_expert_count(descriptor.expert_ref.model_id)
                if n >= self.max_experts_per_model:
                    raise ValueError(f"QUOTA_EXCEEDED:{descriptor.expert_ref.model_id}")
            fp = descriptor.normalized_fingerprint()
            descriptor.fingerprint = fp
            now = _now_ms()
            do_promote = self.auto_promote if promote is None else bool(promote)
            if existing is not None and promote is None:
                # Preserve promotion status on version refresh unless explicitly set.
                do_promote = existing.promoted or self.auto_promote
            owner = principal or (existing.owner_principal if existing else None)
            self._entries[key] = _Entry(
                descriptor=descriptor,
                last_heartbeat_ms=now,
                load_qps=0.0,
                capacity_qps=descriptor.capacity_qps,
                promoted=do_promote,
                routable=False,
                owner_principal=owner,
            )
            self._grant_on_register(key, principal)
            self._refresh_routability_locked()

    def promote(self, refs: list[ExpertRef]) -> None:
        with self._lock:
            for ref in refs:
                entry = self._entries.get(ref.key())
                if entry is None:
                    continue
                entry.promoted = True
            self._refresh_routability_locked()

    def _grant_on_register(self, key: str, principal: str | None) -> None:
        if principal:
            self._acl.setdefault(principal, set()).add(key)
        for p in self.consumer_principals:
            self._acl.setdefault(p, set()).add(key)

    def deregister(self, refs: list[ExpertRef], principal: str | None = None) -> None:
        with self._lock:
            for ref in refs:
                entry = self._entries.get(ref.key())
                if entry is None:
                    continue
                if (
                    self.enforce_ownership
                    and entry.owner_principal
                    and principal != entry.owner_principal
                ):
                    raise PermissionError(f"NOT_OWNER:{ref.key()}")
                self._entries.pop(ref.key(), None)

    def heartbeat(
        self,
        node_id: str,
        expert_refs: list[ExpertRef] | None,
        capacity_qps: dict[str, float] | None = None,
        load_qps: dict[str, float] | None = None,
    ) -> int:
        now = _now_ms()
        with self._lock:
            keys: list[str]
            if expert_refs:
                # Only refresh entries the reporting node actually owns.
                keys = [
                    r.key()
                    for r in expert_refs
                    if (e := self._entries.get(r.key())) is not None
                    and (e.descriptor.node_id is None or e.descriptor.node_id == node_id)
                ]
            else:
                keys = [
                    k
                    for k, e in self._entries.items()
                    if e.descriptor.node_id == node_id
                ]
            for key in keys:
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.last_heartbeat_ms = now
                if capacity_qps and key in capacity_qps:
                    cap = float(capacity_qps[key])
                    if np.isfinite(cap) and cap >= 0:
                        entry.capacity_qps = cap
                if load_qps and key in load_qps:
                    load = float(load_qps[key])
                    if np.isfinite(load) and load >= 0:
                        entry.load_qps = load
            self._refresh_routability_locked()
        return int(self.heartbeat_ttl_ms / 3)

    def refresh_routability(self) -> None:
        with self._lock:
            self._refresh_routability_locked()

    def _refresh_routability_locked(self) -> None:
        now = _now_ms()
        for entry in self._entries.values():
            fresh = now - entry.last_heartbeat_ms <= self.heartbeat_ttl_ms
            entry.routable = bool(entry.promoted and fresh)

    def get(self, ref: ExpertRef) -> ExpertDescriptor | None:
        with self._lock:
            entry = self._entries.get(ref.key())
            return entry.descriptor if entry else None

    def owner(self, ref: ExpertRef) -> str | None:
        with self._lock:
            entry = self._entries.get(ref.key())
            return entry.owner_principal if entry else None

    def load(self, ref: ExpertRef) -> float:
        with self._lock:
            entry = self._entries.get(ref.key())
            return entry.load_qps if entry else 0.0

    def capacity(self, ref: ExpertRef) -> float:
        with self._lock:
            entry = self._entries.get(ref.key())
            return entry.capacity_qps if entry else 0.0

    def set_acl(self, principal: str, allowed_keys: set[str]) -> None:
        with self._lock:
            self._acl[principal] = set(allowed_keys)
            self.allow_all = False

    def grant(self, principal: str, keys: set[str]) -> None:
        with self._lock:
            self._acl.setdefault(principal, set()).update(keys)
            self.allow_all = False

    def _visible(self, principal: str | None, key: str) -> bool:
        if self.allow_all:
            return True
        if principal is None:
            return False
        allowed = self._acl.get(principal, set())
        return key in allowed

    def describe_explicit(
        self, refs: list[ExpertRef], principal: str | None = None
    ) -> list[tuple[ExpertDescriptor, bool]]:
        with self._lock:
            self._refresh_routability_locked()
            out: list[tuple[ExpertDescriptor, bool]] = []
            for ref in refs:
                key = ref.key()
                if not self._visible(principal, key):
                    continue
                entry = self._entries.get(key)
                if entry is None:
                    continue
                out.append((entry.descriptor, entry.routable))
            return out

    def describe_nn(
        self,
        fingerprint: np.ndarray,
        k: int = 32,
        host_dim_in: int | None = None,
        host_dim_out: int | None = None,
        domain_tags: list[str] | None = None,
        principal: str | None = None,
        exclude_model: str | None = None,
    ) -> list[tuple[ExpertDescriptor, float, bool]]:
        """Return (descriptor, cosine_sim, routable) sorted by similarity desc."""
        q = np.asarray(fingerprint, dtype=np.float64).reshape(-1)
        if q.size > _MAX_FINGERPRINT:
            raise ValueError("INVALID_QUERY:fingerprint")
        with self._lock:
            self._refresh_routability_locked()
            scored: list[tuple[ExpertDescriptor, float, bool]] = []
            for key, entry in self._entries.items():
                if not self._visible(principal, key):
                    continue
                d = entry.descriptor
                if exclude_model and d.expert_ref.model_id == exclude_model:
                    continue
                if host_dim_in is not None and d.adapter_id is None and d.dim_in != host_dim_in:
                    continue
                if host_dim_out is not None and d.adapter_id is None and d.dim_out != host_dim_out:
                    continue
                if domain_tags and not set(domain_tags) & set(d.domain_tags):
                    continue
                fp = d.normalized_fingerprint().reshape(-1)
                # Align dims for cosine (pad/trim)
                n = max(q.size, fp.size)
                qq = np.zeros(n)
                ff = np.zeros(n)
                qq[: q.size] = q
                ff[: fp.size] = fp
                qn = np.linalg.norm(qq)
                fn = np.linalg.norm(ff)
                if qn > 1e-12:
                    qq = qq / qn
                if fn > 1e-12:
                    ff = ff / fn
                sim = float(np.dot(qq, ff))
                scored.append((d, sim, entry.routable))
            scored.sort(key=lambda t: t[1], reverse=True)
            return scored[:k]


def _now_ms() -> float:
    return time.time() * 1000.0


def _semver_tuple(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts)
