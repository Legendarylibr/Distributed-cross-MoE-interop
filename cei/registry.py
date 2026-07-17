"""Expert Registry: catalog, heartbeats, fingerprint nearest-neighbor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from cei.types import ExpertDescriptor, ExpertRef


@dataclass
class _Entry:
    descriptor: ExpertDescriptor
    last_heartbeat_ms: float
    load_qps: float = 0.0
    capacity_qps: float = 0.0
    promoted: bool = False
    routable: bool = False


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
        key = descriptor.expert_ref.key()
        existing = self._entries.get(key)
        if existing and not force:
            if _semver_tuple(descriptor.version) < _semver_tuple(existing.descriptor.version):
                raise ValueError(f"STALE_VERSION:{key}")
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
        self._entries[key] = _Entry(
            descriptor=descriptor,
            last_heartbeat_ms=now,
            load_qps=0.0,
            capacity_qps=descriptor.capacity_qps,
            promoted=do_promote,
            routable=False,
        )
        self._grant_on_register(key, principal)
        self.refresh_routability()

    def promote(self, refs: list[ExpertRef]) -> None:
        for ref in refs:
            entry = self._entries.get(ref.key())
            if entry is None:
                continue
            entry.promoted = True
        self.refresh_routability()

    def _grant_on_register(self, key: str, principal: str | None) -> None:
        if principal:
            self._acl.setdefault(principal, set()).add(key)
        for p in self.consumer_principals:
            self._acl.setdefault(p, set()).add(key)

    def deregister(self, refs: list[ExpertRef]) -> None:
        for ref in refs:
            self._entries.pop(ref.key(), None)

    def heartbeat(
        self,
        node_id: str,
        expert_refs: list[ExpertRef] | None,
        capacity_qps: dict[str, float] | None = None,
        load_qps: dict[str, float] | None = None,
    ) -> int:
        now = _now_ms()
        keys: list[str]
        if expert_refs:
            keys = [r.key() for r in expert_refs]
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
                entry.capacity_qps = capacity_qps[key]
            if load_qps and key in load_qps:
                entry.load_qps = load_qps[key]
        self.refresh_routability()
        return int(self.heartbeat_ttl_ms / 3)

    def refresh_routability(self) -> None:
        now = _now_ms()
        for entry in self._entries.values():
            fresh = now - entry.last_heartbeat_ms <= self.heartbeat_ttl_ms
            entry.routable = bool(entry.promoted and fresh)

    def get(self, ref: ExpertRef) -> ExpertDescriptor | None:
        entry = self._entries.get(ref.key())
        return entry.descriptor if entry else None

    def load(self, ref: ExpertRef) -> float:
        entry = self._entries.get(ref.key())
        return entry.load_qps if entry else 0.0

    def capacity(self, ref: ExpertRef) -> float:
        entry = self._entries.get(ref.key())
        return entry.capacity_qps if entry else 0.0

    def set_acl(self, principal: str, allowed_keys: set[str]) -> None:
        self._acl[principal] = set(allowed_keys)
        self.allow_all = False

    def grant(self, principal: str, keys: set[str]) -> None:
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
        self.refresh_routability()
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
        self.refresh_routability()
        q = np.asarray(fingerprint, dtype=np.float64).reshape(-1)
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
            if domain_tags:
                if not set(domain_tags) & set(d.domain_tags):
                    pass
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
