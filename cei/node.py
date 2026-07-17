"""Local MoE expert node: owns experts, leases, ForwardExpert."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from cei.adapters import AdapterHub
from cei.types import (
    ActivationBatch,
    DType,
    ExpertDescriptor,
    ExpertRef,
    Lease,
)


@dataclass
class ExpertModule:
    ref: ExpertRef
    weight: np.ndarray  # (dim_out, dim_in) applied as h @ W.T for (..., dim_in)
    bias: np.ndarray
    domain: str
    specialty_vec: np.ndarray  # for utility scoring in simulator

    def forward(self, h: np.ndarray) -> np.ndarray:
        # h: (batch, dim_in) or (dim_in,)
        return h @ self.weight.T + self.bias


@dataclass
class ExpertNode:
    node_id: str
    model_id: str
    experts: dict[str, ExpertModule] = field(default_factory=dict)
    descriptors: dict[str, ExpertDescriptor] = field(default_factory=dict)
    adapter_hub: AdapterHub | None = None
    leases: dict[str, Lease] = field(default_factory=dict)
    seen_request_ids: dict[str, float] = field(default_factory=dict)
    request_id_ttl_ms: float = 60_000.0
    training_profile: bool = False
    base_latency_ms: float = 2.0
    remote_extra_latency_ms: float = 5.0
    # Deny-by-default: empty acl_allow denies everyone unless acl_open.
    acl_open: bool = False
    acl_allow: set[str] = field(default_factory=set)
    priority_admins: set[str] = field(default_factory=set)
    _load_tokens: float = 0.0

    def add_expert(self, module: ExpertModule, descriptor: ExpertDescriptor) -> None:
        key = module.ref.key()
        self.experts[key] = module
        descriptor.node_id = self.node_id
        self.descriptors[key] = descriptor

    def _check_acl(self, principal: str | None) -> None:
        if self.acl_open:
            return
        if principal is None or principal not in self.acl_allow:
            raise PermissionError("ACL_DENIED")

    def get_capacity_snapshot(self) -> dict[str, float]:
        return {k: d.capacity_qps for k, d in self.descriptors.items()}

    def get_load_snapshot(self) -> dict[str, float]:
        # Approximate per-expert share of recent load tokens
        n = max(len(self.experts), 1)
        share = float(self._load_tokens) / n
        return {k: share for k in self.experts}

    def lease_capacity(
        self,
        expert_ref: ExpertRef,
        tokens_or_qps: float,
        ttl_ms: int,
        principal: str | None = None,
        priority: int = 0,
    ) -> Lease:
        self._check_acl(principal)
        key = expert_ref.key()
        if key not in self.experts:
            raise KeyError("NOT_ROUTABLE")
        desc = self.descriptors[key]
        # Priority>=10 bypass only for configured admin principals.
        bypass = priority >= 10 and principal is not None and principal in self.priority_admins
        if self._load_tokens > desc.capacity_qps * 10 and not bypass:
            raise RuntimeError("CAPACITY_EXHAUSTED")
        lease = Lease(
            lease_id=str(uuid.uuid4()),
            expert_ref=expert_ref,
            deadline_ms=_now_ms() + ttl_ms,
            granted_qps=min(tokens_or_qps, desc.capacity_qps),
        )
        self.leases[lease.lease_id] = lease
        return lease

    def release_capacity(self, lease_id: str) -> None:
        self.leases.pop(lease_id, None)

    def forward_expert(
        self,
        expert_ref: ExpertRef,
        activation: ActivationBatch,
        lease_id: str | None = None,
        adapter_id: str | None = None,
        request_id: str | None = None,
        principal: str | None = None,
        require_lease: bool = False,
    ) -> tuple[ActivationBatch, float]:
        self._check_acl(principal)
        key = expert_ref.key()
        if key not in self.experts:
            raise KeyError("NOT_ROUTABLE")
        if activation.grad_required and not self.training_profile:
            raise RuntimeError("PROFILE_DISABLED")

        now = _now_ms()
        if request_id:
            # purge old
            expired = [r for r, t in self.seen_request_ids.items() if now - t > self.request_id_ttl_ms]
            for r in expired:
                del self.seen_request_ids[r]
            if request_id in self.seen_request_ids:
                # idempotent replay: recompute (no double load charge)
                pass
            else:
                self.seen_request_ids[request_id] = now
                self._load_tokens += float(np.prod(activation.tensor.shape[:-1]))

        if require_lease:
            if not lease_id or lease_id not in self.leases:
                raise RuntimeError("CAPACITY_EXHAUSTED")
            lease = self.leases[lease_id]
            if lease.deadline_ms < now:
                raise RuntimeError("CAPACITY_EXHAUSTED")

        h = activation.tensor
        if adapter_id and self.adapter_hub:
            adapter = self.adapter_hub.get(adapter_id)
            if adapter is None:
                raise RuntimeError("INCOMPATIBLE_DIMS")
            h = adapter.forward_in(h)

        y = self.experts[key].forward(h)

        if adapter_id and self.adapter_hub:
            adapter = self.adapter_hub.get(adapter_id)
            assert adapter is not None
            y = adapter.forward_out(y)

        latency = self.base_latency_ms + self.remote_extra_latency_ms
        return ActivationBatch(tensor=y, dtype=activation.dtype), latency

    def local_forward(self, expert_ref: ExpertRef, h: np.ndarray) -> np.ndarray:
        return self.experts[expert_ref.key()].forward(h)


def _now_ms() -> float:
    return time.time() * 1000.0


def make_expert_module(
    ref: ExpertRef,
    dim: int,
    domain: str,
    rng: np.random.Generator,
    specialty: np.ndarray | None = None,
) -> ExpertModule:
    # Experts biased toward their specialty direction
    if specialty is None:
        specialty = rng.normal(size=(dim,))
        specialty = specialty / (np.linalg.norm(specialty) + 1e-12)
    w = np.eye(dim, dtype=np.float64) * 0.5
    w += np.outer(specialty, specialty) * 1.5
    w += 0.05 * rng.normal(size=(dim, dim))
    bias = 0.01 * specialty
    return ExpertModule(
        ref=ref,
        weight=w,
        bias=bias,
        domain=domain,
        specialty_vec=specialty.astype(np.float64),
    )


def fingerprint_from_weights(module: ExpertModule, dim_fp: int = 64) -> np.ndarray:
    flat = module.weight.reshape(-1)
    if flat.size >= dim_fp:
        fp = flat[:dim_fp].copy()
    else:
        fp = np.zeros(dim_fp, dtype=np.float64)
        fp[: flat.size] = flat
    # Mix in specialty
    s = module.specialty_vec
    n = min(len(s), dim_fp)
    fp[:n] += s[:n]
    nrm = np.linalg.norm(fp)
    return fp / nrm if nrm > 1e-12 else fp
