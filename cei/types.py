"""Data contracts aligned with SPEC.md / schemas/."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid

import numpy as np


class DType(str, Enum):
    F16 = "F16"
    BF16 = "BF16"
    F32 = "F32"
    F8 = "F8"


class CombinationOp(str, Enum):
    REPLACE = "REPLACE"
    AUGMENT = "AUGMENT"


@dataclass(frozen=True, slots=True)
class ExpertRef:
    model_id: str
    layer_id: int
    expert_id: int

    def key(self) -> str:
        return f"{self.model_id}:{self.layer_id}:{self.expert_id}"

    def __str__(self) -> str:
        return self.key()


@dataclass
class Budget:
    max_remote_latency_ms: float = 50.0
    max_remote_experts: int = 4
    max_flops: int = 0
    strict_local_fallback: bool = False
    require_leases: bool = True
    allow_soft_latency: bool = False


@dataclass
class ExpertDescriptor:
    expert_ref: ExpertRef
    version: str
    dim_in: int
    dim_out: int
    dtype: DType
    fingerprint: np.ndarray
    cost_flops: int
    p50_latency_ms: float
    capacity_qps: float
    domain_tags: list[str] = field(default_factory=list)
    adapter_id: Optional[str] = None
    acl_policy_id: Optional[str] = None
    affinity_tags: list[str] = field(default_factory=list)
    node_id: Optional[str] = None

    def normalized_fingerprint(self) -> np.ndarray:
        fp = np.asarray(self.fingerprint, dtype=np.float64)
        n = np.linalg.norm(fp)
        if n < 1e-12:
            return fp
        return fp / n


@dataclass
class CombinationStep:
    layer_id: int
    expert_refs: list[ExpertRef]
    weights: list[float]
    op: CombinationOp = CombinationOp.REPLACE
    lease_ids: list[str] = field(default_factory=list)
    adapter_ids: list[str] = field(default_factory=list)


@dataclass
class CombinationPlan:
    plan_id: str
    host_model_id: str
    steps: list[CombinationStep]
    budget: Budget
    ttl_ms: int = 5000
    score: float = 0.0
    local_only_equivalent: bool = False

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def remote_refs(self, host_model_id: str) -> list[ExpertRef]:
        out: list[ExpertRef] = []
        for step in self.steps:
            for ref in step.expert_refs:
                if ref.model_id != host_model_id:
                    out.append(ref)
        return out

    def arm_key(self) -> str:
        parts: list[str] = []
        for step in sorted(self.steps, key=lambda s: s.layer_id):
            refs = ",".join(sorted(r.key() for r in step.expert_refs))
            parts.append(f"{step.layer_id}:{step.op.value}:{refs}")
        return "|".join(parts) if parts else "local_only"


@dataclass
class ActivationBatch:
    tensor: np.ndarray
    dtype: DType = DType.F32
    grad_required: bool = False
    cache_key: Optional[str] = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)


@dataclass
class FallbackEvent:
    layer_id: int
    reason: str


@dataclass
class Outcome:
    plan_id: str
    host_model_id: str
    reward: float
    utility: float
    latency_ms: float
    capacity_penalty: float
    tokens: int
    fallbacks: list[FallbackEvent] = field(default_factory=list)
    partial: bool = False
    context_embedding: Optional[np.ndarray] = None
    plan: Optional[CombinationPlan] = None


@dataclass
class Lease:
    lease_id: str
    expert_ref: ExpertRef
    deadline_ms: float
    granted_qps: float
