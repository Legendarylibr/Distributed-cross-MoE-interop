"""Convert between CEI dataclasses and protobuf messages."""

from __future__ import annotations

import uuid

import numpy as np

from cei.pb import cei_pb2
from cei.types import (
    ActivationBatch,
    Budget,
    CombinationOp,
    CombinationPlan,
    CombinationStep,
    DType,
    ExpertDescriptor,
    ExpertRef,
    FallbackEvent,
    Outcome,
)

_DTYPE_TO_PB = {
    DType.F16: cei_pb2.F16,
    DType.BF16: cei_pb2.BF16,
    DType.F32: cei_pb2.F32,
    DType.F8: cei_pb2.F8,
}
_PB_TO_DTYPE = {v: k for k, v in _DTYPE_TO_PB.items()}

_OP_TO_PB = {
    CombinationOp.REPLACE: cei_pb2.REPLACE,
    CombinationOp.AUGMENT: cei_pb2.AUGMENT,
}
_PB_TO_OP = {
    cei_pb2.REPLACE: CombinationOp.REPLACE,
    cei_pb2.AUGMENT: CombinationOp.AUGMENT,
    cei_pb2.OP_UNSPECIFIED: CombinationOp.REPLACE,
}


def new_meta(principal_id: str = "cei-dev", request_id: str | None = None) -> cei_pb2.RequestMeta:
    return cei_pb2.RequestMeta(
        request_id=request_id or str(uuid.uuid4()),
        principal_id=principal_id,
        cei_version="0.1.0",
    )


def expert_ref_to_pb(ref: ExpertRef) -> cei_pb2.ExpertRef:
    return cei_pb2.ExpertRef(
        model_id=ref.model_id, layer_id=ref.layer_id, expert_id=ref.expert_id
    )


def expert_ref_from_pb(msg: cei_pb2.ExpertRef) -> ExpertRef:
    return ExpertRef(model_id=msg.model_id, layer_id=msg.layer_id, expert_id=msg.expert_id)


def budget_to_pb(b: Budget) -> cei_pb2.Budget:
    msg = cei_pb2.Budget(
        max_remote_latency_ms=b.max_remote_latency_ms,
        max_remote_experts=b.max_remote_experts,
        max_flops=b.max_flops,
    )
    msg.strict_local_fallback = b.strict_local_fallback
    msg.require_leases = b.require_leases
    msg.allow_soft_latency = b.allow_soft_latency
    return msg


def budget_from_pb(msg: cei_pb2.Budget) -> Budget:
    return Budget(
        max_remote_latency_ms=msg.max_remote_latency_ms or 50.0,
        max_remote_experts=msg.max_remote_experts or 4,
        max_flops=msg.max_flops,
        strict_local_fallback=(
            msg.strict_local_fallback if msg.HasField("strict_local_fallback") else False
        ),
        require_leases=msg.require_leases if msg.HasField("require_leases") else True,
        allow_soft_latency=(
            msg.allow_soft_latency if msg.HasField("allow_soft_latency") else False
        ),
    )


def descriptor_to_pb(d: ExpertDescriptor) -> cei_pb2.ExpertDescriptor:
    return cei_pb2.ExpertDescriptor(
        expert_ref=expert_ref_to_pb(d.expert_ref),
        version=d.version,
        dim_in=d.dim_in,
        dim_out=d.dim_out,
        dtype=_DTYPE_TO_PB.get(d.dtype, cei_pb2.F32),
        domain_tags=list(d.domain_tags),
        fingerprint=[float(x) for x in np.asarray(d.fingerprint).tolist()],
        cost_flops=d.cost_flops,
        p50_latency_ms=d.p50_latency_ms,
        capacity_qps=d.capacity_qps,
        adapter_id=d.adapter_id or "",
        acl_policy_id=d.acl_policy_id or "",
        affinity_tags=list(d.affinity_tags),
        node_id=d.node_id or "",
    )


def descriptor_from_pb(msg: cei_pb2.ExpertDescriptor) -> ExpertDescriptor:
    return ExpertDescriptor(
        expert_ref=expert_ref_from_pb(msg.expert_ref),
        version=msg.version,
        dim_in=msg.dim_in,
        dim_out=msg.dim_out,
        dtype=_PB_TO_DTYPE.get(msg.dtype, DType.F32),
        fingerprint=np.asarray(list(msg.fingerprint), dtype=np.float64),
        cost_flops=msg.cost_flops,
        p50_latency_ms=msg.p50_latency_ms,
        capacity_qps=msg.capacity_qps,
        domain_tags=list(msg.domain_tags),
        adapter_id=msg.adapter_id or None,
        acl_policy_id=msg.acl_policy_id or None,
        affinity_tags=list(msg.affinity_tags),
        node_id=msg.node_id or None,
    )


def plan_to_pb(plan: CombinationPlan) -> cei_pb2.CombinationPlan:
    steps = []
    for s in plan.steps:
        steps.append(
            cei_pb2.CombinationStep(
                layer_id=s.layer_id,
                expert_refs=[expert_ref_to_pb(r) for r in s.expert_refs],
                weights=list(s.weights),
                op=_OP_TO_PB.get(s.op, cei_pb2.REPLACE),
                lease_ids=list(s.lease_ids),
                adapter_ids=list(s.adapter_ids),
            )
        )
    return cei_pb2.CombinationPlan(
        plan_id=plan.plan_id,
        host_model_id=plan.host_model_id,
        steps=steps,
        budget=budget_to_pb(plan.budget),
        ttl_ms=plan.ttl_ms,
        score=plan.score,
        local_only_equivalent=plan.local_only_equivalent,
    )


def plan_from_pb(msg: cei_pb2.CombinationPlan) -> CombinationPlan:
    steps = []
    for s in msg.steps:
        steps.append(
            CombinationStep(
                layer_id=s.layer_id,
                expert_refs=[expert_ref_from_pb(r) for r in s.expert_refs],
                weights=list(s.weights),
                op=_PB_TO_OP.get(s.op, CombinationOp.REPLACE),
                lease_ids=list(s.lease_ids),
                adapter_ids=list(s.adapter_ids),
            )
        )
    return CombinationPlan(
        plan_id=msg.plan_id,
        host_model_id=msg.host_model_id,
        steps=steps,
        budget=budget_from_pb(msg.budget) if msg.HasField("budget") else Budget(),
        ttl_ms=msg.ttl_ms or 5000,
        score=msg.score,
        local_only_equivalent=msg.local_only_equivalent,
    )


def activation_to_pb(act: ActivationBatch) -> cei_pb2.ActivationBatch:
    arr = np.asarray(act.tensor, dtype=np.float64)
    return cei_pb2.ActivationBatch(
        tensor=arr.tobytes(),
        shape=list(arr.shape),
        dtype=_DTYPE_TO_PB.get(act.dtype, cei_pb2.F32),
        grad_required=act.grad_required,
        cache_key=act.cache_key or "",
    )


def activation_from_pb(msg: cei_pb2.ActivationBatch) -> ActivationBatch:
    shape = tuple(int(x) for x in msg.shape) if msg.shape else (-1,)
    arr = np.frombuffer(msg.tensor, dtype=np.float64).reshape(shape)
    return ActivationBatch(
        tensor=arr.copy(),
        dtype=_PB_TO_DTYPE.get(msg.dtype, DType.F32),
        grad_required=msg.grad_required,
        cache_key=msg.cache_key or None,
    )


def outcome_to_report_pb(
    outcome: Outcome,
    meta: cei_pb2.RequestMeta | None = None,
    *,
    attestation: str | None = None,
) -> cei_pb2.ReportOutcomeRequest:
    fb = [
        cei_pb2.FallbackEvent(layer_id=f.layer_id, reason=f.reason) for f in outcome.fallbacks
    ]
    if attestation is None:
        from cei.security import get_config, sign_outcome

        cfg = get_config()
        if cfg.outcome_hmac_secret:
            attestation = sign_outcome(
                plan_id=outcome.plan_id,
                host_model_id=outcome.host_model_id,
                reward=outcome.reward,
                utility=outcome.utility,
                latency_ms=outcome.latency_ms,
                tokens=outcome.tokens,
            )
        else:
            attestation = ""
    req = cei_pb2.ReportOutcomeRequest(
        meta=meta or new_meta(),
        plan_id=outcome.plan_id,
        host_model_id=outcome.host_model_id,
        reward=outcome.reward,
        utility=outcome.utility,
        latency_ms=outcome.latency_ms,
        capacity_penalty=outcome.capacity_penalty,
        tokens=outcome.tokens,
        fallbacks=fb,
        partial=outcome.partial,
        context_embedding=(
            [float(x) for x in outcome.context_embedding.tolist()]
            if outcome.context_embedding is not None
            else []
        ),
        attestation=attestation or "",
    )
    if outcome.plan is not None:
        req.plan_snapshot.CopyFrom(plan_to_pb(outcome.plan))
    return req


def outcome_from_report_pb(msg: cei_pb2.ReportOutcomeRequest) -> Outcome:
    return Outcome(
        plan_id=msg.plan_id,
        host_model_id=msg.host_model_id,
        reward=msg.reward,
        utility=msg.utility,
        latency_ms=msg.latency_ms,
        capacity_penalty=msg.capacity_penalty,
        tokens=msg.tokens,
        fallbacks=[FallbackEvent(layer_id=f.layer_id, reason=f.reason) for f in msg.fallbacks],
        partial=msg.partial,
        context_embedding=(
            np.asarray(list(msg.context_embedding), dtype=np.float64)
            if msg.context_embedding
            else None
        ),
        plan=plan_from_pb(msg.plan_snapshot) if msg.HasField("plan_snapshot") else None,
    )


def local_topk_to_pb(
    local_topk: dict[int, list[tuple[ExpertRef, float]]],
) -> list[cei_pb2.LocalTopKEntry]:
    entries = []
    for layer_id, pairs in local_topk.items():
        entries.append(
            cei_pb2.LocalTopKEntry(
                layer_id=layer_id,
                expert_refs=[expert_ref_to_pb(r) for r, _ in pairs],
                weights=[w for _, w in pairs],
            )
        )
    return entries


def local_topk_from_pb(
    entries: list[cei_pb2.LocalTopKEntry],
) -> dict[int, list[tuple[ExpertRef, float]]]:
    out: dict[int, list[tuple[ExpertRef, float]]] = {}
    for e in entries:
        pairs = [
            (expert_ref_from_pb(r), float(e.weights[i]) if i < len(e.weights) else 1.0)
            for i, r in enumerate(e.expert_refs)
        ]
        out[e.layer_id] = pairs
    return out
