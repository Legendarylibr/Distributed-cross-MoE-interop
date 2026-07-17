"""Learner gRPC servicers (ReportOutcome + EstimateUtility)."""

from __future__ import annotations

import numpy as np

from cei.learner import ContextualBanditLearner
from cei.pb import cei_internal_pb2, cei_internal_pb2_grpc, cei_pb2, cei_pb2_grpc
from cei.security import (
    audit,
    get_config,
    resolve_principal,
    verify_outcome_attestation,
)
from cei import wire


class LearnerServicer(cei_pb2_grpc.CombinationLearnerServicer):
    def __init__(self, learner: ContextualBanditLearner) -> None:
        self.learner = learner

    def ReportOutcome(self, request, context):
        meta_p = request.meta.principal_id if request.meta else None
        principal = resolve_principal(context, meta_p)
        cfg = get_config()
        attestation = getattr(request, "attestation", "") or ""
        ok_attest = verify_outcome_attestation(
            plan_id=request.plan_id,
            host_model_id=request.host_model_id,
            reward=request.reward,
            utility=request.utility,
            latency_ms=request.latency_ms,
            tokens=request.tokens,
            attestation=attestation,
        )
        if cfg.require_outcome_attestation and not ok_attest:
            audit(
                "outcome_deny",
                principal=principal,
                plan_id=request.plan_id,
                reason="ATTESTATION_INVALID",
            )
            return cei_pb2.ReportOutcomeResponse(ok=False, learner_version=str(self.learner.version))
        outcome = wire.outcome_from_report_pb(request)
        self.learner.report(outcome)
        audit(
            "outcome_ok",
            principal=principal,
            plan_id=request.plan_id,
            reward=request.reward,
            attested=ok_attest,
        )
        return cei_pb2.ReportOutcomeResponse(
            ok=True, learner_version=str(self.learner.version)
        )


class LearnerInternalServicer(cei_internal_pb2_grpc.LearnerInternalServicer):
    def __init__(self, learner: ContextualBanditLearner) -> None:
        self.learner = learner

    def EstimateUtility(self, request, context):
        phi = np.asarray(list(request.context_embedding), dtype=np.float64)
        plan = wire.plan_from_pb(request.plan)
        sims = dict(request.fingerprint_sims)
        alpha = request.alpha if request.alpha else 0.5
        u = self.learner.cold_start_utility(phi, plan, sims, alpha=alpha)
        return cei_internal_pb2.EstimateUtilityResponse(
            utility=float(u), learner_version=str(self.learner.version)
        )

    def GetPolicySnapshot(self, request, context):
        snap = self.learner.policy_snapshot()
        if request.known_version == snap["version"] and snap["version"] > 0:
            return cei_internal_pb2.GetPolicySnapshotResponse(
                version=snap["version"],
                ctx_dim=snap["ctx_dim"],
                unchanged=True,
            )
        arms = [
            cei_internal_pb2.ArmParams(
                arm_key=a["arm_key"],
                theta=[float(x) for x in np.asarray(a["theta"]).tolist()],
                count=int(a["count"]),
            )
            for a in snap["arms"]
        ]
        return cei_internal_pb2.GetPolicySnapshotResponse(
            version=snap["version"],
            ctx_dim=snap["ctx_dim"],
            arms=arms,
            unchanged=False,
        )
