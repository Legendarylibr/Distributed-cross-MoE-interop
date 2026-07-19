"""Router gRPC servicer — uses remote registry + learner adapters."""

from __future__ import annotations

import numpy as np

from cei import wire
from cei.client import LearnerClient, RegistryClient
from cei.pb import cei_pb2, cei_pb2_grpc
from cei.router import CombinationRouter
from cei.security import resolve_principal


class RouterServicer(cei_pb2_grpc.CombinationRouterServicer):
    def __init__(
        self,
        registry_addr: str,
        learner_addr: str,
        principal_id: str = "cei-router",
    ) -> None:
        self.registry = RegistryClient(registry_addr, principal_id=principal_id)
        self.learner = LearnerClient(learner_addr, principal_id=principal_id)
        self.registry.connect()
        self.learner.connect()
        self.router = CombinationRouter(registry=self.registry, learner=self.learner)  # type: ignore[arg-type]
        self.router.lambda_lat = 0.002
        self.router.lambda_cap = 0.01

    def ProposeCombinations(self, request, context):
        principal = resolve_principal(context, request.meta)
        local_topk = wire.local_topk_from_pb(list(request.local_topk))
        budget = wire.budget_from_pb(request.budget) if request.HasField("budget") else None
        include_local = (
            request.include_local_only
            if request.HasField("include_local_only")
            else True
        )
        plans = self.router.propose(
            host_model_id=request.host_model_id,
            context_embedding=np.asarray(list(request.context_embedding), dtype=np.float64),
            local_topk=local_topk,
            host_dim=request.host_dim or 32,
            budget=budget,
            layer_hints=list(request.layer_hints) or None,
            n=request.n or 8,
            include_local_only=include_local,
            principal=principal,
        )
        return cei_pb2.ProposeCombinationsResponse(
            plans=[wire.plan_to_pb(p) for p in plans],
            router_policy_version=self.router.version,
        )
