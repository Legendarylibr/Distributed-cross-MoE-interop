"""Registry gRPC servicer."""

from __future__ import annotations

import grpc

from cei import wire
from cei.pb import cei_pb2, cei_pb2_grpc
from cei.registry import ExpertRegistry
from cei.security import (
    audit,
    can_publish,
    get_config,
    resolve_principal,
)


class RegistryServicer(cei_pb2_grpc.ExpertRegistryServicer):
    def __init__(self, registry: ExpertRegistry | None = None) -> None:
        cfg = get_config()
        if registry is None:
            registry = ExpertRegistry(
                allow_all=cfg.registry_allow_all,
                auto_promote=cfg.auto_promote,
                consumer_principals=set(cfg.registry_consumers),
                enforce_ownership=(cfg.profile == "secure"),
            )
        self.registry = registry

    def RegisterExpert(self, request, context):
        principal = resolve_principal(context, request.meta)
        if not can_publish(principal):
            audit("register_deny", principal=principal, reason="PUBLISHER_ACL")
            return cei_pb2.RegisterExpertResponse(ok=False, error_code="ACL_DENIED")
        try:
            d = wire.descriptor_from_pb(request.descriptor)
            promote = bool(request.promote) or get_config().auto_promote
            self.registry.register(
                d, force=request.force, promote=promote, principal=principal
            )
            audit(
                "register_ok",
                principal=principal,
                expert=d.expert_ref.key(),
                promote=promote,
            )
            return cei_pb2.RegisterExpertResponse(ok=True, registry_version="1")
        except PermissionError as exc:
            audit("register_deny", principal=principal, reason=str(exc))
            return cei_pb2.RegisterExpertResponse(ok=False, error_code="ACL_DENIED")
        except ValueError as exc:
            return cei_pb2.RegisterExpertResponse(ok=False, error_code=str(exc))
        except Exception as exc:  # noqa: BLE001
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return cei_pb2.RegisterExpertResponse(ok=False, error_code="INTERNAL")

    def Heartbeat(self, request, context):
        principal = resolve_principal(context, request.meta)
        # Heartbeats mutate capacity/load: publisher-gated like registration.
        if not can_publish(principal):
            audit("heartbeat_deny", principal=principal, reason="PUBLISHER_ACL")
            return cei_pb2.HeartbeatResponse(ok=False, next_heartbeat_ms=0)
        refs = [wire.expert_ref_from_pb(r) for r in request.expert_refs] or None
        next_ms = self.registry.heartbeat(
            request.node_id,
            refs,
            capacity_qps=dict(request.capacity_qps),
            load_qps=dict(request.load_qps),
        )
        return cei_pb2.HeartbeatResponse(ok=True, next_heartbeat_ms=next_ms)

    def Deregister(self, request, context):
        principal = resolve_principal(context, request.meta)
        if not can_publish(principal):
            audit("deregister_deny", principal=principal, reason="PUBLISHER_ACL")
            return cei_pb2.DeregisterResponse(ok=False)
        refs = [wire.expert_ref_from_pb(r) for r in request.expert_refs]
        try:
            self.registry.deregister(refs, principal=principal)
        except PermissionError as exc:
            audit("deregister_deny", principal=principal, reason=str(exc))
            return cei_pb2.DeregisterResponse(ok=False)
        audit("deregister_ok", principal=principal, count=len(refs))
        return cei_pb2.DeregisterResponse(ok=True)

    def DescribeExperts(self, request, context):
        principal = resolve_principal(context, request.meta)
        if request.HasField("nn"):
            nn = request.nn
            import numpy as np

            hits = self.registry.describe_nn(
                fingerprint=np.asarray(list(nn.fingerprint), dtype=np.float64),
                k=nn.k or 32,
                host_dim_in=nn.host_dim_in or None,
                host_dim_out=nn.host_dim_out or None,
                domain_tags=list(nn.domain_tags) or None,
                principal=principal,
            )
            experts = []
            routable = []
            loads = []
            caps = []
            for d, _, r in hits:
                experts.append(wire.descriptor_to_pb(d))
                routable.append(r)
                loads.append(self.registry.load(d.expert_ref))
                caps.append(self.registry.capacity(d.expert_ref) or d.capacity_qps)
            return cei_pb2.DescribeExpertsResponse(
                experts=experts,
                routable=routable,
                load_qps=loads,
                capacity_qps=caps,
            )
        if request.HasField("explicit"):
            refs = [wire.expert_ref_from_pb(r) for r in request.explicit.expert_refs]
            hits = self.registry.describe_explicit(refs, principal=principal)
            experts = []
            routable = []
            loads = []
            caps = []
            for d, r in hits:
                experts.append(wire.descriptor_to_pb(d))
                routable.append(r)
                loads.append(self.registry.load(d.expert_ref))
                caps.append(self.registry.capacity(d.expert_ref) or d.capacity_qps)
            return cei_pb2.DescribeExpertsResponse(
                experts=experts,
                routable=routable,
                load_qps=loads,
                capacity_qps=caps,
            )
        return cei_pb2.DescribeExpertsResponse()
