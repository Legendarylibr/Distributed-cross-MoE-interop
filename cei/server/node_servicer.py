"""Expert node + host RunStep servicers."""

from __future__ import annotations

import logging
import threading

import grpc
import numpy as np

from cei import wire
from cei.client import LearnerClient, NodeClient, RegistryClient, RouterClient
from cei.host import MoEHost
from cei.node import ExpertNode
from cei.pb import cei_internal_pb2, cei_internal_pb2_grpc, cei_pb2, cei_pb2_grpc
from cei.security import audit, can_use_priority, resolve_principal
from cei.types import Budget

_LOG = logging.getLogger("cei.node")


class NodeServicer(cei_pb2_grpc.ExpertNodeServicer):
    def __init__(self, node: ExpertNode) -> None:
        self.node = node

    def LeaseCapacity(self, request, context):
        principal = resolve_principal(context, request.meta)
        priority = int(request.priority or 0)
        if priority >= 10 and not can_use_priority(principal, priority):
            priority = 0
        try:
            lease = self.node.lease_capacity(
                wire.expert_ref_from_pb(request.expert_ref),
                tokens_or_qps=request.tokens_or_qps,
                ttl_ms=request.ttl_ms or 5000,
                principal=principal,
                priority=priority,
            )
            audit(
                "lease_ok",
                principal=principal,
                expert=wire.expert_ref_from_pb(request.expert_ref).key(),
                lease_id=lease.lease_id,
            )
            return cei_pb2.LeaseCapacityResponse(
                lease_id=lease.lease_id,
                lease_deadline_unix_ms=int(lease.deadline_ms),
                granted_qps=lease.granted_qps,
            )
        except PermissionError:
            audit("lease_deny", principal=principal, reason="ACL_DENIED")
            return cei_pb2.LeaseCapacityResponse(error_code="ACL_DENIED")
        except KeyError:
            return cei_pb2.LeaseCapacityResponse(error_code="NOT_ROUTABLE")
        except (RuntimeError, ValueError) as exc:
            return cei_pb2.LeaseCapacityResponse(error_code=str(exc))

    def ReleaseCapacity(self, request, context):
        principal = resolve_principal(context, request.meta)
        try:
            self.node.release_capacity(request.lease_id, principal=principal)
        except PermissionError:
            audit("release_deny", principal=principal, reason="ACL_DENIED")
            return cei_pb2.ReleaseCapacityResponse(ok=False)
        return cei_pb2.ReleaseCapacityResponse(ok=True)

    def ForwardExpert(self, request, context):
        principal = resolve_principal(context, request.meta)
        try:
            act, lat = self.node.forward_expert(
                expert_ref=wire.expert_ref_from_pb(request.expert_ref),
                activation=wire.activation_from_pb(request.activation),
                lease_id=request.lease_id or None,
                adapter_id=request.adapter_id or None,
                request_id=request.meta.request_id or None,
                principal=principal,
                require_lease=bool(request.lease_id),
            )
            audit(
                "forward_ok",
                principal=principal,
                expert=wire.expert_ref_from_pb(request.expert_ref).key(),
                latency_ms=lat,
            )
            return cei_pb2.ForwardExpertResponse(
                activation=wire.activation_to_pb(act),
                actual_latency_ms=lat,
                expert_version="1.0.0",
            )
        except PermissionError as exc:
            reason = str(exc) or "ACL_DENIED"
            audit("forward_deny", principal=principal, reason=reason)
            return cei_pb2.ForwardExpertResponse(error_code=reason)
        except KeyError:
            return cei_pb2.ForwardExpertResponse(error_code="NOT_ROUTABLE")
        except (RuntimeError, ValueError) as exc:
            return cei_pb2.ForwardExpertResponse(error_code=str(exc))

    def ExportWeights(self, request, context):
        principal = resolve_principal(context, request.meta)
        audit("export_deny", principal=principal, reason="DENY_BY_DEFAULT")
        return cei_pb2.ExportWeightsResponse(error_code="ACL_DENIED")


class HostServicer(cei_internal_pb2_grpc.HostServiceServicer):
    """Runs MoEHost.forward with remote peer NodeClients."""

    def __init__(
        self,
        host: MoEHost,
        peer_addrs: dict[str, str],
        router_client: RouterClient,
        learner_client: LearnerClient,
        principal_id: str = "cei-host",
    ) -> None:
        self.host = host
        self.peer_addrs = peer_addrs
        self.router_client = router_client
        self.learner_client = learner_client
        self.principal_id = principal_id
        self._peers: dict[str, NodeClient] = {}

    def _ensure_peers(self) -> None:
        for mid, addr in self.peer_addrs.items():
            if mid == self.host.model_id:
                continue
            if mid in self._peers:
                continue
            c = NodeClient(addr, principal_id=self.principal_id)
            for _ in range(30):
                try:
                    c.connect()
                    import grpc as _grpc

                    _grpc.channel_ready_future(c._channel).result(timeout=2)  # noqa: SLF001
                    self._peers[mid] = c
                    break
                except Exception:
                    import time as _time

                    _time.sleep(1.0)
            else:
                pass

    def RunStep(self, request, context):
        # Driving the host consumes fleet capacity: gate on the node ACL.
        principal = resolve_principal(context, request.meta)
        node = self.host.node
        if not node.acl_open and (principal is None or principal not in node.acl_allow):
            audit("runstep_deny", principal=principal, reason="ACL_DENIED")
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details("ACL_DENIED")
            return cei_internal_pb2.RunStepResponse(error_code="ACL_DENIED")
        try:
            self._ensure_peers()
            x = np.asarray(list(request.x), dtype=np.float64)
            domain_vec = np.asarray(list(request.domain_vec), dtype=np.float64)
            mode = request.mode or "learned"
            budget = (
                wire.budget_from_pb(request.budget)
                if request.HasField("budget")
                else Budget(allow_soft_latency=True, require_leases=True)
            )

            missing = [
                mid
                for mid in self.peer_addrs
                if mid != self.host.model_id and mid not in self._peers
            ]
            if missing and mode != "local":
                print(
                    f"warning: peers not ready after retries: {missing}; "
                    "remote forwards may fallback",
                    flush=True,
                )

            hidden, outcome = self.host.forward_distributed(
                x=x,
                domain_vec=domain_vec,
                peers=self._peers,
                router_client=self.router_client,
                learner_client=self.learner_client,
                budget=budget,
                mode=mode if mode != "local" else "local",
                use_marketplace=(mode != "local"),
                force_local=(mode == "local"),
            )
            return cei_internal_pb2.RunStepResponse(
                outcome=wire.outcome_to_report_pb(outcome),
                hidden=[float(v) for v in hidden.tolist()],
            )
        except Exception as exc:  # noqa: BLE001
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return cei_internal_pb2.RunStepResponse(error_code=str(exc))


def start_heartbeat_loop(
    registry: RegistryClient,
    node: ExpertNode,
    stop_event: threading.Event,
    interval_s: float = 5.0,
) -> threading.Thread:
    def _loop() -> None:
        while not stop_event.is_set():
            try:
                registry.heartbeat(node.node_id, expert_refs=None)
            except Exception as exc:  # noqa: BLE001 — transient; retried next tick
                _LOG.warning("heartbeat failed: %s", exc)
            stop_event.wait(interval_s)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def register_all_experts(registry: RegistryClient, node: ExpertNode) -> None:
    for desc in node.descriptors.values():
        registry.register(desc, force=True, promote=True)
    registry.heartbeat(
        node.node_id,
        None,
        capacity_qps=node.get_capacity_snapshot(),
        load_qps=node.get_load_snapshot(),
    )
