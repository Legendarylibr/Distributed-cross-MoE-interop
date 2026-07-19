"""gRPC client wrappers.

All RPCs carry a deadline (CEI_RPC_TIMEOUT_S, default 10s; RunStep uses
CEI_RUNSTEP_TIMEOUT_S, default 60s) so a hung peer cannot stall callers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import grpc
import numpy as np

from cei import wire
from cei.pb import cei_internal_pb2, cei_internal_pb2_grpc, cei_pb2, cei_pb2_grpc
from cei.tlsutil import make_channel
from cei.types import (
    ActivationBatch,
    Budget,
    CombinationPlan,
    ExpertDescriptor,
    ExpertRef,
    Lease,
    Outcome,
)


def _rpc_timeout() -> float:
    return float(os.environ.get("CEI_RPC_TIMEOUT_S", "10"))


def _runstep_timeout() -> float:
    return float(os.environ.get("CEI_RUNSTEP_TIMEOUT_S", "60"))


@dataclass
class RegistryClient:
    addr: str
    principal_id: str = "cei-dev"
    _channel: grpc.Channel | None = None
    _stub: cei_pb2_grpc.ExpertRegistryStub | None = None
    _load_cache: dict[str, float] = field(default_factory=dict)
    _capacity_cache: dict[str, float] = field(default_factory=dict)

    def connect(self) -> None:
        self._channel = make_channel(self.addr)
        self._stub = cei_pb2_grpc.ExpertRegistryStub(self._channel)

    @property
    def stub(self) -> cei_pb2_grpc.ExpertRegistryStub:
        if self._stub is None:
            self.connect()
        assert self._stub is not None
        return self._stub

    def _ingest_describe(self, resp: cei_pb2.DescribeExpertsResponse) -> None:
        for i, dmsg in enumerate(resp.experts):
            key = f"{dmsg.expert_ref.model_id}:{dmsg.expert_ref.layer_id}:{dmsg.expert_ref.expert_id}"
            if i < len(resp.load_qps):
                self._load_cache[key] = float(resp.load_qps[i])
            if i < len(resp.capacity_qps):
                self._capacity_cache[key] = float(resp.capacity_qps[i])
            elif dmsg.capacity_qps:
                self._capacity_cache[key] = float(dmsg.capacity_qps)

    def register(
        self, descriptor: ExpertDescriptor, force: bool = False, promote: bool = True
    ) -> None:
        resp = self.stub.RegisterExpert(
            cei_pb2.RegisterExpertRequest(
                meta=wire.new_meta(self.principal_id),
                descriptor=wire.descriptor_to_pb(descriptor),
                force=force,
                promote=promote,
            ),
            timeout=_rpc_timeout(),
        )
        if not resp.ok:
            raise RuntimeError(resp.error_code or "REGISTER_FAILED")
        self._capacity_cache[descriptor.expert_ref.key()] = descriptor.capacity_qps

    def heartbeat(
        self,
        node_id: str,
        expert_refs: list[ExpertRef] | None = None,
        capacity_qps: dict[str, float] | None = None,
        load_qps: dict[str, float] | None = None,
    ) -> int:
        req = cei_pb2.HeartbeatRequest(
            meta=wire.new_meta(self.principal_id),
            node_id=node_id,
            expert_refs=[wire.expert_ref_to_pb(r) for r in (expert_refs or [])],
            capacity_qps=capacity_qps or {},
            load_qps=load_qps or {},
        )
        resp = self.stub.Heartbeat(req, timeout=_rpc_timeout())
        if capacity_qps:
            self._capacity_cache.update(capacity_qps)
        if load_qps:
            self._load_cache.update(load_qps)
        return int(resp.next_heartbeat_ms)

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
        nn = cei_pb2.NNQuery(
            fingerprint=[float(x) for x in np.asarray(fingerprint).tolist()],
            k=k,
            host_dim_in=host_dim_in or 0,
            host_dim_out=host_dim_out or 0,
            domain_tags=domain_tags or [],
        )
        resp = self.stub.DescribeExperts(
            cei_pb2.DescribeExpertsRequest(
                meta=wire.new_meta(principal or self.principal_id),
                nn=nn,
                limit=k,
            ),
            timeout=_rpc_timeout(),
        )
        self._ingest_describe(resp)
        out: list[tuple[ExpertDescriptor, float, bool]] = []
        q = np.asarray(fingerprint, dtype=np.float64)
        q = q / (np.linalg.norm(q) + 1e-12)
        for i, dmsg in enumerate(resp.experts):
            d = wire.descriptor_from_pb(dmsg)
            if exclude_model and d.expert_ref.model_id == exclude_model:
                continue
            sim = float(np.dot(q, d.normalized_fingerprint()))
            routable = resp.routable[i] if i < len(resp.routable) else True
            out.append((d, sim, routable))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def get(self, ref: ExpertRef) -> ExpertDescriptor | None:
        resp = self.stub.DescribeExperts(
            cei_pb2.DescribeExpertsRequest(
                meta=wire.new_meta(self.principal_id),
                explicit=cei_pb2.ExplicitRefs(expert_refs=[wire.expert_ref_to_pb(ref)]),
            ),
            timeout=_rpc_timeout(),
        )
        self._ingest_describe(resp)
        if not resp.experts:
            return None
        return wire.descriptor_from_pb(resp.experts[0])

    def load(self, ref: ExpertRef) -> float:
        return float(self._load_cache.get(ref.key(), 0.0))

    def capacity(self, ref: ExpertRef) -> float:
        if ref.key() in self._capacity_cache:
            return float(self._capacity_cache[ref.key()])
        d = self.get(ref)
        return d.capacity_qps if d else 0.0

    def close(self) -> None:
        if self._channel:
            self._channel.close()


@dataclass
class LearnerClient:
    addr: str
    principal_id: str = "cei-dev"
    _channel: grpc.Channel | None = None
    _stub: cei_pb2_grpc.CombinationLearnerStub | None = None
    _internal: cei_internal_pb2_grpc.LearnerInternalStub | None = None
    _cached_snap: dict | None = None

    def connect(self) -> None:
        self._channel = make_channel(self.addr)
        self._stub = cei_pb2_grpc.CombinationLearnerStub(self._channel)
        self._internal = cei_internal_pb2_grpc.LearnerInternalStub(self._channel)

    @property
    def stub(self) -> cei_pb2_grpc.CombinationLearnerStub:
        if self._stub is None:
            self.connect()
        assert self._stub is not None
        return self._stub

    @property
    def internal(self) -> cei_internal_pb2_grpc.LearnerInternalStub:
        if self._internal is None:
            self.connect()
        assert self._internal is not None
        return self._internal

    def report(self, outcome: Outcome) -> str:
        resp = self.stub.ReportOutcome(
            wire.outcome_to_report_pb(outcome, wire.new_meta(self.principal_id)),
            timeout=_rpc_timeout(),
        )
        if not resp.ok:
            raise RuntimeError("OUTCOME_REJECTED")
        return resp.learner_version

    def cold_start_utility(
        self,
        phi: np.ndarray,
        plan: CombinationPlan,
        fingerprint_sims: dict[str, float],
        alpha: float = 0.5,
        snapshot: dict | None = None,
    ) -> float:
        if snapshot is not None:
            from cei.learner import score_plan_from_snapshot

            cached = score_plan_from_snapshot(phi, plan, snapshot, fingerprint_sims, alpha)
            if cached is not None:
                return cached
        resp = self.internal.EstimateUtility(
            cei_internal_pb2.EstimateUtilityRequest(
                meta=wire.new_meta(self.principal_id),
                context_embedding=[float(x) for x in np.asarray(phi).tolist()],
                plan=wire.plan_to_pb(plan),
                fingerprint_sims=fingerprint_sims,
                alpha=alpha,
            ),
            timeout=_rpc_timeout(),
        )
        return float(resp.utility)

    def get_policy_snapshot(self, known_version: int = -1) -> dict:
        resp = self.internal.GetPolicySnapshot(
            cei_internal_pb2.GetPolicySnapshotRequest(
                meta=wire.new_meta(self.principal_id),
                known_version=known_version,
            ),
            timeout=_rpc_timeout(),
        )
        if resp.unchanged and self._cached_snap and self._cached_snap.get("version") == resp.version:
            return self._cached_snap
        snap = {
            "version": resp.version,
            "ctx_dim": resp.ctx_dim,
            "arms": [
                {
                    "arm_key": a.arm_key,
                    "theta": np.asarray(list(a.theta), dtype=np.float64),
                    "count": a.count,
                }
                for a in resp.arms
            ],
        }
        self._cached_snap = snap
        return snap

    def estimate_utility(self, phi: np.ndarray, plan: CombinationPlan) -> float:
        return self.cold_start_utility(phi, plan, {})

    def close(self) -> None:
        if self._channel:
            self._channel.close()


@dataclass
class RouterClient:
    addr: str
    principal_id: str = "cei-dev"
    _channel: grpc.Channel | None = None
    _stub: cei_pb2_grpc.CombinationRouterStub | None = None

    def connect(self) -> None:
        self._channel = make_channel(self.addr)
        self._stub = cei_pb2_grpc.CombinationRouterStub(self._channel)

    @property
    def stub(self) -> cei_pb2_grpc.CombinationRouterStub:
        if self._stub is None:
            self.connect()
        assert self._stub is not None
        return self._stub

    def propose(
        self,
        host_model_id: str,
        context_embedding: np.ndarray,
        local_topk: dict[int, list[tuple[ExpertRef, float]]],
        host_dim: int,
        budget: Budget | None = None,
        layer_hints: list[int] | None = None,
        n: int = 8,
        include_local_only: bool = True,
    ) -> list[CombinationPlan]:
        budget = budget or Budget()
        req = cei_pb2.ProposeCombinationsRequest(
            meta=wire.new_meta(self.principal_id),
            host_model_id=host_model_id,
            context_embedding=[float(x) for x in np.asarray(context_embedding).tolist()],
            layer_hints=layer_hints or [],
            budget=wire.budget_to_pb(budget),
            n=n,
            local_topk=wire.local_topk_to_pb(local_topk),
            host_dim=host_dim,
        )
        req.include_local_only = include_local_only
        resp = self.stub.ProposeCombinations(req, timeout=_rpc_timeout())
        return [wire.plan_from_pb(p) for p in resp.plans]

    def close(self) -> None:
        if self._channel:
            self._channel.close()


@dataclass
class NodeClient:
    addr: str
    principal_id: str = "cei-dev"
    _channel: grpc.Channel | None = None
    _stub: cei_pb2_grpc.ExpertNodeStub | None = None
    _host: cei_internal_pb2_grpc.HostServiceStub | None = None

    def connect(self) -> None:
        self._channel = make_channel(self.addr)
        self._stub = cei_pb2_grpc.ExpertNodeStub(self._channel)
        self._host = cei_internal_pb2_grpc.HostServiceStub(self._channel)

    @property
    def stub(self) -> cei_pb2_grpc.ExpertNodeStub:
        if self._stub is None:
            self.connect()
        assert self._stub is not None
        return self._stub

    @property
    def host_stub(self) -> cei_internal_pb2_grpc.HostServiceStub:
        if self._host is None:
            self.connect()
        assert self._host is not None
        return self._host

    def lease_capacity(
        self, expert_ref: ExpertRef, tokens_or_qps: float, ttl_ms: int, priority: int = 0
    ) -> Lease:
        resp = self.stub.LeaseCapacity(
            cei_pb2.LeaseCapacityRequest(
                meta=wire.new_meta(self.principal_id),
                expert_ref=wire.expert_ref_to_pb(expert_ref),
                tokens_or_qps=tokens_or_qps,
                ttl_ms=ttl_ms,
                priority=priority,
            ),
            timeout=_rpc_timeout(),
        )
        if resp.error_code:
            raise RuntimeError(resp.error_code)
        return Lease(
            lease_id=resp.lease_id,
            expert_ref=expert_ref,
            deadline_ms=float(resp.lease_deadline_unix_ms),
            granted_qps=resp.granted_qps,
        )

    def release_capacity(self, lease_id: str) -> None:
        self.stub.ReleaseCapacity(
            cei_pb2.ReleaseCapacityRequest(
                meta=wire.new_meta(self.principal_id), lease_id=lease_id
            ),
            timeout=_rpc_timeout(),
        )

    def forward_expert(
        self,
        expert_ref: ExpertRef,
        activation: ActivationBatch,
        lease_id: str | None = None,
        adapter_id: str | None = None,
        request_id: str | None = None,
        require_lease: bool = False,
    ) -> tuple[ActivationBatch, float]:
        resp = self.stub.ForwardExpert(
            cei_pb2.ForwardExpertRequest(
                meta=wire.new_meta(self.principal_id, request_id=request_id),
                expert_ref=wire.expert_ref_to_pb(expert_ref),
                lease_id=lease_id or "",
                activation=wire.activation_to_pb(activation),
                adapter_id=adapter_id or "",
            ),
            timeout=_rpc_timeout(),
        )
        if resp.error_code:
            raise RuntimeError(resp.error_code)
        return wire.activation_from_pb(resp.activation), float(resp.actual_latency_ms)

    def run_step(
        self,
        x: np.ndarray,
        domain_vec: np.ndarray,
        mode: str = "learned",
        budget: Budget | None = None,
    ) -> Outcome:
        budget = budget or Budget(allow_soft_latency=True, require_leases=True)
        resp = self.host_stub.RunStep(
            cei_internal_pb2.RunStepRequest(
                meta=wire.new_meta(self.principal_id),
                x=[float(v) for v in np.asarray(x).tolist()],
                domain_vec=[float(v) for v in np.asarray(domain_vec).tolist()],
                mode=mode,
                budget=wire.budget_to_pb(budget),
            ),
            timeout=_runstep_timeout(),
        )
        if resp.error_code:
            raise RuntimeError(resp.error_code)
        return wire.outcome_from_report_pb(resp.outcome)

    def close(self) -> None:
        if self._channel:
            self._channel.close()
