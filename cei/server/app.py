"""cei-serve — start a CEI gRPC role (registry | router | learner | node)."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from concurrent import futures

import grpc
import numpy as np

from cei.adapters import AdapterHub
from cei.client import LearnerClient, RegistryClient, RouterClient
from cei.fleet_build import build_domain_host, peer_addrs_from_env
from cei.learner import ContextualBanditLearner
from cei.pb import cei_internal_pb2, cei_internal_pb2_grpc, cei_pb2, cei_pb2_grpc
from cei.server.adapter_hub_servicer import AdapterHubServicer
from cei.server.learner_servicer import LearnerInternalServicer, LearnerServicer
from cei.server.node_servicer import HostServicer, NodeServicer, register_all_experts
from cei.server.registry_servicer import RegistryServicer
from cei.server.router_servicer import RouterServicer
from cei.tlsutil import add_secure_or_insecure_port, make_channel
from cei.security import adapter_digest, assert_transport_ok, get_config
from cei import wire


def _add_health(server: grpc.Server) -> None:
    try:
        from grpc_health.v1 import health, health_pb2, health_pb2_grpc

        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_servicer.set("cei", health_pb2.HealthCheckResponse.SERVING)
    except Exception:
        pass


def serve_registry(bind: str) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    cei_pb2_grpc.add_ExpertRegistryServicer_to_server(RegistryServicer(), server)
    _add_health(server)
    add_secure_or_insecure_port(server, bind)
    server.start()
    print(f"registry listening on {bind}", flush=True)
    _wait(server)


def serve_learner(bind: str, ctx_dim: int) -> None:
    learner = ContextualBanditLearner(ctx_dim=ctx_dim, batch_size=32)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    cei_pb2_grpc.add_CombinationLearnerServicer_to_server(LearnerServicer(learner), server)
    cei_internal_pb2_grpc.add_LearnerInternalServicer_to_server(
        LearnerInternalServicer(learner), server
    )
    _add_health(server)
    add_secure_or_insecure_port(server, bind)
    server.start()
    print(f"learner listening on {bind}", flush=True)
    _wait(server)


def serve_router(bind: str, registry_addr: str, learner_addr: str) -> None:
    servicer = RouterServicer(registry_addr, learner_addr)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    cei_pb2_grpc.add_CombinationRouterServicer_to_server(servicer, server)
    _add_health(server)
    add_secure_or_insecure_port(server, bind)
    server.start()
    print(f"router listening on {bind}", flush=True)
    _wait(server)


def serve_adapter_hub(bind: str) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    cei_internal_pb2_grpc.add_AdapterHubServicer_to_server(AdapterHubServicer(), server)
    _add_health(server)
    add_secure_or_insecure_port(server, bind)
    server.start()
    print(f"adapter-hub listening on {bind}", flush=True)
    _wait(server)


def serve_node(
    bind: str,
    domain: str,
    registry_addr: str,
    router_addr: str,
    learner_addr: str,
    dim: int,
    seed: int,
    adapter_hub_addr: str | None = None,
) -> None:
    host, node, _ = build_domain_host(domain=domain, dim=dim, seed=seed)
    registry = RegistryClient(registry_addr, principal_id=f"node-{domain}")
    router = RouterClient(router_addr, principal_id=f"node-{domain}")
    learner = LearnerClient(learner_addr, principal_id=f"node-{domain}")
    _wait_for_registry(registry, retries=60)
    registry.connect()
    router.connect()
    learner.connect()
    register_all_experts(registry, node)

    hub_addr = adapter_hub_addr or os.environ.get("CEI_ADAPTER_HUB_ADDR") or ""
    if hub_addr.strip() and node.adapter_hub:
        try:
            _publish_adapters(hub_addr, node.adapter_hub, principal=f"node-{domain}")
        except Exception as exc:
            print(f"adapter publish skipped: {exc}", flush=True)

    stop = threading.Event()

    def heartbeat_loop() -> None:
        while not stop.is_set():
            try:
                registry.heartbeat(
                    node.node_id,
                    None,
                    capacity_qps=node.get_capacity_snapshot(),
                    load_qps=node.get_load_snapshot(),
                )
            except Exception as exc:
                print(f"heartbeat error: {exc}", flush=True)
            stop.wait(5.0)

    threading.Thread(target=heartbeat_loop, daemon=True).start()

    peers = peer_addrs_from_env()
    peers.setdefault(
        host.model_id,
        bind.replace("[::]:", "localhost:").replace("0.0.0.0:", "localhost:"),
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16))
    cei_pb2_grpc.add_ExpertNodeServicer_to_server(NodeServicer(node), server)
    cei_internal_pb2_grpc.add_HostServiceServicer_to_server(
        HostServicer(host, peers, router, learner, principal_id=f"host-{domain}"),
        server,
    )
    _add_health(server)
    add_secure_or_insecure_port(server, bind)
    server.start()
    print(f"node-{domain} listening on {bind} peers={peers}", flush=True)
    try:
        _wait(server)
    finally:
        stop.set()


def _publish_adapters(hub_addr: str, hub: AdapterHub, principal: str) -> None:
    channel = make_channel(hub_addr)
    stub = cei_internal_pb2_grpc.AdapterHubStub(channel)
    for adapter in hub._adapters.values():  # noqa: SLF001
        w_in = np.asarray(adapter.w_in, dtype=np.float64).tobytes()
        w_out = np.asarray(adapter.w_out, dtype=np.float64).tobytes()
        digest = adapter_digest(w_in, w_out)
        stub.UpsertAdapter(
            cei_internal_pb2.UpsertAdapterRequest(
                meta=wire.new_meta(principal),
                adapter=cei_internal_pb2.AdapterBlob(
                    adapter_id=adapter.adapter_id,
                    dim_in_host=adapter.dim_in_host,
                    dim_in_remote=adapter.dim_in_remote,
                    dim_out_remote=adapter.dim_out_remote,
                    dim_out_host=adapter.dim_out_host,
                    w_in=w_in,
                    w_out=w_out,
                    w_in_shape=list(adapter.w_in.shape),
                    w_out_shape=list(adapter.w_out.shape),
                    content_digest=digest,
                ),
            )
        )
    channel.close()


def _wait_for_registry(registry: RegistryClient, retries: int = 30) -> None:
    for _ in range(retries):
        try:
            registry.connect()
            registry.stub.DescribeExperts(
                cei_pb2.DescribeExpertsRequest(
                    meta=wire.new_meta(),
                    explicit=cei_pb2.ExplicitRefs(expert_refs=[]),
                )
            )
            return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("registry not reachable")


def _wait(server: grpc.Server) -> None:
    stop = threading.Event()

    def _handle(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    while not stop.is_set():
        time.sleep(0.5)
    server.stop(grace=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a CEI gRPC role")
    parser.add_argument(
        "role",
        nargs="?",
        default=os.environ.get("CEI_ROLE", "registry"),
        choices=("registry", "router", "learner", "node", "adapter-hub"),
    )
    parser.add_argument("--bind", default=os.environ.get("CEI_BIND", "[::]:50051"))
    parser.add_argument("--registry", default=os.environ.get("CEI_REGISTRY_ADDR", "localhost:50051"))
    parser.add_argument("--router", default=os.environ.get("CEI_ROUTER_ADDR", "localhost:50052"))
    parser.add_argument("--learner", default=os.environ.get("CEI_LEARNER_ADDR", "localhost:50053"))
    parser.add_argument(
        "--adapter-hub",
        default=os.environ.get("CEI_ADAPTER_HUB_ADDR", ""),
        help="Adapter hub address; empty skips publish",
    )
    parser.add_argument("--domain", default=os.environ.get("CEI_DOMAIN", "code"))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("CEI_DIM", "32")))
    parser.add_argument("--ctx-dim", type=int, default=int(os.environ.get("CEI_CTX_DIM", "33")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("CEI_SEED", "0")))
    args = parser.parse_args(argv)

    assert_transport_ok()
    cfg = get_config()
    print(f"cei-serve security_profile={cfg.profile}", flush=True)

    if args.role == "registry":
        serve_registry(args.bind)
    elif args.role == "learner":
        serve_learner(args.bind, args.ctx_dim)
    elif args.role == "router":
        serve_router(args.bind, args.registry, args.learner)
    elif args.role == "adapter-hub":
        serve_adapter_hub(args.bind)
    elif args.role == "node":
        serve_node(
            args.bind,
            args.domain,
            args.registry,
            args.router,
            args.learner,
            args.dim,
            args.seed,
            adapter_hub_addr=args.adapter_hub,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
