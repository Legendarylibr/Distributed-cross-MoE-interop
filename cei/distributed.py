"""Distributed simulation driver — calls HostService.RunStep on remote nodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from cei.client import NodeClient
from cei.fleet_build import orthogonal_domain_vecs
from cei.simulate import DOMAINS, SimResult
from cei.types import Budget


def _wait_nodes(addrs: dict[str, str], retries: int = 90) -> dict[str, NodeClient]:
    clients: dict[str, NodeClient] = {}
    for model_id, addr in addrs.items():
        client = NodeClient(addr, principal_id="cei-driver")
        for i in range(retries):
            try:
                client.connect()
                # Probe with a tiny run that may fail until host ready — use channel readiness
                grpc = __import__("grpc")
                grpc.channel_ready_future(client._channel).result(timeout=2)  # noqa: SLF001
                clients[model_id] = client
                print(f"connected {model_id} @ {addr}", flush=True)
                break
            except Exception as exc:
                if i == retries - 1:
                    raise RuntimeError(f"cannot reach {model_id} at {addr}: {exc}") from exc
                time.sleep(1.0)
    return clients


def run_distributed(
    steps: int = 200,
    seed: int = 0,
    mode: str = "learned",
    node_addrs: dict[str, str] | None = None,
    dim: int = 32,
    cross_domain_prob: float = 0.45,
) -> SimResult:
    node_addrs = node_addrs or {
        "moe-code": os.environ.get("CEI_NODE_CODE", "localhost:50061"),
        "moe-math": os.environ.get("CEI_NODE_MATH", "localhost:50062"),
        "moe-general": os.environ.get("CEI_NODE_GENERAL", "localhost:50063"),
    }
    clients = _wait_nodes(node_addrs)
    domain_vecs = orthogonal_domain_vecs(dim, seed=seed)
    rng = np.random.default_rng(seed)
    budget = Budget(
        max_remote_latency_ms=40.0,
        max_remote_experts=4,
        require_leases=True,
        allow_soft_latency=True,
    )
    result = SimResult(mode=mode)

    for _ in range(steps):
        host_domain = rng.choice(DOMAINS)
        host_id = f"moe-{host_domain}"
        if rng.random() < cross_domain_prob:
            task_domain = rng.choice([d for d in DOMAINS if d != host_domain])
        else:
            task_domain = host_domain
        noise = rng.normal(scale=0.3, size=(dim,))
        x = domain_vecs[task_domain] + noise
        x = x / (np.linalg.norm(x) + 1e-12)

        outcome = clients[host_id].run_step(
            x=x,
            domain_vec=domain_vecs[task_domain],
            mode=mode,
            budget=budget,
        )
        result.utilities.append(outcome.utility)
        result.rewards.append(outcome.reward)
        result.latencies.append(outcome.latency_ms)
        result.fallback_rates.append(1.0 if outcome.fallbacks else 0.0)
        remote = (
            outcome.plan is not None
            and not outcome.plan.local_only_equivalent
            and len(outcome.plan.remote_refs(host_id)) > 0
        )
        result.remote_plan_rates.append(1.0 if remote else 0.0)

    for c in clients.values():
        c.close()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CEI distributed Compose driver")
    parser.add_argument("--steps", type=int, default=int(os.environ.get("CEI_STEPS", "200")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("CEI_SEED", "0")))
    parser.add_argument(
        "--mode",
        default=os.environ.get("CEI_MODE", "learned"),
        choices=("learned", "local", "random", "heuristic"),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run_distributed(steps=args.steps, seed=args.seed, mode=args.mode)
    summary = result.summary()
    summary["mode"] = args.mode
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("CEI distributed simulation")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
