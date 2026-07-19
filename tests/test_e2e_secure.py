"""Multi-process e2e under the secure profile with HMAC auth + attestation.

Mirrors the Compose posture: deny-by-default ACL allowlists, signed request
metadata (CEI_AUTH_SECRET), and HMAC outcome attestation. Verifies both that
the legitimate fleet works and that unauthenticated callers are refused.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import grpc
import numpy as np
import pytest

from cei import security, wire
from cei.client import NodeClient
from cei.distributed import run_distributed
from cei.pb import cei_pb2, cei_pb2_grpc

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

AUTH_SECRET = "e2e-auth-secret"
OUTCOME_SECRET = "e2e-outcome-secret"

SECURE_ENV = {
    "CEI_SECURITY_PROFILE": "secure",
    "CEI_AUTO_PROMOTE": "1",
    "CEI_AUTH_SECRET": AUTH_SECRET,
    "CEI_REQUIRE_AUTH_TOKEN": "1",
    "CEI_OUTCOME_HMAC_SECRET": OUTCOME_SECRET,
    "CEI_REQUIRE_OUTCOME_ATTESTATION": "1",
    "CEI_REGISTRY_ALLOW_ALL": "0",
    "CEI_REGISTRY_PUBLISHERS": "node-code,node-math,node-general",
    "CEI_REGISTRY_CONSUMERS": (
        "cei-router,host-code,host-math,host-general,"
        "node-code,node-math,node-general,cei-driver"
    ),
    "CEI_NODE_ACL_OPEN": "0",
    "CEI_NODE_ACL_ALLOW": "host-code,host-math,host-general,cei-driver",
    "CEI_ADAPTER_WRITERS": "node-code,node-math,node-general",
    "CEI_PRIORITY_ADMINS": "",
}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _spawn(args: list[str], env: dict[str, str], log: Path) -> subprocess.Popen:
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = str(ROOT) + os.pathsep + full_env.get("PYTHONPATH", "")
    full_env.update(env)
    out = open(log, "w")  # noqa: SIM115
    return subprocess.Popen(
        [PY, "-m", "cei.server.app", *args],
        cwd=str(ROOT),
        env=full_env,
        stdout=out,
        stderr=subprocess.STDOUT,
    )


@pytest.fixture(scope="module")
def secure_stack():
    ports = {name: _free_port() for name in ("registry", "router", "learner", "code", "math", "general")}
    procs: list[subprocess.Popen] = []
    logs: list[Path] = []
    tmp = Path(tempfile.mkdtemp(prefix="cei-e2e-secure-"))
    peers = (
        f'{{"moe-code":"127.0.0.1:{ports["code"]}",'
        f'"moe-math":"127.0.0.1:{ports["math"]}",'
        f'"moe-general":"127.0.0.1:{ports["general"]}"}}'
    )

    # The test process itself must sign its requests too.
    saved = {k: os.environ.get(k) for k in SECURE_ENV}
    os.environ.update(SECURE_ENV)
    security.reset_config_cache()

    try:
        def start(name: str, args: list[str], extra: dict[str, str] | None = None) -> None:
            log = tmp / f"{name}.log"
            logs.append(log)
            env = dict(SECURE_ENV)
            if extra:
                env.update(extra)
            procs.append(_spawn(args, env=env, log=log))

        start("registry", ["registry", "--bind", f"127.0.0.1:{ports['registry']}"])
        start("learner", ["learner", "--bind", f"127.0.0.1:{ports['learner']}", "--ctx-dim", "33"])
        time.sleep(0.6)
        start(
            "router",
            [
                "router",
                "--bind", f"127.0.0.1:{ports['router']}",
                "--registry", f"127.0.0.1:{ports['registry']}",
                "--learner", f"127.0.0.1:{ports['learner']}",
            ],
        )
        time.sleep(0.4)
        for domain in ("code", "math", "general"):
            start(
                f"node-{domain}",
                [
                    "node",
                    "--bind", f"127.0.0.1:{ports[domain]}",
                    "--domain", domain,
                    "--registry", f"127.0.0.1:{ports['registry']}",
                    "--router", f"127.0.0.1:{ports['router']}",
                    "--learner", f"127.0.0.1:{ports['learner']}",
                    "--seed", "0",
                ],
                extra={"CEI_PEER_ADDRS": peers},
            )

        addrs = {
            "moe-code": f"127.0.0.1:{ports['code']}",
            "moe-math": f"127.0.0.1:{ports['math']}",
            "moe-general": f"127.0.0.1:{ports['general']}",
        }
        deadline = time.time() + 60
        client = NodeClient(addrs["moe-code"], principal_id="cei-driver")
        last_err = ""
        while time.time() < deadline:
            for p, log in zip(procs, logs, strict=True):
                if p.poll() is not None:
                    text = log.read_text() if log.exists() else ""
                    raise RuntimeError(f"process exited early code={p.returncode}\n{text[-2000:]}")
            try:
                client.connect()
                grpc.channel_ready_future(client._channel).result(timeout=2)  # noqa: SLF001
                break
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(0.5)
        else:
            dump = "\n---\n".join(f"{lg.name}:\n{lg.read_text()[-800:]}" for lg in logs if lg.exists())
            raise RuntimeError(f"nodes did not become ready: {last_err}\n{dump}")
        client.close()
        yield {"addrs": addrs, "ports": ports}
    finally:
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                p.kill()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        security.reset_config_cache()


def test_secure_e2e_learned_flow(secure_stack):
    result = run_distributed(
        steps=25,
        seed=1,
        mode="learned",
        node_addrs=secure_stack["addrs"],
        dim=32,
    )
    summary = result.summary()
    assert summary["utility_mean"] > 0.1
    # Marketplace works end to end under full auth.
    assert summary["remote_plan_rate"] > 0.0
    assert summary["fallback_rate"] < 1.0


def test_secure_e2e_unsigned_meta_rejected(secure_stack):
    """A caller with a valid principal name but no HMAC token is refused."""
    port = secure_stack["ports"]["registry"]
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = cei_pb2_grpc.ExpertRegistryStub(channel)
    meta = cei_pb2.RequestMeta(
        request_id="forged",
        principal_id="node-code",  # correct principal, no token
        ts_unix_ms=int(time.time() * 1000),
    )
    fp = list(np.ones(16) / 4.0)
    resp = stub.RegisterExpert(
        cei_pb2.RegisterExpertRequest(
            meta=meta,
            descriptor=cei_pb2.ExpertDescriptor(
                expert_ref=cei_pb2.ExpertRef(model_id="evil", layer_id=0, expert_id=0),
                version="1.0.0",
                dim_in=32,
                dim_out=32,
                dtype=cei_pb2.F32,
                fingerprint=fp,
                cost_flops=1,
                p50_latency_ms=1.0,
                capacity_qps=10,
                node_id="evil",
            ),
            promote=True,
        ),
        timeout=10,
    )
    assert resp.ok is False
    assert resp.error_code == "ACL_DENIED"
    channel.close()


def test_secure_e2e_forward_denied_for_unknown_principal(secure_stack):
    """Signed token for a principal outside the node ACL is still denied."""
    from cei.types import ActivationBatch, ExpertRef

    client = NodeClient(secure_stack["addrs"]["moe-math"], principal_id="mallory")
    client.connect()
    with pytest.raises(RuntimeError, match="ACL_DENIED"):
        client.forward_expert(
            expert_ref=ExpertRef("moe-math", 1, 0),
            activation=ActivationBatch(tensor=np.ones(32)),
        )
    client.close()


def test_secure_e2e_outcome_replay_rejected(secure_stack):
    """A captured, validly-signed outcome cannot be replayed to the learner."""
    port = secure_stack["ports"]["learner"]
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = cei_pb2_grpc.CombinationLearnerStub(channel)
    from cei.types import Outcome

    outcome = Outcome(
        plan_id="replay-plan",
        host_model_id="moe-code",
        reward=5.0,
        utility=5.0,
        latency_ms=1.0,
        capacity_penalty=0.0,
        tokens=1,
        context_embedding=np.zeros(33),
    )
    req = wire.outcome_to_report_pb(outcome, wire.new_meta("host-code"))
    first = stub.ReportOutcome(req, timeout=10)
    assert first.ok is True
    replay = stub.ReportOutcome(req, timeout=10)
    assert replay.ok is False
    channel.close()
