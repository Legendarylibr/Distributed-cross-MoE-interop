"""Local multi-process e2e (no Docker) for distributed CEI stack."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from cei.client import NodeClient
from cei.distributed import run_distributed
from cei.types import Budget

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _spawn(args: list[str], env: dict[str, str] | None = None, log: Path | None = None) -> subprocess.Popen:
    full_env = os.environ.copy()
    # Ensure child uses same package path
    full_env["PYTHONPATH"] = str(ROOT) + os.pathsep + full_env.get("PYTHONPATH", "")
    full_env.setdefault("CEI_SECURITY_PROFILE", "lab")
    if env:
        full_env.update(env)
    out = open(log, "w") if log else subprocess.DEVNULL  # noqa: SIM115
    return subprocess.Popen(
        [PY, "-m", "cei.server.app", *args],
        cwd=str(ROOT),
        env=full_env,
        stdout=out,
        stderr=subprocess.STDOUT,
    )


@pytest.fixture(scope="module")
def distributed_stack():
    ports = {
        "registry": _free_port(),
        "router": _free_port(),
        "learner": _free_port(),
        "code": _free_port(),
        "math": _free_port(),
        "general": _free_port(),
    }
    procs: list[subprocess.Popen] = []
    logs: list[Path] = []
    tmp = Path(tempfile.mkdtemp(prefix="cei-e2e-"))
    peers = (
        f'{{"moe-code":"127.0.0.1:{ports["code"]}",'
        f'"moe-math":"127.0.0.1:{ports["math"]}",'
        f'"moe-general":"127.0.0.1:{ports["general"]}"}}'
    )
    try:
        def start(name: str, args: list[str], env: dict[str, str] | None = None) -> None:
            log = tmp / f"{name}.log"
            logs.append(log)
            procs.append(_spawn(args, env=env, log=log))

        start("registry", ["registry", "--bind", f"127.0.0.1:{ports['registry']}"])
        start(
            "learner",
            ["learner", "--bind", f"127.0.0.1:{ports['learner']}", "--ctx-dim", "33"],
        )
        time.sleep(0.6)
        start(
            "router",
            [
                "router",
                "--bind",
                f"127.0.0.1:{ports['router']}",
                "--registry",
                f"127.0.0.1:{ports['registry']}",
                "--learner",
                f"127.0.0.1:{ports['learner']}",
            ],
        )
        time.sleep(0.4)
        for domain, key in (("code", "code"), ("math", "math"), ("general", "general")):
            start(
                f"node-{domain}",
                [
                    "node",
                    "--bind",
                    f"127.0.0.1:{ports[key]}",
                    "--domain",
                    domain,
                    "--registry",
                    f"127.0.0.1:{ports['registry']}",
                    "--router",
                    f"127.0.0.1:{ports['router']}",
                    "--learner",
                    f"127.0.0.1:{ports['learner']}",
                    "--seed",
                    "0",
                ],
                env={"CEI_PEER_ADDRS": peers},
            )

        addrs = {
            "moe-code": f"127.0.0.1:{ports['code']}",
            "moe-math": f"127.0.0.1:{ports['math']}",
            "moe-general": f"127.0.0.1:{ports['general']}",
        }
        deadline = time.time() + 60
        client = NodeClient(addrs["moe-code"], principal_id="e2e")
        last_err = ""
        while time.time() < deadline:
            # Fail fast if any process exited
            for p, log in zip(procs, logs):
                if p.poll() is not None:
                    text = log.read_text() if log.exists() else ""
                    raise RuntimeError(f"process exited early code={p.returncode}\n{text[-2000:]}")
            try:
                client.connect()
                import grpc

                grpc.channel_ready_future(client._channel).result(timeout=2)  # noqa: SLF001
                break
            except Exception as exc:
                last_err = str(exc)
                time.sleep(0.5)
        else:
            dump = "\n---\n".join(f"{l.name}:\n{l.read_text()[-800:]}" for l in logs if l.exists())
            raise RuntimeError(f"nodes did not become ready: {last_err}\n{dump}")
        client.close()
        yield addrs
    finally:
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


def test_distributed_e2e_learned(distributed_stack):
    result = run_distributed(
        steps=25,
        seed=1,
        mode="learned",
        node_addrs=distributed_stack,
        dim=32,
    )
    summary = result.summary()
    assert summary["utility_mean"] > 0.1
    assert summary["remote_plan_rate"] > 0.0
    assert summary["fallback_rate"] < 1.0


def test_budget_honored_no_soft_latency_still_runs(distributed_stack):
    client = NodeClient(distributed_stack["moe-code"], principal_id="e2e")
    x = np.ones(32) / np.sqrt(32)
    outcome = client.run_step(
        x,
        x,
        mode="local",
        budget=Budget(
            max_remote_latency_ms=100.0,
            require_leases=False,
            allow_soft_latency=False,
        ),
    )
    assert outcome.plan is not None
    assert outcome.utility is not None
    client.close()
