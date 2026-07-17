# Getting started

This guide gets you from a clean checkout to a successful simulation (in-process and optional multi-node).

## Prerequisites

- Python **3.10+**
- Optional: Docker + Compose v2 for the distributed demo
- Optional: `grpcio-tools` (pulled with `pip install -e ".[dev]"`) to regenerate stubs

## 1. Install

```bash
git clone https://github.com/Legendarylibr/Distributed-cross-MoE-interop.git
cd Distributed-cross-MoE-interop

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify:

```bash
cei-simulate --help
cei-serve --help
pytest -q
```

## 2. In-process simulation

The fastest way to understand CEI is the in-memory fleet (`code` / `math` / `general` domains):

```bash
# Full ablation table (local / random / heuristic / learned)
cei-simulate --mode ablate --steps 400

# Marketplace + learner only
cei-simulate --mode learned --steps 200 --seed 0
```

You should see per-mode summaries such as mean utility, reward, latency percentiles, fallback rate, and remote-plan rate.

**What is happening:** each “host” proposes combination plans via a shared registry + router + learner, may substitute remote experts at up to \(m\) layers, measures a toy utility, and feeds `ReportOutcome` back so the bandit can improve.

Useful flags (see `cei-simulate --help`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `learned` | `local` \| `random` \| `heuristic` \| `learned` \| `ablate` |
| `--steps` | (CLI default) | Sequences / requests to run |
| `--seed` | `0` | RNG seed for fleet + traffic |

## 3. Distributed Compose demo

```bash
docker compose up --build -d \
  registry learner adapter-hub router \
  node-code node-math node-general

docker compose run --rm driver
docker compose down
```

The driver calls each host’s internal `RunStep` RPC and prints JSON metrics. Details and TLS: [deploy-compose.md](deploy-compose.md).

### Without Docker

Start roles with `cei-serve` (see [../README.md](../README.md) § Distributed). For a laptop smoke test, set:

```bash
export CEI_SECURITY_PROFILE=lab
```

so ACLs and publisher allowlists do not block the first register/forward. Prefer `secure` + explicit allowlists when the network is shared ([security.md](security.md)).

## 4. Tests

```bash
pytest -q                 # unit + gRPC + security canaries + local e2e
pytest -q tests/test_security.py
pytest -q tests/test_e2e_distributed.py
```

E2E spins ephemeral ports and child `cei-serve` processes (no Docker required).

## 5. Next reading

- [architecture.md](architecture.md) — topology and lease ownership  
- [learning.md](learning.md) — how plans are scored and updated  
- [security.md](security.md) — before exposing ports beyond localhost  
- [../SPEC.md](../SPEC.md) — normative wording for conformance  

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `ACL_DENIED` on register/forward | Secure profile without your principal on the allowlist |
| `OUTCOME_REJECTED` | Attestation required but HMAC secret mismatch / missing |
| Experts never selected remotely | Not promoted (`CEI_AUTO_PROMOTE=0` and `promote=false`) or heartbeat stale |
| Compose nodes idle | Wait for healthchecks; check `docker compose logs registry node-code` |
| Import / stub errors after editing `.proto` | Re-run `./scripts/gen_proto.sh` and reinstall editable package |
