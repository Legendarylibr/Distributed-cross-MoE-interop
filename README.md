# Cross-Expert Interoperation (CEI)

Hierarchical MoE **learning + protocol** specification: local expert banks per model, a cross-model marketplace, and a combination learner that searches layer×expert bindings under latency and capacity budgets.

## Start here

**[SPEC.md](SPEC.md)** — normative specification (problem, formal model, roles, objective, contracts, evaluation, threats, conformance).

### Reference implementation

Python package [`cei/`](cei/) — in-process simulator **and** networked gRPC services for Docker Compose.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cei-simulate --mode ablate --steps 400
pytest -q
```

### Distributed nodes (Docker Compose)

```bash
docker compose up --build -d registry learner router node-code node-math node-general
docker compose run --rm driver
docker compose down
```

Details: [docs/deploy-compose.md](docs/deploy-compose.md). Optional TLS via `scripts/gen_dev_certs.sh` + `CEI_TLS_*` env vars.

| Mode | Meaning |
|------|---------|
| `local` | A0 — no marketplace |
| `random` | A1 — random remote compatible plans |
| `heuristic` | A2 — top scored plan without treating as bandit explore |
| `learned` | A3 — full ε-greedy / bandit learner |
| `ablate` | Run all four and print comparison table |

### Companion docs

| Doc | Contents |
|-----|----------|
| [docs/architecture.md](docs/architecture.md) | Topology, trust boundaries, role rationale |
| [docs/learning.md](docs/learning.md) | Two-timescale learner, candidate generation, pseudocode |
| [docs/protocol.md](docs/protocol.md) | RPC semantics, fallbacks, security |
| [docs/evaluation.md](docs/evaluation.md) | Metrics, ablations, threats, conformance tests |
| [docs/deploy-compose.md](docs/deploy-compose.md) | Docker Compose multi-node deploy |

### Schemas

| File | Contents |
|------|----------|
| [schemas/cei.proto](schemas/cei.proto) | gRPC / Protobuf services and messages |
| [schemas/*.schema.json](schemas/) | JSON Schema mirrors for key contracts |

## Locked design defaults

- **Topology:** hierarchical (local + marketplace), not a single sharded MoE
- **Learning:** online sampling over scored plans + offline contextual bandit / neural policy
- **Search:** swap / insert / drop at up to \(m=2\) layers
- **Transport:** gRPC + mTLS; optional RDMA for activations; timeout → local fallback

## Out of scope (v0.1)

Training-framework plugins, K8s operators, and shipping model weights or datasets.
