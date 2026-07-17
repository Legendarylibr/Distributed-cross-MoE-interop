# Cross-Expert Interoperation (CEI)

**CEI** lets a *fleet* of specialized Mixture-of-Experts (MoE) models keep their own expert banks, publish experts to a shared marketplace, and **learn which layer×expert combinations** are worth using under latency and capacity budgets.

This repository is both:

1. A **normative specification** ([SPEC.md](SPEC.md)) for the learning problem and interoperation protocol  
2. A **Python reference implementation** (`cei/`) — in-process simulator plus networked gRPC roles you can run with Docker Compose

**Repo:** [Legendarylibr/Distributed-cross-MoE-interop](https://github.com/Legendarylibr/Distributed-cross-MoE-interop)

---

## Why this exists

Organizations often run several MoE models (code, math, multimodal, vertical domains). Each model has its own experts and local top‑*k* router. Specialist capacity is trapped behind model boundaries:

- A code MoE cannot cheaply borrow a strong math expert at a mid-depth layer  
- Operators have no systematic way to discover *which* remote experts help, *where* to place them in the stack, and *when* the latency cost is worth it  

Offline weight merges (BTX / MoErging) produce one checkpoint but lose independent release cycles. A single sharded mega-MoE forces one serving identity. **CEI is hierarchical:** local banks stay first-class; the marketplace is opt-in composition, not a forced merge.

```
  MoE A (code) ──ForwardExpert──► MoE B expert @ layer ℓ
       │                              ▲
       └── ProposeCombinations ──► Router ◄── Registry (fingerprints)
                                       ▲
                                   Learner (bandit / policy)
```

---

## What you get in this repo

| Piece | Role |
|-------|------|
| [SPEC.md](SPEC.md) | Normative problem statement, formal model, roles, contracts, threats |
| [`cei/`](cei/) | Reference simulator + gRPC services (`cei-simulate`, `cei-serve`) |
| [`schemas/`](schemas/) | Protobuf + JSON Schema wire contracts |
| [`docs/`](docs/) | Architecture, learning, protocol, deploy, security |
| [`docker-compose.yml`](docker-compose.yml) | Multi-node fleet: registry, router, learner, adapter-hub, 3 domain nodes |

### Core roles

| Role | Job |
|------|-----|
| **Expert Registry** | Catalog, heartbeats, fingerprint nearest-neighbor, quotas, ACLs |
| **Combination Router** | Builds candidate plans; scores via cached learner policy |
| **Combination Learner** | Updates from `ReportOutcome`; publishes policy snapshots |
| **MoE Host / Node** | Local MoE + `ForwardExpert` / leases for remotes |
| **Adapter Hub** | Dim/domain projection weights (`W_in` / `W_out`) |

---

## Quick start (in-process)

Requires Python ≥ 3.10.

```bash
git clone https://github.com/Legendarylibr/Distributed-cross-MoE-interop.git
cd Distributed-cross-MoE-interop

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Compare local-only vs random / heuristic / learned marketplace
cei-simulate --mode ablate --steps 400

# Single mode
cei-simulate --mode learned --steps 200

pytest -q
```

### Simulation modes

| Mode | Ablation | Meaning |
|------|----------|---------|
| `local` | A0 | No marketplace — host uses only its own experts |
| `random` | A1 | Sample random compatible remote plans (stress marketplace) |
| `heuristic` | A2 | Pick top-scored plan without bandit exploration |
| `learned` | A3 | Full contextual bandit / ε-greedy combination learner |
| `ablate` | — | Run A0–A3 and print a comparison table |

The in-process path builds a toy three-domain fleet (`code`, `math`, `general`), registers experts, proposes combination plans, optionally forwards to “remote” experts in-memory, and reports outcomes to the learner.

---

## Distributed stack (Docker Compose)

Runs real gRPC processes on one bridge network:

```bash
docker compose up --build -d \
  registry learner adapter-hub router \
  node-code node-math node-general

# wait ~15s for register + heartbeats
docker compose run --rm driver

docker compose down
```

| Service | Port | Purpose |
|---------|------|---------|
| `registry` | 50051 | Expert catalog + NN |
| `router` | 50052 | `ProposeCombinations` |
| `learner` | 50053 | Outcomes + policy snapshots |
| `adapter-hub` | 50054 | Adapter blobs |
| `node-code` / `math` / `general` | 50061–63 | MoE hosts + expert nodes |
| `driver` | — | One-shot distributed sim (`profile: driver`) |

Compose defaults to a **secure** security profile with explicit ACL allowlists and HMAC outcome attestation (see [docs/deploy-compose.md](docs/deploy-compose.md) and [docs/security.md](docs/security.md)). Transport is plaintext gRPC for local Compose; enable mTLS with `CEI_TLS_*` for production.

### Local multi-process (no Docker)

```bash
cei-serve registry --bind [::]:50051 &
cei-serve learner --bind [::]:50053 &
cei-serve router --bind [::]:50052 \
  --registry localhost:50051 --learner localhost:50053 &

CEI_SECURITY_PROFILE=lab \
CEI_PEER_ADDRS='{"moe-code":"localhost:50061","moe-math":"localhost:50062","moe-general":"localhost:50063"}' \
  cei-serve node --bind [::]:50061 --domain code \
  --registry localhost:50051 --router localhost:50052 --learner localhost:50053 &
# similarly for math:50062 and general:50063

cei-simulate-distributed --steps 50 --mode learned
```

Use `CEI_SECURITY_PROFILE=lab` for open local experiments; use `secure` + allowlists for anything shared.

---

## How a request flows

1. **Host** runs local MoE gates → builds local top‑*k* and a context embedding \(\phi(x)\).  
2. Host calls **Router** `ProposeCombinations` (budget: max remote experts, latency, leases).  
3. Router queries **Registry** (fingerprint NN) and scores candidates with a **cached policy** from the **Learner**.  
4. Host samples a `CombinationPlan`, takes **leases** on remotes it will call, then **`ForwardExpert`** (activations) to peer nodes.  
5. On success or fallback-to-local, host **`ReportOutcome`** (reward, latency, plan snapshot) → Learner updates.  
6. On timeout / ACL deny / capacity exhausted → **local fallback** (availability over remote quality).

Leases are **host-owned** (not pre-leased by the router). Layer compatibility defaults to `exact_layer` (`CEI_LAYER_COMPAT`).

---

## Package layout

```
cei/
  types.py, registry.py, router.py, learner.py, node.py, host.py
  security.py          # ACLs, attestation, audit
  simulate.py          # in-process fleet
  server/app.py        # cei-serve entrypoints
  client/              # gRPC clients
  pb/                  # generated stubs (from schemas/)
schemas/               # cei.proto, JSON schemas
docs/                  # design docs (see below)
tests/                 # unit, gRPC, security canaries, e2e
```

---

## Documentation map

| Doc | Read when you want… |
|-----|---------------------|
| [docs/README.md](docs/README.md) | Index of all docs |
| [docs/getting-started.md](docs/getting-started.md) | Install, first sim, first Compose run |
| [docs/architecture.md](docs/architecture.md) | Topology, roles, leases, trust boundaries |
| [docs/learning.md](docs/learning.md) | Objective, candidate search, bandit update |
| [docs/protocol.md](docs/protocol.md) | RPC semantics, timeouts, fallbacks |
| [docs/evaluation.md](docs/evaluation.md) | Metrics, ablations A0–A3, conformance |
| [docs/deploy-compose.md](docs/deploy-compose.md) | Compose, TLS, env vars |
| [docs/security.md](docs/security.md) | Profiles, ACLs, attestation, audit |
| [docs/security-redteam.md](docs/security-redteam.md) | Adversarial assessment (threat model) |
| [SPEC.md](SPEC.md) | Full normative specification |

---

## Locked design defaults (v0.1)

- **Topology:** hierarchical (local + marketplace), not one sharded MoE  
- **Learning:** online plan sampling + offline contextual bandit / neural policy  
- **Search:** swap / insert / drop at up to \(m=2\) layers  
- **Transport:** gRPC (+ optional mTLS); timeout → local fallback  
- **Security:** deny-by-default ACLs on distributed roles; promotion gate; optional outcome HMAC  

---

## Out of scope (v0.1)

Training-framework plugins, Kubernetes operators, shipping production model weights or datasets, and full weight-sandbox attestation (promotion flag only in the reference stack).

---

## License

See [LICENSE](LICENSE).
