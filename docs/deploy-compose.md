# Deploy CEI with Docker Compose

Distributed CEI roles as containers on a single bridge network (`cei-net`).

Also see: [getting-started.md](getting-started.md) · [security.md](security.md) · [architecture.md](architecture.md)

Transport is **plaintext gRPC** for local Compose. Security defaults are **`CEI_SECURITY_PROFILE=secure`** with explicit ACL allowlists and HMAC outcome attestation (`x-cei-security` in [`docker-compose.yml`](../docker-compose.yml)). Add mTLS for anything beyond a trusted lab network.

## Services

| Service | Port | Role |
|---------|------|------|
| `registry` | 50051 | Expert catalog + fingerprint NN |
| `learner` | 50053 | Bandit updates + policy snapshots |
| `adapter-hub` | 50054 | Adapter blob store |
| `router` | 50052 | `ProposeCombinations` (cached policy) |
| `node-code` | 50061 | MoE host+experts (code) |
| `node-math` | 50062 | MoE host+experts (math) |
| `node-general` | 50063 | MoE host+experts (general) |
| `driver` | — | One-shot distributed sim (profile `driver`) |

## Quick start

```bash
docker compose up --build -d \
  registry learner adapter-hub router \
  node-code node-math node-general
# wait ~15s for registration/heartbeats
docker compose run --rm driver
docker compose down
```

Or:

```bash
docker compose --profile driver up --build --abort-on-container-exit
```

Copy [`deploy/env.example`](../deploy/env.example) if you run roles outside Compose.

## Point at a remote Docker host

```bash
export DOCKER_HOST=ssh://user@your-vm
docker compose up --build -d
docker compose run --rm driver
```

## TLS (optional)

```bash
chmod +x scripts/gen_dev_certs.sh
./scripts/gen_dev_certs.sh deploy/certs
export CEI_TLS_CERT=$PWD/deploy/certs/server.crt
export CEI_TLS_KEY=$PWD/deploy/certs/server.key
export CEI_TLS_CA=$PWD/deploy/certs/ca.crt
export CEI_TLS_SERVER_NAME=cei.local
# then start cei-serve / compose with those env vars on every service
```

When `CEI_TLS_CERT` + `CEI_TLS_KEY` are set, servers use secure ports and clients use `secure_channel`. Set `CEI_TLS_CA` to require mutual TLS. Set `CEI_REQUIRE_TLS=1` to refuse plaintext startups.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `CEI_PEER_ADDRS` | JSON `model_id → host:port` for cross-node `ForwardExpert` |
| `CEI_REGISTRY_ADDR` / `CEI_ROUTER_ADDR` / `CEI_LEARNER_ADDR` / `CEI_ADAPTER_HUB_ADDR` | Control-plane addresses |
| `CEI_MODE` / `CEI_STEPS` | Driver simulation mode and length |
| `CEI_LAYER_COMPAT` | Layer matching (`exact_layer` default) |
| `CEI_SECURITY_PROFILE` | `secure` or `lab` |
| `CEI_OUTCOME_HMAC_SECRET` / `CEI_REQUIRE_OUTCOME_ATTESTATION` | Attest `ReportOutcome` |
| `CEI_REGISTRY_PUBLISHERS` / `CEI_REGISTRY_CONSUMERS` | Registry write/read principals |
| `CEI_NODE_ACL_ALLOW` / `CEI_ADAPTER_WRITERS` | Node and hub ACLs |
| `CEI_AUTO_PROMOTE` | Auto-mark experts routable on register |
| `CEI_REQUIRE_TLS` | Refuse start without TLS certs |

Full operator guide: [security.md](security.md). Residual risk: [security-redteam.md](security-redteam.md).

## Local (non-Docker) multi-process smoke

```bash
export CEI_SECURITY_PROFILE=lab

cei-serve registry --bind [::]:50051 &
cei-serve learner --bind [::]:50053 &
cei-serve router --bind [::]:50052 --registry localhost:50051 --learner localhost:50053 &

CEI_PEER_ADDRS='{"moe-code":"localhost:50061","moe-math":"localhost:50062","moe-general":"localhost:50063"}' \
  cei-serve node --bind [::]:50061 --domain code \
  --registry localhost:50051 --router localhost:50052 --learner localhost:50053 &
# similarly math:50062 general:50063

cei-simulate-distributed --steps 50 --mode learned
```
