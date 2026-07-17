# Deploy CEI with Docker Compose

Distributed CEI roles as containers on a single bridge network (`cei-net`). Transport is **plaintext gRPC** (dev profile). Production should add mTLS per [docs/protocol.md](protocol.md).

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
docker compose up --build -d registry learner router node-code node-math node-general
# wait ~15s for registration/heartbeats
docker compose run --rm driver
docker compose down
```

Or:

```bash
docker compose --profile driver up --build --abort-on-container-exit
```

## Point at a remote Docker host

```bash
export DOCKER_HOST=ssh://user@your-vm
docker compose up --build -d
docker compose run --rm driver
```

Any host that can run Compose works (local Docker Desktop, a cloud VM, etc.).

## TLS (optional)

Dev plaintext is the Compose default. Compose now runs **`CEI_SECURITY_PROFILE=secure`** with explicit publisher/consumer/node ACLs and HMAC outcome attestation (see `x-cei-security` in `docker-compose.yml`).

For TLS/mTLS:

```bash
chmod +x scripts/gen_dev_certs.sh
./scripts/gen_dev_certs.sh deploy/certs
export CEI_TLS_CERT=$PWD/deploy/certs/server.crt
export CEI_TLS_KEY=$PWD/deploy/certs/server.key
export CEI_TLS_CA=$PWD/deploy/certs/ca.crt
export CEI_TLS_SERVER_NAME=cei.local
# then start cei-serve / compose with those env vars on every service
```

When `CEI_TLS_CERT` + `CEI_TLS_KEY` are set, servers use secure ports and clients use `secure_channel`. Set `CEI_TLS_CA` to require mutual TLS.

- `CEI_PEER_ADDRS` — JSON map `model_id → host:port` for cross-node `ForwardExpert`
- `CEI_REGISTRY_ADDR` / `CEI_ROUTER_ADDR` / `CEI_LEARNER_ADDR`
- `CEI_MODE` / `CEI_STEPS` for the driver
- `CEI_SECURITY_PROFILE` — `secure` (default for serve) or `lab` (open ACLs for local experiments)
- `CEI_OUTCOME_HMAC_SECRET` / `CEI_REQUIRE_OUTCOME_ATTESTATION` — attest `ReportOutcome`
- `CEI_REGISTRY_PUBLISHERS` / `CEI_REGISTRY_CONSUMERS` / `CEI_NODE_ACL_ALLOW` / `CEI_ADAPTER_WRITERS`
- `CEI_AUTO_PROMOTE` — if `0`, experts stay non-routable until `promote=true` on register
- `CEI_REQUIRE_TLS` — refuse to start without `CEI_TLS_CERT`/`CEI_TLS_KEY`

See [docs/security-redteam.md](security-redteam.md) for the threat model these controls address.

## Local (non-Docker) multi-process smoke

```bash
cei-serve registry --bind [::]:50051 &
cei-serve learner --bind [::]:50053 &
cei-serve router --bind [::]:50052 --registry localhost:50051 --learner localhost:50053 &
CEI_PEER_ADDRS='{"moe-code":"localhost:50061","moe-math":"localhost:50062","moe-general":"localhost:50063"}' \
  cei-serve node --bind [::]:50061 --domain code --registry localhost:50051 --router localhost:50052 --learner localhost:50053 &
# similarly math:50062 general:50063
cei-simulate-distributed --steps 50 --mode learned
```
