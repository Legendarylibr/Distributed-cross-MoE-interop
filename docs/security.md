# Security (operators)

How the reference stack enforces marketplace trust. Threat analysis: [security-redteam.md](security-redteam.md). Architecture trust boundaries: [architecture.md](architecture.md).

## Profiles

| `CEI_SECURITY_PROFILE` | Intent |
|------------------------|--------|
| **`secure`** (default for `cei-serve`) | Deny-by-default ACLs; empty publisher list blocks registration; configure allowlists |
| **`lab`** | Local experiments: registry `allow_all`, node ACL open, auto-promote; still logs audits |

Compose ships **`secure`** with explicit allowlists and an HMAC secret (see `docker-compose.yml` `x-cei-security`). Pytest e2e uses **`lab`** so subprocesses can register freely.

## Controls (checklist)

| Control | Env / behavior |
|---------|----------------|
| Registry describe ACL | `CEI_REGISTRY_ALLOW_ALL=0` + `CEI_REGISTRY_CONSUMERS` |
| Who may publish experts | `CEI_REGISTRY_PUBLISHERS` (comma-separated principals); heartbeats are publisher-gated too |
| Registry ownership | In `secure`, only the registering principal may re-register (force) or deregister an expert; heartbeats only refresh experts owned by the reporting `node_id` |
| Promotion gate | New experts non-routable until `promote=true` or `CEI_AUTO_PROMOTE=1` |
| Node forward/lease ACL | `CEI_NODE_ACL_ALLOW` or `CEI_NODE_ACL_OPEN=1` (lab only); `RunStep` uses the same ACL |
| Lease binding | Leases are bound to `(expert_ref, principal)`; forwards with mismatched leases fail `LEASE_MISMATCH`; only the grantee may release; expired leases are purged |
| Lease priority bypass | Only `CEI_PRIORITY_ADMINS` may use `priority ≥ 10` |
| Adapter Hub writes | `CEI_ADAPTER_WRITERS`; blobs carry `content_digest` (SHA-256); `CEI_REQUIRE_ADAPTER_DIGEST=1` (secure default) rejects undigested uploads; matrices are shape/dim/finite-validated |
| Outcome integrity | `CEI_OUTCOME_HMAC_SECRET` + `CEI_REQUIRE_OUTCOME_ATTESTATION=1`; attestation binds `meta.request_id`, learner keeps a replay cache (duplicate reports rejected) |
| Request authentication | `CEI_AUTH_SECRET` + `CEI_REQUIRE_AUTH_TOKEN=1`: `RequestMeta.auth_token` = HMAC-SHA256 over `principal|request_id|ts`, freshness window `CEI_AUTH_MAX_SKEW_MS` (default 120 s) |
| Plan TTL | Plans carry `issued_unix_ms`; hosts refuse remote forwards on expired plans (`PLAN_EXPIRED` fallback) |
| Input validation | Wire tensors (shape/size/finite), descriptors (ids/dims/fingerprint/capacity), NN queries, lease TTL/QPS all validated at the boundary |
| Transport | Optional TLS/mTLS via `CEI_TLS_CERT` / `KEY` / `CA`; `CEI_REQUIRE_TLS=1` refuses plaintext |
| Principal binding | Peer cert identity > HMAC auth token > (dev-only) bare meta principal; `CEI_TRUST_META_PRINCIPAL=0` disables the bare fallback |
| RPC deadlines | All client RPCs carry deadlines (`CEI_RPC_TIMEOUT_S`, `CEI_RUNSTEP_TIMEOUT_S`) |
| Audit | JSON lines on logger `cei.audit` (`register_*`, `forward_*`, `lease_*`, `outcome_*`, `runstep_*`, …) |

## Principals in Compose

Reference principals used by the stock Compose file:

| Principal | Used by |
|-----------|---------|
| `node-code` / `node-math` / `node-general` | Nodes registering experts / publishing adapters |
| `host-code` / `host-math` / `host-general` | Host→peer `ForwardExpert` / leases |
| `cei-router` | Router→Registry describe |
| `cei-driver` / `e2e` | Driver / tests |

`RequestMeta.principal_id` is **client-asserted** unless mTLS peer identity or an HMAC `auth_token` (`CEI_AUTH_SECRET`) proves it. Treat plaintext + bare meta principal as **dev only**.

## Recommended production posture

1. `CEI_SECURITY_PROFILE=secure`  
2. mTLS everywhere (`CEI_TLS_*` + `CEI_REQUIRE_TLS=1`)  
3. `CEI_AUTH_SECRET` set fleet-wide + `CEI_REQUIRE_AUTH_TOKEN=1` (defense in depth alongside mTLS)  
4. `CEI_TRUST_META_PRINCIPAL=0`  
5. Non-empty publisher / consumer / node / adapter allowlists  
6. Shared or rotated `CEI_OUTCOME_HMAC_SECRET` with attestation required  
7. `CEI_AUTO_PROMOTE=0` and an explicit promotion step after sandbox eval  
8. Retain `cei.audit` streams  

## Lab one-liner

```bash
export CEI_SECURITY_PROFILE=lab
cei-serve registry --bind [::]:50051
```

## Related code

- [`cei/security.py`](../cei/security.py) — config, HMAC request auth, attestation, replay cache, digests, principal resolution  
- Servicers under [`cei/server/`](../cei/server/) — enforcement + audit calls  
- Canaries: [`tests/test_security.py`](../tests/test_security.py), [`tests/test_auth.py`](../tests/test_auth.py), [`tests/test_validation.py`](../tests/test_validation.py), [`tests/test_leases_ownership.py`](../tests/test_leases_ownership.py), [`tests/test_secure_surface.py`](../tests/test_secure_surface.py), [`tests/test_tls.py`](../tests/test_tls.py), [`tests/test_e2e_secure.py`](../tests/test_e2e_secure.py)  
