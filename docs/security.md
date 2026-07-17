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
| Who may publish experts | `CEI_REGISTRY_PUBLISHERS` (comma-separated principals) |
| Promotion gate | New experts non-routable until `promote=true` or `CEI_AUTO_PROMOTE=1` |
| Node forward/lease ACL | `CEI_NODE_ACL_ALLOW` or `CEI_NODE_ACL_OPEN=1` (lab only) |
| Lease priority bypass | Only `CEI_PRIORITY_ADMINS` may use `priority ≥ 10` |
| Adapter Hub writes | `CEI_ADAPTER_WRITERS`; blobs carry `content_digest` (SHA-256) |
| Outcome integrity | `CEI_OUTCOME_HMAC_SECRET` + `CEI_REQUIRE_OUTCOME_ATTESTATION=1` |
| Transport | Optional TLS/mTLS via `CEI_TLS_CERT` / `KEY` / `CA`; `CEI_REQUIRE_TLS=1` refuses plaintext |
| Principal binding | Prefer peer cert identity; `CEI_TRUST_META_PRINCIPAL=0` ignores spoofable metadata when mTLS is present |
| Audit | JSON lines on logger `cei.audit` (`register_*`, `forward_*`, `lease_*`, `outcome_*`, …) |

## Principals in Compose

Reference principals used by the stock Compose file:

| Principal | Used by |
|-----------|---------|
| `node-code` / `node-math` / `node-general` | Nodes registering experts / publishing adapters |
| `host-code` / `host-math` / `host-general` | Host→peer `ForwardExpert` / leases |
| `cei-router` | Router→Registry describe |
| `cei-driver` / `e2e` | Driver / tests |

`RequestMeta.principal_id` is **client-asserted** unless mTLS peer identity is available. Treat plaintext + meta principal as **dev only**.

## Recommended production posture

1. `CEI_SECURITY_PROFILE=secure`  
2. mTLS everywhere (`CEI_TLS_*` + `CEI_REQUIRE_TLS=1`)  
3. `CEI_TRUST_META_PRINCIPAL=0`  
4. Non-empty publisher / consumer / node / adapter allowlists  
5. Shared or rotated `CEI_OUTCOME_HMAC_SECRET` with attestation required  
6. `CEI_AUTO_PROMOTE=0` and an explicit promotion step after sandbox eval  
7. Retain `cei.audit` streams  

## Lab one-liner

```bash
export CEI_SECURITY_PROFILE=lab
cei-serve registry --bind [::]:50051
```

## Related code

- [`cei/security.py`](../cei/security.py) — config, HMAC, digests, principal resolution  
- Servicers under [`cei/server/`](../cei/server/) — enforcement + audit calls  
- Canaries: [`tests/test_security.py`](../tests/test_security.py)  
