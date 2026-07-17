# CEI documentation

Guides and design notes for **Cross-Expert Interoperation**. The normative source of truth is still [../SPEC.md](../SPEC.md); these pages are the readable path into the system.

## Start here

1. [Getting started](getting-started.md) — install, simulate, Compose  
2. [Architecture](architecture.md) — who talks to whom and why  
3. [Security](security.md) — profiles, ACLs, attestation  

## Design

| Doc | Contents |
|-----|----------|
| [architecture.md](architecture.md) | Hierarchical topology, roles, leases, layer compat, HA notes |
| [learning.md](learning.md) | Reward objective, candidate generation, two-timescale learner |
| [protocol.md](protocol.md) | gRPC semantics, deadlines, fallbacks, error codes |
| [evaluation.md](evaluation.md) | Metrics, ablations A0–A3, threat checklist |

## Operate

| Doc | Contents |
|-----|----------|
| [deploy-compose.md](deploy-compose.md) | Docker Compose services, TLS, environment variables |
| [security.md](security.md) | `CEI_SECURITY_PROFILE`, allowlists, HMAC outcomes, audit log |
| [security-redteam.md](security-redteam.md) | AI-REDTEAM-ULTRA adversarial assessment |

## Contracts

Wire formats live under [`../schemas/`](../schemas/):

- `cei.proto` — public registry / router / learner / node RPCs  
- `cei_internal.proto` — policy snapshot, host `RunStep`, adapter hub  
- `*.schema.json` — JSON mirrors for key messages  

Regenerate Python stubs:

```bash
./scripts/gen_proto.sh
```

## Reading order (suggested)

**New to CEI:** getting-started → architecture §1–2 → learning §1–3 → protocol §1–3  

**Deploying a fleet:** deploy-compose → security → architecture trust boundaries → security-redteam (residual risk)  

**Implementing a new language client:** schemas + protocol → SPEC roles/contracts → evaluation conformance  
