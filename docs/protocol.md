# Protocol — Cross-Expert Interoperation RPC

Normative message semantics for CEI. Parent: [SPEC.md](../SPEC.md).  
Schemas: [schemas/cei.proto](../schemas/cei.proto), JSON Schema files under [schemas/](../schemas/).

---

## 1. Transport profile

| Profile | Requirement |
|---------|-------------|
| **baseline** (MUST) | gRPC over HTTP/2 with protobuf payloads; mTLS |
| **rdma** (MAY) | Device-buffer references in `ActivationBatch.buffer_ref`; control plane still gRPC |
| **json** (MAY) | JSON encoding of the same messages for debugging; not for hot activation path |

Timeouts: clients MUST set deadlines. Default `ForwardExpert` deadline = `budget.max_remote_latency_ms` (or 50ms if unset).

---

## 2. Common headers

All RPCs carry:

| Field | Required | Description |
|-------|----------|-------------|
| `request_id` | YES | UUID; idempotency key |
| `principal_id` | YES | Authenticated caller (from mTLS cert SAN or token) |
| `trace_id` | NO | Distributed tracing |
| `cei_version` | YES | e.g. `"0.1.0"` |

---

## 3. Registry service

### 3.1 `RegisterExpert`

**Request:** `ExpertDescriptor` (+ optional replace-if-version).  
**Response:** `{ ok, registry_version, warnings[] }`.

**Semantics:**

- Upserts by `expert_ref`.
- If `version` is older than stored (semver), reject with `STALE_VERSION` unless `force=true`.
- Fingerprint MUST be L2-normalized within \(10^{-3}\) tolerance or server normalizes and warns.

### 3.2 `Heartbeat`

**Request:** `{ node_id, expert_refs[] | all, capacity_snapshot, load_snapshot, ts }`.  
**Response:** `{ ok, next_heartbeat_ms }`.

**Semantics:**

- Experts not heartbeated within `heartbeat_ttl` (default 15s) become **non-routable**.
- Capacity/load update PredCap in the learner/router.

### 3.3 `Deregister`

**Request:** `{ expert_refs[], reason }`.  
**Response:** `{ ok }`.

In-flight leases MAY continue until expiry; new leases MUST fail.

### 3.4 `DescribeExperts`

**Request:** `{ query: NNQuery | ExplicitRefs, filters, limit }`.

`NNQuery`: `{ fingerprint, k, host_dims, domain_tags[] }`.  
`ExplicitRefs`: `{ expert_refs[] }`.

**Response:** `{ experts: ExpertDescriptor[], routable_flags[] }`.

**Semantics:** Only routable, ACL-visible experts. Cosine NN over fingerprints.

---

## 4. Router service

### 4.1 `ProposeCombinations`

**Request:**

```text
{
  host_model_id,
  context_embedding,      # float[]
  layer_hints[],          # optional int layer ids
  budget,                 # Budget
  n,                      # max plans (default 8)
  include_local_only      # default true
}
```

**Response:**

```text
{
  plans: CombinationPlan[],   # length ≤ n, scored desc
  router_policy_version
}
```

**Semantics:**

- MUST include local-only if `include_local_only`.
- MUST NOT return incompatible refs without `adapter_id`.
- If `budget.require_leases`, Router SHOULD attempt `LeaseCapacity` and attach `lease_id`s to plan steps; on failure, omit that plan or strip the failing ref per soft/hard budget flags.
- Algorithm: [docs/learning.md](learning.md) §3.

---

## 5. Node data-plane service

### 5.1 `LeaseCapacity`

**Request:** `{ expert_ref, tokens_or_qps, ttl_ms, priority }`.  
**Response:** `{ lease_id, lease_deadline, granted_qps }`.

**Errors:** `CAPACITY_EXHAUSTED`, `ACL_DENIED`, `NOT_ROUTABLE`.

### 5.2 `ReleaseCapacity`

**Request:** `{ lease_id }`.  
**Response:** `{ ok }`. Idempotent.

### 5.3 `ForwardExpert`

**Request:**

```text
{
  request_id,
  expert_ref,
  lease_id?,              # required if node policy demands leases
  activation: ActivationBatch,
  adapter_id?,
  metadata: { host_model_id, host_layer_id, plan_id, token_count }
}
```

**Response:**

```text
{
  activation: ActivationBatch,  # output hidden (and optional aux)
  actual_latency_ms,
  expert_version
}
```

**Semantics:**

- Validate ACL, lease (if required), dtype/dims (apply adapter if named).
- At-most-once side effects w.r.t. capacity accounting keyed by `request_id` within TTL (default 60s).
- If `grad_required` and training profile disabled → `PROFILE_DISABLED`.
- MUST NOT return raw weights.

### 5.4 `ExportWeights` (optional profile `export`)

**Request:** `{ expert_ref, format }`.  
**Response:** weight blob or deny.  
**Default ACL:** deny all. Audit every grant.

---

## 6. Learner service

### 6.1 `ReportOutcome`

**Request:**

```text
{
  plan_id,
  host_model_id,
  reward,                 # U - λ costs as computed by host, or raw U + components
  utility,
  latency_ms,
  capacity_penalty,
  tokens,
  fallbacks: { layer_id, reason }[],
  partial,                # true if aborted
  context_embedding?,     # optional echo for offline training
  plan_snapshot?          # CombinationPlan if learner is stateless w.r.t. plan store
}
```

**Response:** `{ ok, learner_version }`.

**Semantics:** Hosts MUST report executed plans. Learners MAY drop malformed events with metrics.

---

## 7. Fallback and error contract

| Code | Host action |
|------|-------------|
| `DEADLINE_EXCEEDED` | Local fallback for layer |
| `CAPACITY_EXHAUSTED` | Try alternate plan or local |
| `ACL_DENIED` | Never retry same ref without auth change; local |
| `NOT_ROUTABLE` | Refresh Describe; local |
| `INCOMPATIBLE_DIMS` | Treat as router bug; local; report |
| `PROFILE_DISABLED` | Disable grad path; local |
| `UNAVAILABLE` (Router) | Local-only entire forward |

`budget.strict_local_fallback=true` ⇒ any remote failure forces full local-only continuation for remaining layers in that request.

---

## 8. Security profile

### 8.1 Transport

- mTLS required in production profile.
- Certificate SAN or SPIFFE ID → `principal_id`.

### 8.2 Authorization

ACL tuple: `(principal_id, expert_ref, action)` with `action ∈ {describe, forward, lease, export}`.

Registry filters `DescribeExperts` by `describe`. Nodes enforce `forward` / `lease` / `export`.

### 8.3 Data minimization

- Registry stores descriptors, not activations.
- Activation logs opt-in; retention capped.
- `privacy` profile: optional Gaussian noise on activations or truncated seq length (informative parameters left to deployer).

### 8.4 Integrity

- Descriptor `version` monotonic per `expert_ref`.
- Plans carry `ttl_ms`; hosts MUST NOT execute expired plans without re-propose.

---

## 9. Sequencing (happy path)

```text
Host                         Router              Registry           RemoteNode           Learner
  |--ProposeCombinations------>|                    |                    |                   |
  |                            |--DescribeExperts-->|                    |                   |
  |                            |<--descriptors------|                    |                   |
  |                            |--LeaseCapacity------------------------->|                   |
  |                            |<--lease_id------------------------------|                   |
  |<--CombinationPlan[]--------|                    |                    |                   |
  |--ForwardExpert------------------------------------------------------>|                   |
  |<--ActivationBatch----------------------------------------------------|                   |
  |--ReleaseCapacity---------------------------------------------------->|                   |
  |--ReportOutcome------------------------------------------------------------------------------->|
```

---

## 10. Versioning

- `cei_version` on every call.
- Minor 0.x additions MUST be backward compatible (new optional fields).
- Breaking changes bump minor in 0.x or major when ≥1.0.

Wire types: see [schemas/cei.proto](../schemas/cei.proto).
