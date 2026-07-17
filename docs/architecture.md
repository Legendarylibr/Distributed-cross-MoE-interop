# Architecture — Hierarchical Cross-Expert Interoperation

Rationale and topology for CEI. Parent: [SPEC.md](../SPEC.md).

---

## 1. Why hierarchical (not single sharded MoE)

| Approach | Strength | Weakness for multi-model fleets |
|----------|----------|----------------------------------|
| Single MoE + expert parallelism | Mature all-to-all EP stacks | Forces one training/serving identity; hard to isolate domains and ownership |
| Offline merge (BTX / MoErging) | Strong single checkpoint | No live borrowing; re-merge cost; loses independent release cycles |
| **Hierarchical CEI (this spec)** | Independent models + shared marketplace | Extra latency; needs compat/adapters and admission control |

CEI keeps **local sovereignty** (each model owns its bank and release train) while allowing **selective composition** when the combination learner finds positive utility after cost.

---

## 2. Topology

```
                    ┌─────────────────────┐
                    │ Combination Learner │
                    │ + policy snapshot   │
                    └─────────▲───────────┘
                              │ ReportOutcome / GetPolicySnapshot
                    ┌─────────┴───────────┐
   ProposeComb.     │ Combination Router  │     (scores from cached policy)
┌──────────────────►│  (policy cache)     │
│                   └─────────▲───────────┘
│                             │ DescribeExperts
│                   ┌─────────┴───────────┐
│                   │  Expert Registry    │
│                   │  + fingerprint NN   │
│                   │  + per-model quotas │
│                   └─────────▲───────────┘
│                             │ Register / Heartbeat
│         ┌───────────────────┼───────────────────┐
│         │                   │                   │
│  ┌──────┴──────┐     ┌──────┴──────┐     ┌──────┴──────┐
│  │ MoE Host A  │     │ MoE Host B  │     │ MoE Host C  │
│  │ local bank  │◄───►│ local bank  │◄───►│ local bank  │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
│         │ ForwardExpert (data plane)            │
└─────────┴───────────────────────────────────────┘
                    ▲
                    │ GetAdapter
              ┌─────┴──────┐
              │ Adapter Hub│
              └────────────┘
```

### Trust boundaries

1. **Intra-host:** Local experts and local router — highest trust, no RPC.
2. **Marketplace control plane:** Registry, Router, Learner, Adapter Hub — authenticated operators; **no raw user tensors** on Registry / Learner / Adapter Hub paths.
3. **Data plane:** `ForwardExpert` carries activations — treat as **sensitive**; mTLS + ACL required in production; optional privacy (DP / truncation) profile.

### Normative vs reference

| Concern | Normative (architecture) | Reference impl (`cei/`) |
|---------|--------------------------|-------------------------|
| Lease ownership | **Host-owned** (see §3.6) | Host leases at execute time |
| Policy scoring | Router uses **cached policy snapshot** | `GetPolicySnapshot` + TTL cache; RPC fallback for cold arms |
| Layer compatibility | Configurable mode (default `exact_layer`) | Enforced in combination search |
| Adapter Hub | First-class control-plane role | gRPC `AdapterHub` (+ in-process hub on nodes) |
| Registry HA | Replicated / sharded (see §9) | Single process (dev) |
| Fingerprints | Probe-batch or weight projection | Weight/specialty projection |

---

## 3. Role details

### 3.1 Local MoE node (Host + ExpertNode)

- Runs standard MoE layers for its host model.
- Publishes one `ExpertDescriptor` per exportable expert.
- Serves `ForwardExpert` under lease and ACL.
- **Owns leases** for remotes it will call (see §3.6).
- MAY refuse `grad_required=true` unless `training` profile is enabled.
- Applies adapters locally (cached from Adapter Hub) when `adapter_id` is set.

### 3.2 Expert Registry

- Source of truth for descriptors, versions, health, load/capacity.
- Approximate NN over fingerprints.
- Marks experts **non-routable** if heartbeat older than `heartbeat_ttl` (default 15s), or if not **promoted** (`promote` on register / `CEI_AUTO_PROMOTE`).
- Enforces **per-`model_id` expert quotas** (admission control on `RegisterExpert`).
- Fingerprints are **not** a security boundary; ACLs are (deny-by-default unless `allow_all` / lab profile).
- Publisher allowlist (`CEI_REGISTRY_PUBLISHERS`) gates `RegisterExpert`; consumers (`CEI_REGISTRY_CONSUMERS`) receive describe ACL grants.

### 3.3 Combination Router

- Builds candidate sets per [docs/learning.md](learning.md).
- Scores candidates from a **policy cache** refreshed from Learner (`GetPolicySnapshot`); does **not** require per-candidate RPCs on the hot path.
- Applies **layer compatibility** (§5) and budget filters.
- Does **not** take capacity leases (host-owned leasing).

### 3.4 Combination Learner

- Consumes `ReportOutcome` streams; updates contextual bandit / policy.
- When `CEI_REQUIRE_OUTCOME_ATTESTATION=1`, rejects unauthenticated rewards (HMAC over canonical outcome fields).
- Exposes `GetPolicySnapshot` (arm parameters + version) for Router caches.
- Owns stickiness and load-balance auxiliaries in the value model.

### 3.5 Adapter Hub

- First-class control-plane service: `adapter_id → (W_in, W_out, dims, dtype, content_digest)`.
- Writer ACL (`CEI_ADAPTER_WRITERS`); nodes verify digest when provided.
- Descriptors reference `adapter_id` when dims/domains differ.
- Nodes **fetch and cache** adapters; Hub never sees activations.

### 3.6 Lease ownership (locked)

**Host-owned leases.** The executing host:

1. Receives a `CombinationPlan` (no leases attached, or advisory only).
2. Calls `LeaseCapacity` on each remote expert it will invoke.
3. Calls `ForwardExpert` with `lease_id`.
4. Releases leases when the step completes (or on failure).

Rationale: only the host knows actual token counts and timeouts; router pre-lease would hold capacity for plans that are never sampled.

---

## 4. Composition unit

A combination is a **sparse edit** of the host’s layer×expert binding:

- Default: edit at most \(m=2\) layers.
- Each edited layer lists expert refs + weights + `op` (`replace` or `augment`) + optional `adapter_ids`.
- Unedited layers: pure local top-\(k\).

Search is capped (`max_candidates`, default 64) after NN top-4 and budget filters.

---

## 5. Layer compatibility (locked default)

Remote experts are filtered by `CEI_LAYER_COMPAT` / router `layer_compat`:

| Mode | Rule |
|------|------|
| **`exact_layer`** (default) | Remote `expert_ref.layer_id` must equal the host layer being edited |
| **`any_with_adapter`** | Any layer if dims match **or** an `adapter_id` is advertised |
| **`free`** | Any routable compatible-dim (or adapted) expert — research only |

**Representation risk:** Even with matching dims/layer, residual streams may not align across independently trained models. Fingerprints are a retrieval prior, not a proof of semantic compatibility. Prefer `exact_layer` + same-family models, or require adapters trained for the host↔remote pair.

---

## 6. Fingerprints

L2-normalized vectors (default dim 64). Construction (informative):

1. **Preferred:** fixed probe batch through the expert; pool activations.
2. **Reference:** random projection / slice of weights + specialty bias.

---

## 7. Failure and degradation

| Failure | Host behavior |
|---------|----------------|
| Router unavailable | Local-only |
| Registry stale / empty NN | Local-only |
| Policy cache stale | Refresh snapshot; if Learner down, score with fingerprint prior only |
| Lease denied | Drop that remote ref or whole plan per budget |
| Forward timeout | Local fallback for that layer; report `fallback=true` |
| Adapter missing | Treat as incompatible; exclude from candidates / fallback |

---

## 8. Privacy (data plane)

- Activations on `ForwardExpert` may leak inputs — **sensitive by default**.
- Production: mTLS, minimal retention, no tensor logs.
- Optional `privacy` profile: truncation and/or DP noise (deployer parameters).
- Control plane MUST NOT receive activation tensors.

---

## 9. Scaling and HA (informative → target)

**Control plane** scales independently of the activation data plane.

| Component | Scaling | HA target |
|-----------|---------|-----------|
| Registry | Shard by `model_id` or fingerprint IVF | ≥2 replicas behind consistent discovery; heartbeat ownership per shard; clients tolerate stale NN briefly |
| Router | Stateless + policy cache | Horizontal replicas; cache TTL / version fencing |
| Learner | Single-writer policy version | Primary + warm standby; snapshot export for routers |
| Adapter Hub | Content-addressed by `adapter_id` | Replicated blob/kv; nodes local-cache |
| Hosts / ExpertNodes | Scale with fleet | Independent; affinity_tags for placement |

Until HA is deployed, treat Registry and Learner as **single points of failure** and keep local-only fallback mandatory.

---

## 10. Comparison diagram

```
Single EP MoE:     [Host] ==all2all== [Expert shards of SAME model]

BTX merge:         [Expert LLMs] --offline mix--> [ONE MoE checkpoint]

CEI hierarchical:  [MoE A] --RPC Forward--> [Experts of B/C]
                   [MoE A] --Propose------> [Router + cached policy]
                   [MoE A] --leases-------> [Remote nodes]
                   local banks remain first-class
```
