# Cross-Expert Interoperation Specification (Hierarchical MoE)

**Status:** Normative  
**Version:** 0.2.0  
**Scope:** Learning algorithm + distributed interoperation protocol for a fleet of Mixture-of-Experts (MoE) models with local expert banks and a cross-model expert marketplace.

Companion documents:

- [docs/README.md](docs/README.md) — documentation index
- [docs/getting-started.md](docs/getting-started.md) — install and first runs
- [docs/architecture.md](docs/architecture.md) — topology, roles, rationale
- [docs/learning.md](docs/learning.md) — combination learner algorithms and pseudocode
- [docs/protocol.md](docs/protocol.md) — RPC semantics and wire sketches
- [docs/evaluation.md](docs/evaluation.md) — metrics, ablations, threats, conformance tests
- [docs/deploy-compose.md](docs/deploy-compose.md) — Docker Compose multi-node deploy
- [docs/security.md](docs/security.md) — operator security profiles and controls
- [docs/security-redteam.md](docs/security-redteam.md) — adversarial risk assessment (AI-REDTEAM-ULTRA)
- [schemas/](schemas/) — Protobuf and JSON Schema contracts
- [cei/](cei/) — reference Python simulator (Registry, Router, Learner, hosts)
- [docker-compose.yml](docker-compose.yml) — registry / router / learner / node-* / driver

---

## 1. Problem statement

### 1.1 Motivation

Organizations increasingly operate a **fleet** of specialized MoE models (code, math, multimodal, domain verticals). Each model maintains its own expert bank and local top-\(k\) router. Specialist capacity is trapped inside model boundaries: a code MoE cannot cheaply borrow a strong math expert at a mid-depth layer, and operators cannot systematically discover which **layer placements** and **expert identities** (local or remote) maximize quality under latency and capacity budgets.

### 1.2 Goals

1. Expose experts across models through a **hierarchical marketplace** without collapsing the fleet into a single sharded MoE.
2. **Learn** policies over layer×expert combinations that maximize task utility subject to cost.
3. Define a **normative interoperation protocol** so any conforming node can register, propose combinations, forward activations, and report outcomes.

### 1.3 Non-goals

- Replacing expert parallelism (EP) inside a single MoE.
- Full training-framework integration (DeepSpeed, Megatron, etc.).
- Production orchestration (Kubernetes operators).
- Shipping concrete model weights or benchmark datasets (named only in evaluation).

### 1.4 Design defaults (locked)

| Decision | Choice |
|----------|--------|
| Topology | Hierarchical: local banks + cross-model marketplace |
| Composition unit | Combination path over layers with local/remote expert refs |
| Compatibility | Descriptor match or advertised adapter |
| Learning | Two-timescale: online soft/local + offline contextual bandit / neural policy |
| Transport | gRPC-style RPC; optional RDMA for activation tensors; timeout → local fallback |

---

## 2. Formal model

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| \(M_i\), \(i \in \{1,\ldots,N\}\) | MoE model in the fleet |
| \(L_{i,\ell}\), \(\ell \in \{1,\ldots,L_i\}\) | Layer \(\ell\) of model \(i\) |
| \(E_{i,\ell,k}\), \(k \in \{1,\ldots,K_{i,\ell}\}\) | Expert \(k\) at layer \(\ell\) of model \(i\) |
| \(\mathrm{ref}(E) = (i,\ell,k)\) | Globally unique expert reference |
| \(h_\ell \in \mathbb{R}^{d}\) | Hidden state entering layer \(\ell\) (host model) |
| \(g_{i,\ell}\) | Local router at \(L_{i,\ell}\) |
| \(r\) | Cross-model combination router (marketplace) |
| \(\pi\) | Combination policy (maps context → combination) |
| \(\mathcal{D}\) | Task / data distribution |

An **expert reference** is the triple \((model\_id, layer\_id, expert\_id)\). Remote use may additionally name an `adapter_id`.

### 2.2 Local MoE step

For host model \(M_i\) at layer \(\ell\), the local router produces scores and a top-\(k\) set \(S^{\mathrm{loc}}_{i,\ell}(h)\):

\[
z = g_{i,\ell}(h),\qquad
S^{\mathrm{loc}}_{i,\ell}(h) = \mathrm{TopK}(z, k),\qquad
y^{\mathrm{loc}} = \sum_{e \in S^{\mathrm{loc}}} w_e \, E_e(h)
\]

with normalized weights \(w_e\) from the gated scores (softmax over selected experts, or sigmoid-style as configured by the host).

### 2.3 Combinations

A **combination** for a forward on host \(M_i\) is a partial function over layers that may replace or augment the local expert set:

\[
c = \bigl\{(\ell, S_\ell) : \ell \in \mathcal{L}_c\bigr\}
\]

where \(S_\ell\) is a multiset of expert refs (local or remote), \(|S_\ell| \le k_{\max}\), and \(\mathcal{L}_c \subseteq \{1,\ldots,L_i\}\) is the set of layers participating in cross composition (often small; default search depth \(m=2\)).

Layers not in \(\mathcal{L}_c\) use pure local routing.

### 2.4 Compatibility

Let \(\mathrm{desc}(E)\) be the published **ExpertDescriptor** (Section 6). Experts \(E_a\) (host-needed) and \(E_b\) (candidate) are compatible if:

\[
\mathrm{compat}(E_a, E_b) \iff
\begin{cases}
\mathrm{dim\_in}(E_b)=\mathrm{dim\_in}(E_a) \land \mathrm{dim\_out}(E_b)=\mathrm{dim\_out}(E_a) \\
\quad\text{or an adapter }A\text{ is advertised with matching dims,} \\
\text{and domain / dtype constraints in the request budget are satisfied.}
\end{cases}
\]

Adapters \(A: \mathbb{R}^{d_{\mathrm{host}}} \to \mathbb{R}^{d_{\mathrm{remote}}}\) (and inverse on outputs) live in the **Adapter Hub**.

### 2.5 Cross-augmented layer

Given plan \(c\) and hidden \(h\) at host layer \(\ell\):

\[
S_\ell =
\begin{cases}
S^{\mathrm{loc}}_{i,\ell}(h) & \ell \notin \mathcal{L}_c \\
\mathrm{Resolve}(c,\ell) & \ell \in \mathcal{L}_c
\end{cases}
\]

\(\mathrm{Resolve}\) may **swap**, **insert**, or **drop** relative to local top-\(k\) as specified by the combination plan. Remote experts are invoked via `ForwardExpert` (Section 5); local experts run in-process.

\[
y_\ell = \sum_{e \in S_\ell} w_e \, \tilde{E}_e(h),\qquad
\tilde{E}_e =
\begin{cases}
E_e & \text{local} \\
A^{\mathrm{out}} \circ E_e \circ A^{\mathrm{in}} & \text{remote with adapter} \\
E_e & \text{remote, dim-matched}
\end{cases}
\]

### 2.6 Objective

Let \(y_\pi(x)\) be the model output under policy \(\pi\), \(U\) a task utility (negative loss, reward, or eval metric), \(\mathrm{Lat}\) measured end-to-end latency (or p95 estimate), and \(\mathrm{Cap}\) a capacity / congestion penalty (queueing, lease denial, load imbalance):

\[
\max_\pi \;
\mathbb{E}_{x \sim \mathcal{D}}
\Bigl[
U\bigl(y_\pi(x)\bigr)
- \lambda_{\mathrm{lat}}\,\mathrm{Lat}(\pi,x)
- \lambda_{\mathrm{cap}}\,\mathrm{Cap}(\pi,x)
\Bigr]
\]

Optional stickiness regularizer (discourage thrashing across episodes):

\[
- \lambda_{\mathrm{stick}} \, \mathbb{E}\bigl[d(\pi_t, \pi_{t-1})\bigr]
\]

where \(d\) is a distance on combination space (e.g., Hamming over layer bindings).

---

## 3. Hierarchical roles

| Role | Responsibility |
|------|----------------|
| **Local MoE node** | Own experts; run local top-\(k\); publish descriptors; serve remote `ForwardExpert`; enforce ACLs and capacity leases |
| **Expert Registry** | Catalog, versioning, health, capacity quotas, fingerprint index for nearest-neighbor retrieval |
| **Combination Router** | Build candidate set \(C(x)\); emit ranked `CombinationPlan`; coordinate leases |
| **Combination Learner** | Update \(\pi\) from `ReportOutcome`; maintain value estimates / policy params; publish stickiness priors |
| **Adapter Hub** | Store and serve adapters; advertise adapter↔expert bindings in descriptors |
| **Host client** | Owning model process that requests plans, executes forwards, reports outcomes |

A single physical process may co-locate several roles (e.g., router + learner). Normative behavior is defined per role, not per process.

```
┌─────────────────────────────────────────────────────────────┐
│                     Model Fleet (hosts)                      │
│   MoE A ──local──► Experts A.ℓ.k                             │
│   MoE B ──local──► Experts B.ℓ.k                             │
│   MoE C ──local──► Experts C.ℓ.k                             │
└────────────┬──────────────────────────┬──────────────────────┘
             │ publish / ForwardExpert  │ Propose / Report
             ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Cross-Model Marketplace                         │
│  Registry ◄──► Combination Router ◄──► Combination Learner   │
│       │              │                                       │
│       └──── Descriptor / Activation Cache ────┘              │
│                    Adapter Hub                               │
└─────────────────────────────────────────────────────────────┘
```

Detailed rationale: [docs/architecture.md](docs/architecture.md).

---

## 4. Learning algorithm (summary)

Normative algorithms and pseudocode: [docs/learning.md](docs/learning.md).

### 4.1 Two timescales

1. **Online (token / sequence):** Host runs local soft routing. For layers in the active plan, apply marketplace bindings from a small candidate set \(C(x)\) chosen by the Combination Router. Prefer Gumbel-top-\(k\) or \(\varepsilon\)-greedy over scored candidates under the latency budget.
2. **Offline / episodic:** After batches of `ReportOutcome`, the Combination Learner updates a **contextual bandit** (or neural policy) with reward \(U - \lambda_{\mathrm{lat}}\mathrm{Lat} - \lambda_{\mathrm{cap}}\mathrm{Cap}\). Episodic search explores swap/insert/drop at up to \(m\) layers (default \(m=2\)).

### 4.2 Candidate generation (normative outline)

1. Compute local top-\(k\) refs at candidate layers (or use layer hints from the host).
2. Query Registry for nearest fingerprints (cosine) among `compat` experts, filtered by budget and ACLs.
3. Generate neighbors via **swap / insert / drop** at ≤ \(m\) layers.
4. Score \(\hat{U} - \lambda\) costs; return top-\(n\) `CombinationPlan`s (default \(n=8\)).

### 4.3 What is learned

The system jointly learns:

- **Which layers** may import remote experts (\(\mathcal{L}_c\)).
- **Which expert identities** bind at those layers.
- Not merely routing weights inside a fixed expert graph.

Optional: when a remote combination wins repeatedly, **distill** into a local expert or adapter (non-normative optimization; see learning doc).

---

## 5. Distributed interoperation protocol (summary)

Full message semantics: [docs/protocol.md](docs/protocol.md). Schemas: [schemas/cei.proto](schemas/cei.proto), [schemas/](schemas/).

### 5.1 Message types (normative)

| Message | Direction | Purpose |
|---------|-----------|---------|
| `RegisterExpert` | Node → Registry | Publish / update descriptor |
| `Heartbeat` | Node → Registry | Liveness + capacity snapshot |
| `Deregister` | Node → Registry | Remove expert(s) |
| `DescribeExperts` | Router → Registry | Batch fingerprints, costs, capacity |
| `ProposeCombinations` | Host → Router | Context embedding, layer hints, budget → ranked plans |
| `LeaseCapacity` / `ReleaseCapacity` | Router/Host → Node | Burst / admission control |
| `ForwardExpert` | Host → Node | Hidden states in → activations out |
| `ReportOutcome` | Host → Learner | Reward, latency, tokens, plan id |
| `ExportWeights` | Host → Node | **Optional, deny-by-default** weight export |

### 5.2 Semantics

- **Lease ownership (locked):** **Host-owned.** The host that executes a plan calls `LeaseCapacity` / `ReleaseCapacity` on remotes. The Router does not pre-lease (avoids holding capacity for unsampled plans). See [docs/architecture.md](docs/architecture.md) §3.6.
- **Idempotency:** Every mutating or forward request carries `request_id` (UUID). Servers treat duplicates as at-most-once for side effects; `ForwardExpert` may be exactly-once w.r.t. billing/capacity if the node tracks `request_id` within a TTL window.
- **Leases:** Capacity leases expire at `lease_deadline`. Stale leases MUST NOT be used for new forwards. Leases are bound to the `(expert_ref, principal)` they were granted for; forwards or releases with a mismatched expert or principal MUST be rejected (`LEASE_MISMATCH`).
- **Plan TTL:** Plans carry `issued_unix_ms` + `ttl_ms`. Hosts MUST NOT execute remote steps of an expired plan; degrade to local (`PLAN_EXPIRED`).
- **Fallback:** If remote RTT exceeds `budget.max_remote_latency_ms`, or lease/forward fails, the host MUST degrade to **local-only** combination for that layer (or the whole plan if `budget.strict_local_fallback` is set).
- **Security:** mTLS between all roles in production; scoped expert ACLs (`principal → expert_ref → {forward, describe, export}`); activation-only RPC by default. Activations are sensitive (privacy profile optional). When mTLS is unavailable, `RequestMeta.auth_token` (HMAC-SHA256 over `principal|request_id|ts` with a fleet secret) proves principal identity; outcome attestations are bound to `request_id` and servers MUST reject replays. Servers MUST validate untrusted wire inputs (tensor shapes/sizes, descriptor fields, adapter matrices) before use.
- **Layer compatibility:** Default `exact_layer` — remote expert layer id must match the host layer being edited (`CEI_LAYER_COMPAT`).
- **Policy cache:** Router scores from Learner `GetPolicySnapshot` cache; not per-candidate RPCs on the hot path.

### 5.3 Wire sketch

```text
Host  --ProposeCombinations-->  Router
Router --DescribeExperts------>  Registry
Router --LeaseCapacity-------->  Remote nodes
Host  --ForwardExpert*-------->  Local / remote expert nodes
Host  --ReportOutcome--------->  Learner
```

---

## 6. Data contracts

Canonical types are defined in [schemas/cei.proto](schemas/cei.proto) and mirrored in JSON Schema under [schemas/](schemas/). Normative fields:

### 6.1 `ExpertDescriptor`

| Field | Type | Notes |
|-------|------|-------|
| `expert_ref` | `(model_id, layer_id, expert_id)` | Globally unique |
| `version` | string / semver | Registry versioning |
| `dim_in`, `dim_out` | int | Hidden dims |
| `dtype` | enum | e.g. `F16`, `BF16`, `F32` |
| `domain_tags` | string[] | e.g. `code`, `math` |
| `fingerprint` | float[d] | Default \(d=64\); L2-normalized for cosine NN |
| `cost_flops` | int64 | Nominal FLOPs per token |
| `p50_latency_ms` | float | Self-reported or measured |
| `capacity_qps` | float | Soft capacity |
| `adapter_id` | optional string | Required if dims differ from typical host |
| `acl_policy_id` | optional string | Registry-side policy handle |

### 6.2 `CombinationPlan`

| Field | Type | Notes |
|-------|------|-------|
| `plan_id` | string | UUID |
| `host_model_id` | string | |
| `steps` | list of `{layer_id, expert_refs[], weights[], op}` | `op ∈ {replace, augment}` |
| `budget` | `Budget` | Latency, FLOPs, max remote hops |
| `ttl_ms` | int | Plan validity |
| `score` | float | Router estimate \(\hat{U}-\lambda\mathrm{cost}\) |

### 6.3 `ActivationBatch`

| Field | Type | Notes |
|-------|------|-------|
| `tensor` | bytes or remote buffer ref | Row-major hidden states |
| `shape` | `[batch, seq, dim]` or packed MoE layout | |
| `dtype` | enum | |
| `seq_layout` | metadata | padding mask / token→expert map |
| `grad_required` | bool | If true, node MAY refuse unless training profile enabled |

### 6.4 `Budget`

| Field | Type | Notes |
|-------|------|-------|
| `max_remote_latency_ms` | float | Per-forward or plan-level |
| `max_remote_experts` | int | Cap on remote refs in a plan |
| `max_flops` | int64 | Optional |
| `strict_local_fallback` | bool | Whole-plan vs per-layer fallback |

---

## 7. Evaluation protocol

### 7.1 Metrics

| Metric | Definition |
|--------|------------|
| Task quality | Domain metrics (accuracy, pass@k, perplexity, reward) |
| Latency | Cross-node and e2e p50 / p99 |
| Utilization | Expert load; Gini coefficient across marketplace experts |
| Fallback rate | Fraction of plans/layers that degraded to local-only |
| Combination regret | \(U(\pi^\star) - U(\pi)\) vs oracle local-only and vs expensive full search |
| Lease denial rate | Failed `LeaseCapacity` / attempted |

### 7.2 Ablations (required for claims of learning benefit)

1. **Local-only** — no marketplace.
2. **Random remote** — random compatible swaps at \(m\) layers.
3. **Fixed heuristic** — nearest fingerprint, no learner update.
4. **Learned combinations** — full two-timescale system.

**Controlled comparison (normative):** Ablation modes MUST be compared as paired runs — identical seeds, fleet configuration, and task stream — so that the mode is the only variable that changes between rungs. Results SHOULD be reported as mean and variance over ≥3 seeds. Comparisons that vary anything besides the ablation mode (seed, workload, budgets, fleet topology) do not support claims of learning benefit.

### 7.3 Workloads

Multi-domain mixture where each fleet model is strong on a subset (e.g., code / math / general). Report quality–latency Pareto curves under fixed capacity quotas.

---

## 8. Threats and operational constraints

| Threat | Mitigation |
|--------|------------|
| Poisoned marketplace experts | Reputation scores; sandbox eval before promotion; ACL deny lists |
| Capacity collapse | Admission control, fair-share leases, shedding low-priority plans |
| Routing thrashing | Stickiness prior \(\lambda_{\mathrm{stick}}\); plan TTL hysteresis |
| Activation privacy leakage | Optional DP noise / truncation profile; minimize logging of tensors |
| Weight exfiltration | `ExportWeights` deny-by-default; audit |
| Stale descriptors | Heartbeat expiry; Registry MUST mark unhealthy experts non-routable |

---

## 9. Conformance

An implementation is **CEI-conformant** if it:

1. Implements Local MoE node + at least one of Registry, Router, Learner with the message types in Section 5.
2. Honors `compat`, budgets, lease expiry, and local fallback.
3. Publishes descriptors with required fields in Section 6.1.
4. Uses `ReportOutcome` fields sufficient for the learner objective in Section 2.6.

Optional profiles: `training` (grad-enabled forwards), `rdma` (tensor transport), `export` (weight export), `privacy` (DP activations).

---

## 10. Document history

| Version | Date | Notes |
|---------|------|-------|
| 0.1.0 | 2026-07-17 | Initial hierarchical MoE cross-expert interoperation spec |
| 0.2.0 | 2026-07-18 | Production hardening: HMAC request auth (`RequestMeta.auth_token`), replay-protected outcome attestation, lease binding to `(expert, principal)`, plan TTL (`issued_unix_ms`), mandatory wire-input validation |
