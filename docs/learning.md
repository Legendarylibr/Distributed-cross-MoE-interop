# Combination Learning — Cross-Expert Interoperation

Normative algorithms for discovering optimal **layer × expert** combinations across a hierarchical MoE fleet. Parent: [SPEC.md](../SPEC.md).

---

## 1. Objective (recap)

\[
R(\pi,x) = U\bigl(y_\pi(x)\bigr)
- \lambda_{\mathrm{lat}}\,\mathrm{Lat}(\pi,x)
- \lambda_{\mathrm{cap}}\,\mathrm{Cap}(\pi,x)
- \lambda_{\mathrm{stick}}\,d(\pi,\pi_{\mathrm{prev}})
\]

Policy \(\pi\) maps context features \(\phi(x)\) to a `CombinationPlan` (or a distribution over plans).

**Context features \(\phi(x)\)** (recommended):

- Host `model_id`
- Sequence / prompt embedding (mean pool of early hidden states or a dedicated encoder)
- Domain soft tags (optional classifier)
- Current load vector from Registry (optional, for capacity-aware routing)
- Layer hint mask from the host

---

## 2. Two timescales

```
┌──────────────────────────────────────────────────────────┐
│ ONLINE (per token / sequence)                            │
│  local soft route → score C(x) → sample plan → forward   │
│  → measure Lat, Cap → emit ReportOutcome                 │
└────────────────────────────┬─────────────────────────────┘
                             │ batches of outcomes
                             ▼
┌──────────────────────────────────────────────────────────┐
│ OFFLINE / EPISODIC                                       │
│  update bandit / neural π ← rewards                      │
│  refresh fingerprint NN index stats                      │
│  optional distillation of winning remote bindings        │
└──────────────────────────────────────────────────────────┘
```

| Timescale | Cadence | What is updated |
|-----------|---------|-----------------|
| Online | Every request (or microbatch) | Instantiation of plan; exploration noise |
| Offline | Every \(B\) outcomes (default \(B=256\)) or wall-clock epoch | Value estimates / policy parameters |

---

## 3. Candidate generation (normative)

Default search depth \(m = 2\) layers; candidate list size \(n = 8\).

### 3.1 Operators

Given base local set \(S^{\mathrm{loc}}_\ell\) and a remote candidate expert \(e^\star\):

| Op | Effect |
|----|--------|
| **swap** | Replace lowest-weight local expert with \(e^\star\) |
| **insert** | Add \(e^\star\) if \(\|S\| < k_{\max}\); else swap |
| **drop** | Remove one remote binding (return toward local-only) |

At most \(m\) layers may differ from pure local in any single candidate.

### 3.2 Algorithm

```text
function ProposeCandidates(host_i, φ, layer_hints, budget, m=2, n=8):
  L ← SelectLayers(layer_hints, host_i)   # prefer mid-depth if empty
  base ← { ℓ → LocalTopK(host_i, ℓ) for ℓ in L }

  pool ← []
  for ℓ in L:
    q ← FingerprintQuery(Registry, desc_of(base[ℓ]), k_nn=32)
    q ← filter q by compat(., host layer dims) and ACL and budget
    pool.append((ℓ, q))

  C ← { LocalOnlyPlan(base) }             # always include baseline
  for subset S of L with |S| ≤ m:
    for each ℓ in S:
      for e in top_nn(pool[ℓ], k=4):
        for op in {swap, insert}:
          c ← ApplyOp(base, ℓ, e, op)
          if Feasible(c, budget): C.add(c)
    # drop neighbors from previously used remote plans (stickiness explore)
    C.add(DropRemoteVariants(base, prior_plans, S))

  scored ← []
  for c in C:
    û ← EstimateUtility(π_hat, φ, c)      # learner value model
    cost ← λ_lat * PredLat(c) + λ_cap * PredCap(c)
    scored.append((û - cost, c))

  return TopN(scored, n)                  # as CombinationPlan list
```

### 3.3 Sampling at serve time

Given scored plans \(\{(s_j, c_j)\}\):

- **ε-greedy:** with prob \(\varepsilon\) uniform among top-\(n\); else \(\arg\max s_j\). Default \(\varepsilon=0.05\).
- **Gumbel-top-1:** \(c^\star = \arg\max_j (s_j + \mathrm{Gumbel}(0,1)/\tau)\). Default \(\tau=1.0\).

Hosts MAY pin a plan for `ttl_ms` to reduce thrashing (stickiness).

---

## 4. Online path (host)

```text
function ForwardWithMarketplace(x, host_i, budget):
  φ ← EmbedContext(x)
  plans ← Router.ProposeCombinations(φ, layer_hints, budget)
  c ← Sample(plans)                       # ε-greedy or Gumbel
  LeaseAll(c)                             # LeaseCapacity per remote ref

  h ← Embed(x)
  for ℓ in 1..L_i:
    if ℓ in c.layers:
      try:
        y ← ExecuteStep(c, ℓ, h, budget)  # local + ForwardExpert
      catch Timeout | LeaseFail | RpcError:
        y ← LocalMoEStep(host_i, ℓ, h)    # mandatory fallback
        record_fallback(ℓ)
    else:
      y ← LocalMoEStep(host_i, ℓ, h)
    h ← ResidualAttnMix(h, y)             # host-defined transformer block
  ŷ ← Head(h)

  R ← TaskUtility(ŷ, x) - λ_lat*Lat - λ_cap*Cap
  Learner.ReportOutcome(c.plan_id, R, Lat, tokens, fallbacks)
  ReleaseLeases(c)
  return ŷ
```

Local soft routing inside `LocalMoEStep` remains differentiable for ordinary host training. Marketplace discrete choices are treated as **non-differentiable** environment actions unless the `training` profile enables straight-through / REINFORCE estimators (optional).

---

## 5. Offline / episodic learner

### 5.1 Contextual bandit (default, normative baseline)

Maintain for each arm \(a\) (discretized plan template: layer set + expert-ref multiset pattern):

- Context-aware score \(f_\theta(\phi, a)\) (linear or small MLP)
- Update on each batch with squared error or pairwise ranking loss toward observed \(R\)

```text
function UpdateLearner(batch):
  for (φ, a, R, Lat, Cap) in batch:
    pred ← f_θ(φ, a)
    loss += (pred - R)^2
    # optional: load-balance bonus toward underused experts
    loss += λ_bal * BalancePenalty(a, load_stats)
  θ ← θ - η ∇loss
  UpdateStickinessPrior(batch)
```

**Arm encoding:** hash `(sorted layer ids, sorted expert_refs, ops)` into a fixed codebook of size \(A\) (default \(65\,536\)) with collision-tolerant embeddings.

### 5.2 Neural policy (optional upgrade)

Policy \(\pi_\psi(a \mid \phi)\) trained with REINFORCE / PPO-lite on episodic reward \(R\), baseline \(b(\phi) = f_\theta(\phi)\). MUST still emit valid `CombinationPlan`s under `compat` and budget constraints (mask illegal arms).

### 5.3 Auxiliary signals

| Signal | Purpose |
|--------|---------|
| Load Gini / per-expert QPS | \(\lambda_{\mathrm{bal}}\) penalty |
| Fallback rate | Increase \(\lambda_{\mathrm{lat}}\) or shrink \(m\) |
| Stickiness \(d(\pi_t,\pi_{t-1})\) | Hysteresis |
| Distillation trigger | If remote binding wins ≥ \(T\) times (default 100) with margin, enqueue distill job |

---

## 6. Distillation (optional, non-normative for conformance)

When remote expert \(e^\star\) at host layer \(\ell\) wins repeatedly:

1. Allocate or select a local student expert \(E_{i,\ell,k_{\mathrm{new}}}\).
2. Minimize \(\|E_{i,\ell,k_{\mathrm{new}}}(h) - \tilde{E}_{e^\star}(h)\|^2\) on replayed activations (or online).
3. Promote student into local bank; deprecate marketplace dependency for that binding.

This reduces steady-state cross-node traffic while preserving quality.

---

## 7. EstimateUtility and cost models

Until enough data exists, cold-start:

\[
\hat{U}(\phi,c) = U_{\mathrm{local}} + \alpha \sum_{e \in c} \mathrm{cos}(f_e, \phi_{\mathrm{domain}})
\]

\[
\mathrm{PredLat}(c) = \sum_{e \in c_{\mathrm{remote}}} \bigl(p50_e + \widehat{\mathrm{RTT}}(\mathrm{node}_e)\bigr)
\]

\[
\mathrm{PredCap}(c) = \sum_{e} \max\bigl(0,\; \mathrm{load}_e / \mathrm{capacity}_e - \tau\bigr)
\]

Registry heartbeats supply \(p50\), load, and capacity.

---

## 8. Hyperparameter defaults

| Name | Default | Notes |
|------|---------|-------|
| \(m\) | 2 | Max layers with remote edits |
| \(n\) | 8 | Plans returned by router |
| \(k_{\mathrm{nn}}\) | 32 | Fingerprint NN pool |
| \(\varepsilon\) | 0.05 | Explore rate |
| \(\tau\) | 1.0 | Gumbel temperature |
| \(B\) | 256 | Learner batch |
| \(\lambda_{\mathrm{lat}}\) | task-normalized so median Lat ≈ 0.1·\|U\| | Calibrate per deployment |
| \(\lambda_{\mathrm{cap}}\) | same order as \(\lambda_{\mathrm{lat}}\) | |
| \(\lambda_{\mathrm{stick}}\) | 0.01 | |
| \(\lambda_{\mathrm{bal}}\) | 0.01 | |
| plan `ttl_ms` | 5000 | |

---

## 9. Correctness requirements

1. **Baseline inclusion:** Every `ProposeCombinations` response MUST include a local-only plan (or explicitly signal `local_only_equivalent=true` on the top plan).
2. **Feasibility:** No plan may include incompatible experts without an `adapter_id`.
3. **Budget:** Plans violating `max_remote_experts` or predicted latency hard caps MUST be excluded (not merely down-scored), unless `budget.allow_soft_latency` is set.
4. **Reporting:** Hosts MUST call `ReportOutcome` for executed plans, including fallbacks (`reward` still required; set `partial=true` if aborted).

---

## 10. Relation to prior art (informative)

- **Branch-Train-MiX:** offline merge of experts into one MoE — complementary; CEI keeps models separate and composes at inference/training via RPC.
- **ReXMoE / PathMoE:** cross-layer reuse inside one model — CEI extends the combination search across **model** boundaries via the marketplace.
