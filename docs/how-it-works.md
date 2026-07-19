# How CEI works — a plain-language breakdown

This page breaks the system down from first principles. No prior CEI knowledge
assumed; light MoE background helps. The normative wording lives in
[../SPEC.md](../SPEC.md) — this is the readable version.

---

## 1. Thirty-second version

A **Mixture-of-Experts (MoE)** model replaces the single feed-forward block in
each transformer layer with a *bank* of smaller feed-forward networks
("experts") plus a tiny router that picks the top-\(k\) experts per token. Only
the chosen experts run, so the model gets big capacity at small per-token cost.

CEI asks: if you operate a **fleet** of separate MoE models — one strong at
code, one at math, one general — why should each model be limited to its own
expert bank? CEI lets a model, at inference time, **borrow individual experts
from other models over the network**, but only when a learned policy predicts
the borrowed expert improves the answer by more than the network round-trip
costs.

Concretely: a code-specialized model answering a math-heavy prompt can swap
one or two of its mid-depth expert slots for experts served by the math
model's node, run the rest of the forward pass locally as usual, and report
back how well that went so the policy improves.

---

## 2. The cast of characters

| Role | One-line job | Reference code |
|------|--------------|----------------|
| **MoE Host** | Runs a model's forward pass; decides per request whether to borrow | `cei/host.py` |
| **Expert Node** | Serves a model's experts to others: leases capacity, runs `ForwardExpert` | `cei/node.py` |
| **Expert Registry** | Catalog of every published expert: who has what, is it healthy, what does it look like (fingerprint) | `cei/registry.py` |
| **Combination Router** | Given a request's context, proposes a shortlist of "plans" (which layers to edit, with whose experts) | `cei/router.py` |
| **Combination Learner** | Contextual bandit that learns which plans actually pay off, from reported outcomes | `cei/learner.py` |
| **Adapter Hub** | Stores small projection matrices that translate between models with mismatched hidden dimensions | `cei/adapters.py` |

A "node" in the reference deployment bundles a Host and an Expert Node in one
process: it both consumes remote experts and serves its own.

The **control plane** (Registry, Router, Learner, Adapter Hub) never sees user
data — only descriptors, scores, and rewards. The **data plane** is exactly
one RPC, `ForwardExpert`, and it is the only place activations (which can leak
user input) cross the network. That separation is a load-bearing security
property, not an implementation detail.

---

## 3. Life of a request

What happens when a prompt arrives at host `moe-code`, step by step.

### Step 0 — before any traffic: publish and heartbeat

Each node fingerprints its experts (an L2-normalized vector summarizing what
the expert "does") and registers an `ExpertDescriptor` per exportable expert
with the Registry: layer id, input/output dims, fingerprint, capacity,
version. Nodes then heartbeat every few seconds with load and capacity. An
expert that misses heartbeats for ~15 s becomes non-routable; an expert that
was never **promoted** (explicitly, or via `CEI_AUTO_PROMOTE`) is never
routable at all. Promotion is the "this expert passed sandbox eval" gate.

### Step 1 — embed the context

The host computes a small context embedding \(\phi(x)\) for the incoming
request (in the reference sim, a domain-flavored vector; in a real model, a
pooled early hidden state). This is the *only* request-derived data that goes
to the control plane, and it is deliberately lossy.

### Step 2 — ask the Router for plans

The host calls `ProposeCombinations(φ, budget)`. The Router:

1. Queries the Registry's nearest-neighbor index: "which routable experts have
   fingerprints closest to this context?" (cosine similarity, top-32 pool).
2. Filters by hard constraints — layer compatibility (`exact_layer` by
   default), dimension match or an available adapter, ACL visibility, and the
   caller's latency/expert budget.
3. Builds candidate **plans**. A plan is a *sparse edit* of the host's normal
   routing: "at layer 2, replace your weakest local expert with
   `moe-math/layer2/expert5` at weight 0.6". At most \(m=2\) layers may be
   edited per plan, and a pure local-only plan is always included as the
   baseline.
4. Scores every candidate with the Learner's cached policy (predicted utility
   minus predicted latency and capacity cost) and returns the top \(n=8\).

### Step 3 — pick one plan

The host samples from the shortlist — usually the top-scored plan, sometimes
an exploratory pick (ε-greedy, ε=0.05) so the learner keeps getting signal on
near-misses. Plans carry an issue timestamp and TTL; a stale plan is discarded
and the host falls back to local rather than execute outdated routing.

### Step 4 — lease before you borrow

For each remote expert in the chosen plan, the host calls `LeaseCapacity` on
the owning node: "reserve N tokens/QPS for the next T ms". The node checks its
ACL, checks headroom, and returns a `lease_id` **bound to this expert and this
caller's principal**. Leases are the marketplace's admission control: a noisy
neighbor can't starve a node it never got a lease from, and a stolen lease id
is useless to anyone but the grantee.

### Step 5 — the forward pass, with borrowed pieces

The host runs its transformer layers normally. At an edited layer it sends the
layer input activations via `ForwardExpert(lease_id, activation)` to the
remote node, which validates the lease (right expert, right principal, not
expired), validates the tensor (shape, dims, finite values), runs its expert
(applying an adapter first if dims differ), and returns the output
activations. The host mixes that output with its local experts per the plan's
weights and continues.

**If anything goes wrong — timeout, lease rejection, node crash — the host
substitutes its own local experts for that layer and keeps going.** Fallback
is mandatory and unconditional; the marketplace can only ever *add* capability,
never take availability away. Leases are released in a `finally` block, so
failures don't strand reserved capacity.

### Step 6 — measure and report

The host computes the realized reward: task utility minus weighted latency and
capacity penalties. It sends `ReportOutcome(plan_id, reward, latency, …)` to
the Learner, HMAC-signed and bound to the request id so a malicious node can't
forge or replay flattering rewards for its own experts.

### Step 7 — the loop closes

The Learner folds batches of outcomes into its per-arm value estimates (an
"arm" is a plan template: which layers, whose experts, what ops). Routers
periodically pull a versioned **policy snapshot**, so scoring on the hot path
is a cache lookup, not an RPC. Over time, plans that genuinely help get
proposed more; plans that looked good on fingerprint similarity but didn't pay
off get suppressed.

---

## 4. Why it's built this way

**Why not one giant sharded MoE?** Expert-parallel MoE assumes one model, one
training run, one operator. CEI's premise is *sovereign* models — separate
owners, release trains, and trust domains — that cooperate selectively. Each
model keeps working if the marketplace disappears.

**Why not merge the models offline (BTX-style)?** Merging produces a single
checkpoint: you lose independent releases, and you pay the merge cost again
every time any constituent improves. CEI composes at inference time, so each
model evolves on its own schedule.

**Why leases instead of best-effort calls?** Without admission control, a
popular expert becomes a DoS target and its owner has no lever. Leases make
capacity explicit, bounded, and attributable to a principal.

**Why a bandit rather than supervised routing?** Nobody has labels for "which
cross-model expert combination is best for this prompt". The reward signal
only exists after you try a plan, which is exactly the contextual-bandit
setting. Fingerprint similarity provides the cold-start prior; observed
rewards take over from there.

**Why fingerprints if they're not trusted?** Fingerprints are a *retrieval*
mechanism (find plausibly-relevant experts fast), not a security or quality
guarantee. Anyone can publish a flattering fingerprint — that's why routing
eligibility is gated by ACLs and promotion, and why the learner scores plans
by *observed* reward.

---

## 5. Where trust changes hands

Every arrow that crosses a process boundary is a place someone can lie. The
v0.2 hardening assigns a control to each:

| Someone could… | …and is stopped by |
|----------------|--------------------|
| Impersonate another principal | mTLS peer identity, or HMAC request tokens (`CEI_AUTH_SECRET`) over `principal\|request_id\|timestamp` with a freshness window |
| Register experts for a model they don't own | Publisher allowlist + ownership: only the registering principal may re-register or deregister |
| Poison another node's health metrics | Heartbeats only refresh experts owned by the reporting node |
| Use someone else's lease | Leases bound to `(expert_ref, principal)`; mismatches fail `LEASE_MISMATCH` |
| Inflate rewards for their own experts | Outcome attestation: HMAC over canonical outcome fields, bound to the request id |
| Replay a legitimate signed outcome | Learner replay cache rejects duplicate request ids |
| Send malformed tensors to crash a node | Wire validation: shape/size/rank/finite checks before any compute |
| Execute a stale routing decision | Plan TTL: hosts refuse expired plans and fall back to local |

Full operator checklist: [security.md](security.md). Adversarial analysis:
[security-redteam.md](security-redteam.md).

---

## 6. Seeing it run

The fastest way to make this concrete is the in-process simulator:

```bash
cei-simulate --mode ablate --steps 400
```

That runs the same traffic through four fleets — local-only, random remote
swaps, fixed fingerprint heuristic, and the full learner — and prints the
comparison. In this synthetic setting the learner should dominate on utility
at comparable latency; the gap over `local` is the value the marketplace
added, and the gap over `heuristic` is the value *learning* added on top of
retrieval. Keep §7 in mind when reading the numbers: the utility function is
built to reward domain-matched swaps, so this validates the loop, not the
research hypothesis.

Then the real multi-process version, with gRPC, HMAC auth, and attestation:

```bash
docker compose up --build -d registry learner adapter-hub router \
  node-code node-math node-general
docker compose run --rm driver
```

Guides: [getting-started.md](getting-started.md) ·
[deploy-compose.md](deploy-compose.md).

---

## 7. What this does and doesn't prove

Being honest about the gap between the machinery and the claim:

- **Proven by this repo:** the *protocol* works end-to-end — registration,
  fingerprint retrieval, plan proposal, leasing, cross-node forwarding with
  fallback, reward reporting, bandit improvement, and every security control
  in §5 — across real processes, real gRPC, and hostile-input tests.
- **Assumed, not proven:** that a borrowed expert actually helps. The
  simulator's utility function rewards domain-matched swaps *by construction*,
  so the learner beating local-only here shows the learning loop functions,
  not that real models benefit. Whether independently trained residual streams
  are compatible enough for borrowing (even through adapters) is the open
  research question this spec exists to make testable.
- **Simplified on purpose:** experts are random matrices, latency is sampled
  from a distribution, the learner is a linear bandit, state is in-memory
  (a restart loses the catalog, the policy, and the replay cache), and the
  Registry/Learner are single processes. The full list with consequences is
  in [../README.md § Honest limitations](../README.md#honest-limitations).

If you take one thing away: **the numbers the simulator prints measure the
plumbing, not the science.** Integrating real checkpoints and re-running the
A0–A3 ablations ([evaluation.md](evaluation.md)) is the experiment that
matters.

---

## 8. Glossary

| Term | Meaning |
|------|---------|
| **Expert** | One feed-forward sub-network inside an MoE layer |
| **Expert bank** | All experts a model owns at one layer |
| **Descriptor** | Registry record for one expert: ref, dims, fingerprint, capacity, version |
| **Fingerprint** | L2-normalized vector summarizing an expert's specialty; used for NN retrieval only |
| **Plan (`CombinationPlan`)** | A sparse edit of the host's layer×expert binding, with weights and ops |
| **Arm** | A plan template as seen by the bandit learner (layer set + expert refs + ops) |
| **Lease** | Time-boxed, principal-bound capacity reservation on a remote expert |
| **Promotion** | Operator gate making a registered expert routable |
| **Attestation** | HMAC over canonical outcome fields binding a reward report to one request |
| **Principal** | Authenticated caller identity (mTLS SAN, HMAC token, or dev-only asserted) |
| **Adapter** | Small `(W_in, W_out)` projection pair translating between mismatched hidden dims |
| **Fallback** | Mandatory local substitution when any remote step fails |
