# Evaluation & Threats (CEI)

Expanded checklist aligned with [SPEC.md](../SPEC.md) §§7–8. Use this when running experiments or reviewing deployments.

---

## 1. Evaluation protocol

### 1.1 Required metrics

| ID | Metric | How to compute |
|----|--------|----------------|
| Q1 | Task quality | Domain metric on held-out mix (accuracy, pass@k, PPL, reward) |
| L1 | E2E latency p50 / p99 | Host wall time per request |
| L2 | Cross-node latency p50 / p99 | Sum of `ForwardExpert.actual_latency_ms` + RTT |
| U1 | Expert utilization Gini | Across marketplace experts over window \(W\) |
| F1 | Fallback rate | Layers or plans with `fallback=true` / total |
| R1 | Regret vs local-only | \(U(\pi) - U(\pi_{\mathrm{local}})\) (negative = worse) |
| R2 | Regret vs full search | vs expensive oracle (\(m\) larger or exhaustive NN) |
| C1 | Lease denial rate | `CAPACITY_EXHAUSTED` / lease attempts |

### 1.2 Required ablations

| ID | System | Expectation |
|----|--------|-------------|
| A0 | Local-only | Baseline quality & latency |
| A1 | Random remote compatible swaps | Stress-test marketplace without learning |
| A2 | Fixed heuristic (nearest fingerprint) | No `ReportOutcome` learner updates |
| A3 | Full CEI learner | Should Pareto-dominate A0–A2 on quality–latency under quota |

**Controls (normative, SPEC §7.2):** A0–A3 MUST run as paired comparisons — same seed, same fleet, same task stream — so the mode is the only changed variable. Report mean ± variance over ≥3 seeds. The reference harness (`cei.simulate.run_ablations`) implements the paired design; sweep the `seed` argument for multi-seed reporting.

### 1.3 Workload recipe (informative)

1. Fleet of ≥3 MoE hosts, each strong on a disjoint domain slice.
2. Eval mixture: 1/3 in-domain per model + 1/3 cross-domain queries that benefit from remote experts.
3. Sweep `max_remote_latency_ms` and `max_remote_experts`; plot Pareto curves of Q1 vs L1.

### 1.4 Reporting

Publish: hyperparams from [docs/learning.md](learning.md) §8, `cei_version`, policy/registry versions, and ablation table A0–A3.

---

## 2. Threats & operational controls

| ID | Threat | Control (MUST / SHOULD) |
|----|--------|-------------------------|
| T1 | Poisoned expert | SHOULD sandbox-eval before promotion; MUST support ACL deny lists |
| T2 | Capacity collapse | MUST lease + admission control; SHOULD fair-share by principal |
| T3 | Routing thrashing | SHOULD stickiness prior + plan TTL hysteresis |
| T4 | Activation leakage | SHOULD disable tensor logs by default; MAY enable `privacy` profile |
| T5 | Weight exfiltration | MUST deny `ExportWeights` by default; MUST audit grants |
| T6 | Stale / zombie experts | MUST expire heartbeats → non-routable |
| T7 | Router spoofing | MUST mTLS; MUST bind plans to `principal_id` ACLs on forward |
| T8 | Replay / double-bill | MUST idempotent `request_id` windows on capacity accounting |

### 2.1 Incident playbooks (informative)

- **Poisoning suspected:** pin Registry to last-known-good versions; force local-only via feature flag; revoke `forward` ACL for suspect `model_id`.
- **Latency SLO breach:** lower \(m\) and \(n\); raise \(\lambda_{\mathrm{lat}}\); shed low-priority principals.
- **Learner divergence:** freeze policy version; roll back `router_policy_version`; serve A2 heuristic until recalibrated.

---

## 3. Conformance test sketch

1. Register two nodes with compatible fingerprints; `DescribeExperts` NN returns remote.
2. `ProposeCombinations` includes local-only and ≥1 remote plan under loose budget.
3. Kill remote mid-flight → host completes with fallback and `ReportOutcome.partial` or fallbacks set.
4. Expired lease → `ForwardExpert` fails; host does not hang past deadline.
5. Principal without ACL → `ACL_DENIED`; no activation returned.
