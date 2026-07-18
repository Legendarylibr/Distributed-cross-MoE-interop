"""Build a single-domain MoE host+node (for distributed node containers)."""

from __future__ import annotations

import json
import os

import numpy as np

from cei.adapters import AdapterHub
from cei.host import MoEHost
from cei.node import ExpertNode, fingerprint_from_weights, make_expert_module
from cei.simulate import DOMAINS
from cei.types import DType, ExpertDescriptor, ExpertRef


def orthogonal_domain_vecs(dim: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    domain_vecs: dict[str, np.ndarray] = {}
    for d in DOMAINS:
        v = rng.normal(size=(dim,))
        for prev in domain_vecs.values():
            v = v - np.dot(v, prev) * prev
        domain_vecs[d] = v / (np.linalg.norm(v) + 1e-12)
    return domain_vecs


def build_domain_host(
    domain: str,
    dim: int = 32,
    num_layers: int = 4,
    experts_per_layer: int = 4,
    seed: int = 0,
) -> tuple[MoEHost, ExpertNode, dict[str, np.ndarray]]:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain}")
    domain_vecs = orthogonal_domain_vecs(dim, seed=seed)
    # Stable per-domain offset: hash() is salted per process (PYTHONHASHSEED)
    rng = np.random.default_rng(seed + 1000 * (DOMAINS.index(domain) + 1))
    model_id = f"moe-{domain}"
    hub = AdapterHub()
    identity = AdapterHub.identity(f"identity-{dim}", dim, rng)
    hub.register(identity)
    node = ExpertNode(
        node_id=f"node-{domain}",
        model_id=model_id,
        base_latency_ms=1.0,
        remote_extra_latency_ms=4.0,
        adapter_hub=hub,
        acl_open=False,
        acl_allow=set(),
        priority_admins=set(),
    )
    from cei.security import get_config

    cfg = get_config()
    if cfg.node_acl_open:
        node.acl_open = True
    else:
        node.acl_allow = set(cfg.node_acl_allow)
    node.priority_admins = set(cfg.priority_admins)

    layer_expert_ids: dict[int, list[int]] = {}
    gate_weights: dict[int, np.ndarray] = {}

    for ell in range(num_layers):
        ids = list(range(experts_per_layer))
        layer_expert_ids[ell] = ids
        W = rng.normal(scale=0.1, size=(experts_per_layer, dim))
        for k in ids:
            ref = ExpertRef(model_id, ell, k)
            if k == 0:
                spec = domain_vecs[domain].copy()
            elif k == 1:
                other = DOMAINS[(DOMAINS.index(domain) + 1) % len(DOMAINS)]
                spec = 0.6 * domain_vecs[domain] + 0.4 * domain_vecs[other]
                spec = spec / (np.linalg.norm(spec) + 1e-12)
            else:
                spec = rng.normal(size=(dim,))
                spec = spec / (np.linalg.norm(spec) + 1e-12)
            module = make_expert_module(ref, dim, domain, rng, specialty=spec)
            fp = fingerprint_from_weights(module)
            fp = 0.5 * fp + 0.5 * np.pad(spec, (0, max(0, len(fp) - len(spec))))[: len(fp)]
            fp = fp / (np.linalg.norm(fp) + 1e-12)
            desc = ExpertDescriptor(
                expert_ref=ref,
                version="1.0.0",
                dim_in=dim,
                dim_out=dim,
                dtype=DType.F32,
                fingerprint=fp,
                cost_flops=dim * dim,
                p50_latency_ms=3.0,
                capacity_qps=1000.0,
                domain_tags=[domain],
                node_id=node.node_id,
            )
            node.add_expert(module, desc)
            W[k] = spec + 0.05 * rng.normal(size=(dim,))
        gate_weights[ell] = W

    host = MoEHost(
        model_id=model_id,
        node=node,
        router=None,
        learner=None,
        num_layers=num_layers,
        dim=dim,
        top_k=2,
        domain=domain,
        gate_weights=gate_weights,
        layer_expert_ids=layer_expert_ids,
        rng=rng,
    )
    return host, node, domain_vecs


def peer_addrs_from_env() -> dict[str, str]:
    raw = os.environ.get("CEI_PEER_ADDRS", "{}")
    return json.loads(raw)
