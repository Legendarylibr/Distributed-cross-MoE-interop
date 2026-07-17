"""Adapter hub: dim projections when cross-model experts differ."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Adapter:
    adapter_id: str
    dim_in_host: int
    dim_in_remote: int
    dim_out_remote: int
    dim_out_host: int
    w_in: np.ndarray
    w_out: np.ndarray

    def forward_in(self, h: np.ndarray) -> np.ndarray:
        # h: (..., dim_in_host) -> (..., dim_in_remote)
        return h @ self.w_in

    def forward_out(self, y: np.ndarray) -> np.ndarray:
        return y @ self.w_out


@dataclass
class AdapterHub:
    _adapters: dict[str, Adapter] = field(default_factory=dict)

    def register(self, adapter: Adapter) -> None:
        self._adapters[adapter.adapter_id] = adapter

    def get(self, adapter_id: str) -> Adapter | None:
        return self._adapters.get(adapter_id)

    @staticmethod
    def identity(adapter_id: str, dim: int, rng: np.random.Generator) -> Adapter:
        # Near-identity with tiny noise for sim stability
        w = np.eye(dim, dtype=np.float64) + 0.01 * rng.normal(size=(dim, dim))
        return Adapter(
            adapter_id=adapter_id,
            dim_in_host=dim,
            dim_in_remote=dim,
            dim_out_remote=dim,
            dim_out_host=dim,
            w_in=w,
            w_out=np.linalg.inv(w) if abs(np.linalg.det(w)) > 1e-6 else np.eye(dim),
        )

    @staticmethod
    def random_proj(
        adapter_id: str,
        dim_host: int,
        dim_remote: int,
        rng: np.random.Generator,
    ) -> Adapter:
        w_in = rng.normal(scale=1.0 / np.sqrt(dim_host), size=(dim_host, dim_remote))
        w_out = rng.normal(scale=1.0 / np.sqrt(dim_remote), size=(dim_remote, dim_host))
        return Adapter(
            adapter_id=adapter_id,
            dim_in_host=dim_host,
            dim_in_remote=dim_remote,
            dim_out_remote=dim_remote,
            dim_out_host=dim_host,
            w_in=w_in,
            w_out=w_out,
        )
