"""Adapter Hub gRPC servicer."""

from __future__ import annotations

import numpy as np

from cei.adapters import Adapter, AdapterHub
from cei.pb import cei_internal_pb2, cei_internal_pb2_grpc
from cei.security import adapter_digest, audit, can_write_adapter, resolve_principal


class AdapterHubServicer(cei_internal_pb2_grpc.AdapterHubServicer):
    def __init__(self, hub: AdapterHub | None = None) -> None:
        self.hub = hub or AdapterHub()
        self._digests: dict[str, str] = {}

    def UpsertAdapter(self, request, context):
        meta_p = request.meta.principal_id if request.meta else None
        principal = resolve_principal(context, meta_p)
        if not can_write_adapter(principal):
            audit("adapter_upsert_deny", principal=principal, reason="WRITER_ACL")
            return cei_internal_pb2.UpsertAdapterResponse(ok=False, error_code="ACL_DENIED")
        try:
            blob = request.adapter
            digest = adapter_digest(bytes(blob.w_in), bytes(blob.w_out))
            if blob.content_digest and blob.content_digest != digest:
                audit(
                    "adapter_upsert_deny",
                    principal=principal,
                    adapter_id=blob.adapter_id,
                    reason="DIGEST_MISMATCH",
                )
                return cei_internal_pb2.UpsertAdapterResponse(
                    ok=False, error_code="DIGEST_MISMATCH"
                )
            w_in = np.frombuffer(blob.w_in, dtype=np.float64).reshape(
                tuple(int(x) for x in blob.w_in_shape)
            )
            w_out = np.frombuffer(blob.w_out, dtype=np.float64).reshape(
                tuple(int(x) for x in blob.w_out_shape)
            )
            self.hub.register(
                Adapter(
                    adapter_id=blob.adapter_id,
                    dim_in_host=blob.dim_in_host,
                    dim_in_remote=blob.dim_in_remote,
                    dim_out_remote=blob.dim_out_remote,
                    dim_out_host=blob.dim_out_host,
                    w_in=w_in.copy(),
                    w_out=w_out.copy(),
                )
            )
            self._digests[blob.adapter_id] = digest
            audit(
                "adapter_upsert_ok",
                principal=principal,
                adapter_id=blob.adapter_id,
                digest=digest,
            )
            return cei_internal_pb2.UpsertAdapterResponse(ok=True, content_digest=digest)
        except Exception as exc:  # noqa: BLE001
            return cei_internal_pb2.UpsertAdapterResponse(ok=False, error_code=str(exc))

    def GetAdapter(self, request, context):
        adapter = self.hub.get(request.adapter_id)
        if adapter is None:
            return cei_internal_pb2.GetAdapterResponse(error_code="NOT_FOUND")
        w_in = np.asarray(adapter.w_in, dtype=np.float64).tobytes()
        w_out = np.asarray(adapter.w_out, dtype=np.float64).tobytes()
        digest = self._digests.get(adapter.adapter_id) or adapter_digest(w_in, w_out)
        return cei_internal_pb2.GetAdapterResponse(
            adapter=cei_internal_pb2.AdapterBlob(
                adapter_id=adapter.adapter_id,
                dim_in_host=adapter.dim_in_host,
                dim_in_remote=adapter.dim_in_remote,
                dim_out_remote=adapter.dim_out_remote,
                dim_out_host=adapter.dim_out_host,
                w_in=w_in,
                w_out=w_out,
                w_in_shape=list(adapter.w_in.shape),
                w_out_shape=list(adapter.w_out.shape),
                content_digest=digest,
            )
        )
