"""Request-auth tokens (HMAC), principal resolution, and replay protection."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cei import security, wire
from cei.learner import ContextualBanditLearner
from cei.pb import cei_pb2
from cei.server.learner_servicer import LearnerServicer
from cei.types import Outcome


@pytest.fixture
def auth_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CEI_SECURITY_PROFILE", "secure")
    monkeypatch.setenv("CEI_AUTH_SECRET", "fleet-secret")
    monkeypatch.setenv("CEI_REQUIRE_AUTH_TOKEN", "1")
    monkeypatch.setenv("CEI_OUTCOME_HMAC_SECRET", "outcome-secret")
    monkeypatch.setenv("CEI_REQUIRE_OUTCOME_ATTESTATION", "1")
    security.reset_config_cache()
    yield
    security.reset_config_cache()


def test_sign_and_verify_meta_roundtrip(auth_env):
    now = int(time.time() * 1000)
    token = security.sign_meta("host-code", "req-1", now)
    meta = cei_pb2.RequestMeta(
        request_id="req-1", principal_id="host-code", ts_unix_ms=now, auth_token=token
    )
    assert security.verify_meta(meta) is True


def test_verify_meta_rejects_tampered_principal(auth_env):
    now = int(time.time() * 1000)
    token = security.sign_meta("host-code", "req-1", now)
    meta = cei_pb2.RequestMeta(
        request_id="req-1", principal_id="attacker", ts_unix_ms=now, auth_token=token
    )
    assert security.verify_meta(meta) is False


def test_verify_meta_rejects_stale_timestamp(auth_env):
    stale = int(time.time() * 1000) - 10 * 60 * 1000  # 10 minutes old
    token = security.sign_meta("host-code", "req-1", stale)
    meta = cei_pb2.RequestMeta(
        request_id="req-1", principal_id="host-code", ts_unix_ms=stale, auth_token=token
    )
    assert security.verify_meta(meta) is False


def test_verify_meta_rejects_missing_token(auth_env):
    meta = cei_pb2.RequestMeta(
        request_id="req-1", principal_id="host-code", ts_unix_ms=int(time.time() * 1000)
    )
    assert security.verify_meta(meta) is False


def test_resolve_principal_requires_valid_token(auth_env):
    now = int(time.time() * 1000)
    good = cei_pb2.RequestMeta(
        request_id="r",
        principal_id="host-code",
        ts_unix_ms=now,
        auth_token=security.sign_meta("host-code", "r", now),
    )
    assert security.resolve_principal(None, good) == "host-code"
    # Unsigned meta is rejected outright when tokens are required.
    bad = cei_pb2.RequestMeta(request_id="r2", principal_id="host-code", ts_unix_ms=now)
    assert security.resolve_principal(None, bad) is None


def test_resolve_principal_lab_profile_trusts_meta(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CEI_SECURITY_PROFILE", "lab")
    monkeypatch.delenv("CEI_AUTH_SECRET", raising=False)
    monkeypatch.delenv("CEI_REQUIRE_AUTH_TOKEN", raising=False)
    security.reset_config_cache()
    meta = cei_pb2.RequestMeta(request_id="r", principal_id="dev")
    assert security.resolve_principal(None, meta) == "dev"
    security.reset_config_cache()


def test_new_meta_signs_automatically(auth_env):
    meta = wire.new_meta("host-code")
    assert meta.auth_token
    assert security.verify_meta(meta) is True


def test_replay_cache_blocks_duplicates():
    cache = security.ReplayCache(ttl_ms=60_000)
    assert cache.check_and_add("id-1") is True
    assert cache.check_and_add("id-1") is False
    assert cache.check_and_add("") is False  # empty ids are never fresh


def test_replay_cache_ttl_eviction():
    cache = security.ReplayCache(ttl_ms=10)
    assert cache.check_and_add("id-1", now_ms=0.0) is True
    # After TTL, the same id is fresh again.
    assert cache.check_and_add("id-1", now_ms=100.0) is True


def test_replay_cache_bounds_memory():
    cache = security.ReplayCache(ttl_ms=1e12, max_entries=10)
    for i in range(25):
        assert cache.check_and_add(f"id-{i}", now_ms=float(i)) is True
    assert len(cache._seen) <= 10


def _outcome(reward: float = 1.0) -> Outcome:
    return Outcome(
        plan_id="p1",
        host_model_id="moe-code",
        reward=reward,
        utility=reward,
        latency_ms=1.0,
        capacity_penalty=0.0,
        tokens=1,
        context_embedding=np.zeros(4),
    )


def test_outcome_attestation_bound_to_request_id(auth_env):
    learner = ContextualBanditLearner(ctx_dim=4, batch_size=8)
    servicer = LearnerServicer(learner)
    req = wire.outcome_to_report_pb(_outcome())
    assert servicer.ReportOutcome(req, context=None).ok is True
    # Same signed request replayed → rejected.
    assert servicer.ReportOutcome(req, context=None).ok is False


def test_outcome_attestation_not_transferable_between_requests(auth_env):
    learner = ContextualBanditLearner(ctx_dim=4, batch_size=8)
    servicer = LearnerServicer(learner)
    original = wire.outcome_to_report_pb(_outcome())
    forged = cei_pb2.ReportOutcomeRequest()
    forged.CopyFrom(original)
    # New request_id with the old attestation must fail verification.
    forged.meta.request_id = "different-request"
    assert servicer.ReportOutcome(forged, context=None).ok is False


def test_outcome_rejects_nonfinite_reward(auth_env):
    learner = ContextualBanditLearner(ctx_dim=4, batch_size=8)
    servicer = LearnerServicer(learner)
    req = wire.outcome_to_report_pb(_outcome(reward=float("nan")))
    assert servicer.ReportOutcome(req, context=None).ok is False


def test_outcome_attestation_survives_wire_roundtrip(auth_env):
    """Proto floats are 32-bit; signature must verify after serialization."""
    learner = ContextualBanditLearner(ctx_dim=4, batch_size=8)
    servicer = LearnerServicer(learner)
    # Values chosen to lose precision under float32 truncation.
    outcome = _outcome()
    outcome.reward = 0.6893124712345678
    outcome.utility = 0.7041234987654321
    outcome.latency_ms = 9.123456789
    req = wire.outcome_to_report_pb(outcome)
    roundtripped = cei_pb2.ReportOutcomeRequest.FromString(req.SerializeToString())
    assert servicer.ReportOutcome(roundtripped, context=None).ok is True


def test_outcome_reward_tamper_detected(auth_env):
    learner = ContextualBanditLearner(ctx_dim=4, batch_size=8)
    servicer = LearnerServicer(learner)
    req = wire.outcome_to_report_pb(_outcome(reward=0.5))
    req.reward = 100.0  # tamper after signing
    assert servicer.ReportOutcome(req, context=None).ok is False
