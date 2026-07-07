"""Tests for worker_service.validate_environment (P1-153).

The API and worker run as separate Railway services with independent env
vars — MAPBOX_TOKEN being set on the API alone silently broke geocoding for
non-preapproved locations. This isn't fatal (REDIS_URL still is), but the
worker should at least warn about it at startup instead of failing silently.
"""

import logging

import worker_service


def test_missing_mapbox_token_logs_warning(monkeypatch, caplog):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    with caplog.at_level(logging.WARNING, logger="worker_service"):
        worker_service.validate_environment()
    assert any("MAPBOX_TOKEN" in r.message for r in caplog.records)


def test_mapbox_token_set_logs_no_warning(monkeypatch, caplog):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("MAPBOX_TOKEN", "fake-token")
    with caplog.at_level(logging.WARNING, logger="worker_service"):
        worker_service.validate_environment()
    assert not any("MAPBOX_TOKEN" in r.message for r in caplog.records)


def test_missing_redis_url_still_exits(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("MAPBOX_TOKEN", "fake-token")
    try:
        worker_service.validate_environment()
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert exc.code == 1
