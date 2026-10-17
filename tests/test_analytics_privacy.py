"""Client IP addresses are neither stored with analytics records, served by the
analytics endpoints, nor written to the logs (P1-184).

The IP is still used transiently for rate limiting; only the stored record, the
public read endpoints and the log lines are covered here.
"""

import json
import logging
import re
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from analytics_storage import AnalyticsStorage
from models import AnalyticsData

TEST_IP = "203.0.113.77"  # TEST-NET-3, never a real client

VALID_PAYLOAD = {
    "session_duration": 60,
    "api_calls": 1,
    "api_failure_rate": "0%",
    "retry_attempts": 0,
    "location_failures": 0,
    "error_count": 0,
    "recent_errors": [],
    "session_id": "privacy-test-session",
}


# ── Storage ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    return MagicMock()


@pytest.fixture
def storage(mock_redis):
    return AnalyticsStorage(mock_redis)


def _stored_record(mock_redis) -> dict:
    # setex(key, ttl, json_string)
    return json.loads(mock_redis.setex.call_args_list[0].args[2])


def test_store_analytics_does_not_save_an_ip_address(storage, mock_redis):
    analytics_id = storage.store_analytics(AnalyticsData(**VALID_PAYLOAD))

    record = _stored_record(mock_redis)
    assert "client_ip" not in record
    assert TEST_IP not in json.dumps(record)
    assert record["id"] == analytics_id
    assert record["session_id"] == "privacy-test-session"


def test_analytics_id_has_the_same_shape_without_an_ip_derived_suffix(storage):
    analytics_id = storage.store_analytics(AnalyticsData(**VALID_PAYLOAD))
    assert re.fullmatch(r"analytics_\d+_\d{1,4}", analytics_id)


def test_get_recent_analytics_strips_ip_from_records_stored_before_the_change(storage, mock_redis):
    mock_redis.lrange.return_value = [b"a1", b"a2"]
    mock_redis.mget.return_value = [
        json.dumps({"id": "a1", "client_ip": TEST_IP, "session_id": "s1"}),
        json.dumps({"id": "a2", "session_id": "s2"}),
    ]

    result = storage.get_recent_analytics(limit=2)

    assert [r["id"] for r in result] == ["a1", "a2"]
    assert all("client_ip" not in r for r in result)
    assert result[0]["session_id"] == "s1"


def test_get_analytics_by_session_strips_ip(storage, mock_redis):
    mock_redis.lrange.return_value = [b"a1"]
    mock_redis.mget.return_value = [json.dumps({"id": "a1", "client_ip": TEST_IP, "session_id": "wanted"})]

    result = storage.get_analytics_by_session("wanted")

    assert len(result) == 1
    assert "client_ip" not in result[0]


# ── Endpoints and logs ───────────────────────────────────────────────────────


@pytest.fixture
def client():
    from main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clear_test_ip_rate_limit():
    """The test IP is not one the shared analytics fixtures clear."""

    def _clear():
        try:
            from main import redis_client

            if redis_client:
                redis_client.delete(f"analytics_limit:{TEST_IP}")
        except Exception:  # noqa: S110
            pass

    _clear()
    yield
    _clear()


def test_submitted_ip_is_not_served_by_the_analytics_endpoints(client):
    headers = {"X-Forwarded-For": TEST_IP}

    submit = client.post("/analytics", json=VALID_PAYLOAD, headers=headers)
    assert submit.status_code == 200

    for path in ("/analytics/recent?limit=50", "/analytics/session/privacy-test-session", "/analytics/summary"):
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert TEST_IP not in response.text, path
        assert "client_ip" not in response.text, path


def test_ip_is_not_logged_by_the_analytics_endpoint(client, caplog):
    headers = {"X-Forwarded-For": TEST_IP}
    with caplog.at_level(logging.DEBUG):
        client.post("/analytics", json=VALID_PAYLOAD, headers=headers)  # stored
        client.post("/analytics", content=b"", headers={**headers, "Content-Type": "application/json"})  # empty body
        client.post("/analytics", content=b"{}", headers={**headers, "Content-Type": "text/plain"})  # wrong type
        client.post("/analytics", content=b"{not json", headers={**headers, "Content-Type": "application/json"})

    assert caplog.records, "expected the analytics handler to log something"
    assert TEST_IP not in caplog.text


def test_ip_is_not_logged_when_the_analytics_rate_limit_is_hit(client, caplog):
    headers = {"X-Forwarded-For": TEST_IP}
    with (
        patch("routers.analytics.RATE_LIMIT_ENABLED", True),
        patch("routers.analytics.ANALYTICS_RATE_LIMIT", 1),
        caplog.at_level(logging.DEBUG),
    ):
        first = client.post("/analytics", json=VALID_PAYLOAD, headers=headers)
        second = client.post("/analytics", json=VALID_PAYLOAD, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429
    assert "RATE LIMIT EXCEEDED" in caplog.text
    assert TEST_IP not in caplog.text


def test_ip_is_not_logged_by_the_request_logging_middleware(client, caplog):
    with patch("middleware.logging.DEBUG", True), caplog.at_level(logging.DEBUG):
        client.get("/analytics/recent?limit=1", headers={"X-Forwarded-For": TEST_IP})

    assert TEST_IP not in caplog.text
