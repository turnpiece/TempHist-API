"""Every log line that used to carry the client IP is exercised here and checked
to be free of it (P1-184).

The analytics handler and the middleware are called directly. Going through the
HTTP stack would not reach several of these lines: the request-size middleware
rejects bad content types and sizes before the route runs.
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import Response

import main
from middleware.request_size import request_size_middleware
from routers.analytics import submit_analytics

TEST_IP = "203.0.113.77"  # TEST-NET-3, never a real client

PAYLOAD = json.dumps(
    {
        "session_duration": 60,
        "api_calls": 1,
        "api_failure_rate": "0%",
        "retry_attempts": 0,
        "location_failures": 0,
        "error_count": 0,
        "recent_errors": [],
    }
).encode()


def make_request(path="/analytics", method="POST", headers=None, body=b"", receive_error=None) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    raw_headers.append((b"x-forwarded-for", TEST_IP.encode()))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": raw_headers,
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    sent = False

    async def receive():
        nonlocal sent
        if receive_error:
            raise receive_error
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    return Request(scope, receive)


JSON_HEADERS = {"content-type": "application/json"}


@pytest.fixture(autouse=True)
def capture(caplog):
    caplog.set_level(logging.DEBUG)


def assert_logged_without_ip(caplog, marker: str) -> None:
    """The line was logged, and nothing logged contains the client IP.

    Checked inside each test: caplog.text only holds the current phase's records,
    so a check in fixture teardown would see an empty log and pass vacuously.
    """
    assert marker in caplog.text, f"expected a log line containing {marker!r}"
    assert TEST_IP not in caplog.text, "client IP leaked into the logs"


# ── POST /analytics handler ──────────────────────────────────────────────────


@pytest.fixture
def storage():
    s = MagicMock()
    s.store_analytics.return_value = "analytics_1_1"
    return s


async def submit(request, storage):
    with patch("routers.analytics.RATE_LIMIT_ENABLED", False):
        return await submit_analytics(request, redis_client=MagicMock(), analytics_storage=storage)


async def test_handler_rejects_wrong_content_type(caplog, storage):
    with pytest.raises(HTTPException) as exc:
        await submit(make_request(headers={"content-type": "text/plain"}, body=b"x"), storage)
    assert exc.value.status_code == 415
    assert_logged_without_ip(caplog, "INVALID CONTENT-TYPE")


async def test_handler_rejects_declared_oversize_body(caplog, storage):
    headers = {**JSON_HEADERS, "content-length": str(2 * 1024 * 1024)}
    with pytest.raises(HTTPException) as exc:
        await submit(make_request(headers=headers, body=PAYLOAD), storage)
    assert exc.value.status_code == 413
    assert_logged_without_ip(caplog, "REQUEST TOO LARGE")


async def test_handler_tolerates_unparseable_content_length(caplog, storage):
    headers = {**JSON_HEADERS, "content-length": "not-a-number"}
    response = await submit(make_request(headers=headers, body=PAYLOAD), storage)
    assert response.status == "success"
    assert_logged_without_ip(caplog, "INVALID CONTENT-LENGTH")


async def test_handler_rejects_empty_body(caplog, storage):
    with pytest.raises(HTTPException) as exc:
        await submit(make_request(headers=JSON_HEADERS, body=b""), storage)
    assert exc.value.status_code == 400
    assert_logged_without_ip(caplog, "EMPTY BODY")


async def test_handler_rejects_oversize_actual_body(caplog, storage):
    with pytest.raises(HTTPException) as exc:
        await submit(make_request(headers=JSON_HEADERS, body=b"x" * (1024 * 1024 + 1)), storage)
    assert exc.value.status_code == 413
    assert_logged_without_ip(caplog, "BODY TOO LARGE")


async def test_handler_rejects_invalid_json(caplog, storage):
    with pytest.raises(HTTPException) as exc:
        await submit(make_request(headers=JSON_HEADERS, body=b"{not json"), storage)
    assert exc.value.status_code == 400
    assert_logged_without_ip(caplog, "JSON PARSE ERROR")


async def test_handler_reports_a_body_read_failure(caplog, storage):
    request = make_request(headers=JSON_HEADERS, receive_error=RuntimeError("connection reset"))
    with pytest.raises(HTTPException) as exc:
        await submit(request, storage)
    assert exc.value.status_code == 400
    assert_logged_without_ip(caplog, "BODY READ ERROR")


async def test_handler_reports_a_validation_failure(caplog, storage):
    body = json.dumps({"session_duration": 60, "api_calls": 1, "error_count": "not a number"}).encode()
    with patch("routers.analytics.AnalyticsData", side_effect=ValueError("bad")):
        with pytest.raises(HTTPException) as exc:
            await submit(make_request(headers=JSON_HEADERS, body=body), storage)
    assert exc.value.status_code == 422
    assert_logged_without_ip(caplog, "VALIDATION ERROR")


async def test_handler_stores_and_logs_success(caplog, storage):
    response = await submit(make_request(headers=JSON_HEADERS, body=PAYLOAD), storage)
    assert response.status == "success"
    # The store is called with the data only: the IP is not passed on
    (args, kwargs) = storage.store_analytics.call_args
    assert len(args) == 1 and not kwargs
    assert_logged_without_ip(caplog, "ANALYTICS STORED")


async def test_handler_reports_a_storage_failure(caplog, storage):
    storage.store_analytics.side_effect = RuntimeError("redis down")
    with pytest.raises(HTTPException) as exc:
        await submit(make_request(headers=JSON_HEADERS, body=PAYLOAD), storage)
    assert exc.value.status_code == 500
    assert_logged_without_ip(caplog, "STORAGE ERROR")


async def test_handler_logs_sanitization_warnings(caplog, storage):
    with patch("routers.analytics._sanitize_analytics_payload", return_value=(json.loads(PAYLOAD), ["a warning"])):
        await submit(make_request(headers=JSON_HEADERS, body=PAYLOAD), storage)
    assert_logged_without_ip(caplog, "SANITIZATION WARNINGS")


async def test_handler_reports_an_unexpected_failure(caplog, storage):
    with patch("routers.analytics._sanitize_analytics_payload", side_effect=RuntimeError("boom")):
        with pytest.raises(HTTPException) as exc:
            await submit(make_request(headers=JSON_HEADERS, body=PAYLOAD), storage)
    assert exc.value.status_code == 500
    assert_logged_without_ip(caplog, "UNEXPECTED ERROR")


# ── request-size middleware ──────────────────────────────────────────────────


async def call_next(_request):
    return Response("ok")


async def test_request_size_middleware_logs_bad_content_type_without_ip(caplog):
    response = await request_size_middleware(make_request(headers={"content-type": "text/plain"}), call_next)
    assert response.status_code == 415
    assert_logged_without_ip(caplog, "INVALID CONTENT-TYPE")


async def test_request_size_middleware_logs_oversize_without_ip(caplog):
    headers = {**JSON_HEADERS, "content-length": str(2 * 1024 * 1024)}
    response = await request_size_middleware(make_request(headers=headers), call_next)
    assert response.status_code == 413
    assert_logged_without_ip(caplog, "REQUEST TOO LARGE")


async def test_request_size_middleware_logs_unparseable_length_without_ip(caplog):
    headers = {**JSON_HEADERS, "content-length": "abc"}
    response = await request_size_middleware(make_request(headers=headers), call_next)
    assert response.status_code == 200
    assert_logged_without_ip(caplog, "INVALID CONTENT-LENGTH")


# ── main.py middleware helpers ───────────────────────────────────────────────


def test_blacklisted_ip_block_is_logged_without_the_ip(caplog):
    request = make_request(path="/v1/records/daily/london/01-01", method="GET")
    with patch.object(main, "is_ip_blacklisted", return_value=True), patch.object(main, "DEBUG", True):
        response = main._check_ip_blacklist(request, TEST_IP)
    assert response.status_code == 403
    assert_logged_without_ip(caplog, "BLACKLISTED IP BLOCKED")


def _limits_patches(**overrides):
    defaults = {
        "RATE_LIMIT_ENABLED": True,
        "DEBUG": True,
        "location_monitor": MagicMock(),
        "request_monitor": MagicMock(),
    }
    defaults.update(overrides)
    return [patch.object(main, name, value) for name, value in defaults.items()]


def _run_limits(request, *, whitelisted=False, **overrides):
    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in _limits_patches(**overrides):
            stack.enter_context(p)
        stack.enter_context(patch.object(main, "is_ip_whitelisted", return_value=whitelisted))
        return main._check_rate_limits(request, TEST_IP, False)


def test_whitelisted_client_is_logged_without_the_ip(caplog):
    request = make_request(path="/v1/records/daily/london/01-01", method="GET")
    assert _run_limits(request, whitelisted=True) is None
    assert_logged_without_ip(caplog, "WHITELISTED CLIENT")


def test_non_vc_endpoint_skip_is_logged_without_the_ip(caplog):
    request = make_request(path="/health", method="GET")
    assert _run_limits(request) is None
    assert_logged_without_ip(caplog, "NON-VC ENDPOINT")


def test_rate_limit_exceeded_is_logged_without_the_ip(caplog):
    monitor = MagicMock()
    monitor.check_request_rate.return_value = (False, "too many requests")
    request = make_request(path="/v1/records/daily/london/01-01", method="GET")
    response = _run_limits(request, request_monitor=monitor)
    assert response.status_code == 429
    assert_logged_without_ip(caplog, "RATE LIMIT EXCEEDED")


def test_allowed_request_checks_are_logged_without_the_ip(caplog):
    request_monitor, location_monitor = MagicMock(), MagicMock()
    request_monitor.check_request_rate.return_value = (True, None)
    location_monitor.check_location_diversity.return_value = (True, None)
    request = make_request(path="/v1/records/daily/london/01-01", method="GET")
    assert _run_limits(request, request_monitor=request_monitor, location_monitor=location_monitor) is None
    assert_logged_without_ip(caplog, "RATE LIMIT CHECK")
    assert_logged_without_ip(caplog, "LOCATION DIVERSITY CHECK")


def test_location_diversity_limit_is_logged_without_the_ip(caplog):
    request_monitor, location_monitor = MagicMock(), MagicMock()
    request_monitor.check_request_rate.return_value = (True, None)
    location_monitor.check_location_diversity.return_value = (False, "too many places")
    request = make_request(path="/v1/records/daily/london/01-01", method="GET")
    response = _run_limits(request, request_monitor=request_monitor, location_monitor=location_monitor)
    assert response.status_code == 429
    assert_logged_without_ip(caplog, "LOCATION DIVERSITY LIMIT")
