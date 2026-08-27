"""Tests for utils.open_meteo_client."""

import asyncio
import logging
from datetime import date, timedelta
from unittest.mock import MagicMock

import aiohttp
import pytest

from utils import open_meteo_client


class FakeOpenMeteoStats:
    def __init__(self):
        self.calls = []
        self.attempts = []
        self.successes = []
        self.failures = []

    def record_call(self, endpoint=None):
        self.calls.append(endpoint)

    def record_attempt(self, endpoint=None):
        self.attempts.append(endpoint)

    def record_success(self, endpoint=None):
        self.successes.append(endpoint)

    def record_failure(self, reason, *, endpoint=None, terminal=True, timeout=False):
        self.failures.append(
            {
                "reason": reason,
                "endpoint": endpoint,
                "terminal": terminal,
                "timeout": timeout,
            }
        )


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status = status
        self._payload = payload or {}
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requested_urls = []

    def get(self, *args, **kwargs):
        if args:
            self.requested_urls.append(args[0])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def fake_get_client(responses):
    """Return a _get_client stand-in; the session it hands out is on `.session`."""
    session = FakeSession(responses)

    async def _fake_get_client():
        return session

    _fake_get_client.session = session
    return _fake_get_client


@pytest.fixture
def no_retry_sleep(monkeypatch):
    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


@pytest.mark.asyncio
async def test_fetch_timeline_for_location_caches_timezone_from_metadata(monkeypatch):
    async def fake_geocode(location):
        return 40.0, -105.0, None

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 10.0, "tempmax": 11.0, "tempmin": 9.0}],
            {"resolvedAddress": None, "timezone": "America/Denver", "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    location = "Nowhere Town, USA"
    days, metadata = await open_meteo_client.fetch_timeline_for_location(
        location, date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] == "America/Denver"
    assert len(days) == 1
    store_mock.assert_called_once_with(location, "America/Denver")


@pytest.mark.asyncio
async def test_fetch_timeline_for_location_uses_geocode_timezone_hint_when_metadata_missing(monkeypatch):
    async def fake_geocode(location):
        return 51.5, -0.1, "Europe/London"

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 12.0, "tempmax": 13.0, "tempmin": 11.0}],
            {"resolvedAddress": None, "timezone": None, "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    _days, metadata = await open_meteo_client.fetch_timeline_for_location(
        "London", date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] == "Europe/London"
    store_mock.assert_called_once_with("London", "Europe/London")


@pytest.mark.asyncio
async def test_fetch_timeline_for_location_skips_cache_when_no_timezone_known(monkeypatch):
    async def fake_geocode(location):
        return 0.0, 0.0, None

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 20.0, "tempmax": 21.0, "tempmin": 19.0}],
            {"resolvedAddress": None, "timezone": None, "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    _days, metadata = await open_meteo_client.fetch_timeline_for_location(
        "Mystery Place", date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] is None
    store_mock.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_single_date_caches_timezone_from_metadata(monkeypatch):
    async def fake_geocode(location):
        return 41.0, -87.0, None

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 7.0, "tempmax": 8.0, "tempmin": 6.0}],
            {"resolvedAddress": None, "timezone": "America/Chicago", "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    result = await open_meteo_client.fetch_single_date("Random Hamlet", "2024-06-01")

    assert "days" in result
    store_mock.assert_called_once_with("Random Hamlet", "America/Chicago")


@pytest.mark.asyncio
async def test_fetch_single_date_uses_geocode_timezone_hint_when_metadata_missing(monkeypatch):
    async def fake_geocode(location):
        return 35.0, 139.0, "Asia/Tokyo"

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 22.0, "tempmax": 23.0, "tempmin": 21.0}],
            {"resolvedAddress": None, "timezone": None, "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    store_mock = MagicMock()
    monkeypatch.setattr("cache.keys.store_location_timezone", store_mock)

    await open_meteo_client.fetch_single_date("Some Tokyo Suburb", "2024-06-01")

    store_mock.assert_called_once_with("Some Tokyo Suburb", "Asia/Tokyo")


@pytest.mark.asyncio
async def test_fetch_timeline_for_location_swallows_cache_errors(monkeypatch):
    async def fake_geocode(location):
        return 0.0, 0.0, None

    async def fake_fetch_days(lat, lon, start, end):
        return (
            [{"datetime": start.isoformat(), "temp": 5.0, "tempmax": 6.0, "tempmin": 4.0}],
            {"resolvedAddress": None, "timezone": "UTC", "latitude": lat, "longitude": lon},
        )

    monkeypatch.setattr(open_meteo_client, "geocode_location", fake_geocode)
    monkeypatch.setattr(open_meteo_client, "fetch_days", fake_fetch_days)

    def boom(*args, **kwargs):
        raise RuntimeError("redis exploded")

    monkeypatch.setattr("cache.keys.store_location_timezone", boom)

    days, metadata = await open_meteo_client.fetch_timeline_for_location(
        "Anywhere", date(2024, 6, 1), date(2024, 6, 1)
    )

    assert metadata["timezone"] == "UTC"
    assert len(days) == 1


@pytest.mark.asyncio
async def test_fetch_days_success_records_open_meteo_stats(monkeypatch):
    stats = FakeOpenMeteoStats()
    payload = {
        "latitude": 51.5,
        "longitude": -0.1,
        "timezone": "Europe/London",
        "daily": {
            "time": ["2024-06-01"],
            "temperature_2m_mean": [15.0],
            "temperature_2m_max": [18.0],
            "temperature_2m_min": [12.0],
        },
    }

    monkeypatch.setattr(open_meteo_client, "_get_client", fake_get_client([FakeResponse(200, payload)]))
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: stats)

    days, _metadata = await open_meteo_client.fetch_days(51.5, -0.1, date(2024, 6, 1), date(2024, 6, 1))

    assert len(days) == 1
    assert stats.calls == ["archive"]
    assert stats.attempts == ["archive"]
    assert stats.successes == ["archive"]
    assert stats.failures == []


@pytest.mark.asyncio
async def test_fetch_days_429_exhaustion_records_terminal_failure(monkeypatch, no_retry_sleep):
    stats = FakeOpenMeteoStats()
    responses = [
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(429, headers={"Retry-After": "0"}),
        FakeResponse(429, headers={"Retry-After": "0"}),
    ]

    monkeypatch.setattr(open_meteo_client, "_get_client", fake_get_client(responses))
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: stats)

    days, _metadata = await open_meteo_client.fetch_days(51.5, -0.1, date(2024, 6, 1), date(2024, 6, 1))

    assert days == []
    assert len(stats.attempts) == 3
    assert [failure["reason"] for failure in stats.failures] == [
        "rate_limited",
        "rate_limited",
        "rate_limit_exceeded",
    ]
    assert stats.failures[-1]["terminal"] is True


@pytest.mark.asyncio
async def test_fetch_days_timeout_exhaustion_records_timeouts(monkeypatch, no_retry_sleep):
    stats = FakeOpenMeteoStats()
    responses = [asyncio.TimeoutError("timeout"), asyncio.TimeoutError("timeout"), asyncio.TimeoutError("timeout")]

    monkeypatch.setattr(open_meteo_client, "_get_client", fake_get_client(responses))
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: stats)

    days, _metadata = await open_meteo_client.fetch_days(51.5, -0.1, date(2024, 6, 1), date(2024, 6, 1))

    assert days == []
    assert len(stats.attempts) == 3
    assert [failure["terminal"] for failure in stats.failures] == [False, False, True]
    assert all(failure["reason"] == "connection_timeout" for failure in stats.failures)
    assert all(failure["timeout"] is True for failure in stats.failures)


@pytest.mark.asyncio
async def test_fetch_days_non_200_records_http_failure_without_retry(monkeypatch):
    stats = FakeOpenMeteoStats()

    monkeypatch.setattr(open_meteo_client, "_get_client", fake_get_client([FakeResponse(503)]))
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: stats)

    days, _metadata = await open_meteo_client.fetch_days(51.5, -0.1, date(2024, 6, 1), date(2024, 6, 1))

    assert days == []
    assert len(stats.attempts) == 1
    assert stats.failures == [
        {
            "reason": "http_503",
            "endpoint": "archive",
            "terminal": True,
            "timeout": False,
        }
    ]


# ── API key handling ──────────────────────────────────────────────────────────


def _ok_payload(day="2024-06-01"):
    return {
        "latitude": 51.5,
        "longitude": -0.1,
        "timezone": "Europe/London",
        "daily": {
            "time": [day],
            "temperature_2m_mean": [15.0],
            "temperature_2m_max": [18.0],
            "temperature_2m_min": [12.0],
        },
    }


@pytest.mark.asyncio
async def test_fetch_days_omits_apikey_when_unset(monkeypatch):
    monkeypatch.setattr("config.OPEN_METEO_API_KEY", "")
    get_client = fake_get_client([FakeResponse(200, _ok_payload())])
    monkeypatch.setattr(open_meteo_client, "_get_client", get_client)
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: None)

    await open_meteo_client.fetch_days(51.5, -0.1, date(2024, 6, 1), date(2024, 6, 1))

    assert len(get_client.session.requested_urls) == 1
    assert "apikey" not in get_client.session.requested_urls[0]


@pytest.mark.asyncio
async def test_fetch_days_appends_apikey_to_archive_and_forecast(monkeypatch):
    monkeypatch.setattr("config.OPEN_METEO_API_KEY", "secret-key")
    # A range straddling the archive/forecast boundary issues one request to each.
    today = date.today()
    start = today - timedelta(days=30)
    get_client = fake_get_client([FakeResponse(200, _ok_payload()), FakeResponse(200, _ok_payload())])
    monkeypatch.setattr(open_meteo_client, "_get_client", get_client)
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: None)

    await open_meteo_client.fetch_days(51.5, -0.1, start, today)

    urls = get_client.session.requested_urls
    assert len(urls) == 2
    assert any("/v1/archive" in url for url in urls)
    assert any("/v1/forecast" in url for url in urls)
    assert all(url.endswith("&apikey=secret-key") for url in urls)


@pytest.mark.asyncio
async def test_fetch_days_url_encodes_apikey(monkeypatch):
    monkeypatch.setattr("config.OPEN_METEO_API_KEY", "key with/slash")
    get_client = fake_get_client([FakeResponse(200, _ok_payload())])
    monkeypatch.setattr(open_meteo_client, "_get_client", get_client)
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: None)

    await open_meteo_client.fetch_days(51.5, -0.1, date(2024, 6, 1), date(2024, 6, 1))

    assert get_client.session.requested_urls[0].endswith("&apikey=key%20with%2Fslash")


@pytest.mark.asyncio
async def test_fetch_days_never_logs_the_apikey(monkeypatch, caplog, no_retry_sleep):
    monkeypatch.setattr("config.OPEN_METEO_API_KEY", "secret-key")
    monkeypatch.setattr(open_meteo_client, "_get_client", fake_get_client([FakeResponse(503)]))
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: None)

    with caplog.at_level(logging.WARNING, logger=open_meteo_client.__name__):
        await open_meteo_client.fetch_days(51.5, -0.1, date(2024, 6, 1), date(2024, 6, 1))

    assert "OM returned status 503" in caplog.text
    assert "secret-key" not in caplog.text


@pytest.mark.asyncio
async def test_fetch_days_sanitizes_apikey_from_client_error_logs(monkeypatch, caplog, no_retry_sleep):
    monkeypatch.setattr("config.OPEN_METEO_API_KEY", "secret-key")
    # aiohttp errors can carry the resolved request URL, key and all.
    exc = aiohttp.ClientError("Cannot connect to https://customer-api.open-meteo.com/v1/forecast?apikey=secret-key")
    monkeypatch.setattr(open_meteo_client, "_get_client", fake_get_client([exc, exc, exc]))
    monkeypatch.setattr(open_meteo_client, "_get_open_meteo_stats", lambda: None)

    with caplog.at_level(logging.WARNING, logger=open_meteo_client.__name__):
        await open_meteo_client.fetch_days(51.5, -0.1, date(2024, 6, 1), date(2024, 6, 1))

    assert "secret-key" not in caplog.text
    assert "[REDACTED]" in caplog.text
