"""Tests for cache warming HTTP-layer behaviour (issue #69).

Asserts that one `warm_all_locations` cycle:
- creates a single shared aiohttp.ClientSession (not one per URL), and
- no longer fans out to the `/average`, `/trend`, `/summary` subresources.
"""

from unittest.mock import MagicMock

import aiohttp
import pytest

from cache import warming as warming_module
from cache.warming import CacheWarmer


class _FakeResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    """Minimal stand-in for aiohttp.ClientSession that records every URL."""

    def __init__(self, *args, **kwargs):
        # Record the captured URLs onto the class so the test can introspect
        # them without holding a reference to the instance.
        self.urls: list[str] = []
        type(self).instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, **kwargs):
        self.urls.append(url)
        return _FakeResponse()


@pytest.fixture
def fake_session_cls(monkeypatch):
    """Patch aiohttp.ClientSession used inside cache.warming with a recorder."""

    class _RecordingSession(_FakeSession):
        instances: list = []

    class _FakeConnector:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeTimeout:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(warming_module.aiohttp, "ClientSession", _RecordingSession)
    # Stub the connector/timeout too so we don't leave real OS resources
    # dangling — the fake session never uses them anyway.
    monkeypatch.setattr(warming_module.aiohttp, "TCPConnector", _FakeConnector)
    monkeypatch.setattr(warming_module.aiohttp, "ClientTimeout", _FakeTimeout)
    return _RecordingSession


@pytest.fixture
def warmer():
    redis_client = MagicMock()
    redis_client.ping.return_value = True
    # No usage tracker → falls back to preapproved list.
    w = CacheWarmer(redis_client, usage_tracker=None)
    # Override the heavy preapproved file read with a tiny fixed list.
    w.get_preapproved_locations = lambda: ["London, England, UK", "Paris, Île-de-France, France"]
    return w


@pytest.fixture
def small_run(monkeypatch):
    """Keep the cycle tiny so URL counts stay readable in assertions."""
    monkeypatch.setattr(warming_module, "CACHE_WARMING_ENABLED", True)
    monkeypatch.setattr(warming_module, "API_ACCESS_TOKEN", "test-token")
    monkeypatch.setattr(warming_module, "BASE_URL", "http://test.invalid")
    monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER1_SIZE", 2)
    monkeypatch.setattr(warming_module, "CACHE_WARMING_TIER2_MAX", 0)
    monkeypatch.setattr(warming_module, "CACHE_WARMING_DAYS_BACK", 1)
    monkeypatch.setattr(warming_module, "CACHE_WARMING_CONCURRENT_REQUESTS", 2)


@pytest.mark.asyncio
async def test_single_session_per_run(warmer, fake_session_cls, small_run):
    result = await warmer.warm_all_locations()

    assert result["status"] == "completed"
    # Exactly one ClientSession is created for the whole cycle.
    assert len(fake_session_cls.instances) == 1


@pytest.mark.asyncio
async def test_no_subresource_warming(warmer, fake_session_cls, small_run):
    await warmer.warm_all_locations()

    assert len(fake_session_cls.instances) == 1
    urls = fake_session_cls.instances[0].urls
    assert urls, "expected at least one URL to be warmed"
    for url in urls:
        # The base record endpoint must still be warmed, but never the
        # average/trend/summary subresources — those re-derive data from the
        # bundle cache that the base call already populated.
        assert not url.endswith("/average"), url
        assert not url.endswith("/trend"), url
        assert not url.endswith("/summary"), url
