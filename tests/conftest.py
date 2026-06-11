"""Shared pytest fixtures.

`close_module_http_clients` exists to drain module-level aiohttp.ClientSession
singletons at the end of the test session. Without it, any test that touches
open_meteo / visual_crossing / mapbox prints
``Unclosed client session`` / ``Unclosed connector`` at process exit, because
those modules only close their clients via an explicit close_client() that
production code calls from FastAPI's shutdown hook — never from tests.
"""

import asyncio

import pytest


@pytest.fixture(scope="session", autouse=True)
def close_module_http_clients():
    yield

    async def _close_all() -> None:
        # Each close call is best-effort; modules that were never imported or
        # whose client was never instantiated are silently skipped.
        try:
            from utils import open_meteo_client

            await open_meteo_client.close_client()
        except Exception:
            pass

        try:
            from utils import visual_crossing_client

            await visual_crossing_client.close_client()
        except Exception:
            pass

        try:
            from routers import locations as locations_router

            client = getattr(locations_router, "_mapbox_client", None)
            if client is not None and not client.closed:
                await client.close()
            locations_router._mapbox_client = None
        except Exception:
            pass

    try:
        asyncio.run(_close_all())
    except RuntimeError:
        # If a loop is already running (rare in teardown), fall back to a fresh
        # loop. We don't want teardown to crash the test session.
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_close_all())
        finally:
            loop.close()
