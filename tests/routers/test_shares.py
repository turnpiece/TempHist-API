"""Tests for social sharing endpoints.

Covers:
- POST /v1/shares  — create share (requires Firebase auth)
- GET /v1/shares  — list recent shares (public, deduplicated)
- GET /v1/shares/{share_id}  — retrieve share metadata (public)
- GET /v1/og/{share_id}.png  — OG preview image (public)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app as main_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(main_app) as c:
        yield c


@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(
        "os.environ",
        {
            "CACHE_ENABLED": "true",
            "API_ACCESS_TOKEN": "test_api_token",
        },
    ):
        yield


@pytest.fixture(autouse=True)
def mock_firebase():
    """Stub Firebase token verification so auth middleware passes."""
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "testuser"}):
        yield


@pytest.fixture
def mock_redis():
    """Return a MagicMock Redis client and override the dependency."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None  # cache miss by default
    redis_mock.setex.return_value = True
    with patch("routers.dependencies._redis_client", redis_mock):
        yield redis_mock


VALID_SHARE = {
    "id": "aB3xY7qZ",
    "location": "London, England, United Kingdom",
    "period": "yearly",
    "identifier": "04-11",
    "ref_year": 2024,
    "unit": "celsius",
    "created_at": "2024-04-11T10:00:00+00:00",
}

VALID_CREATE_BODY = {
    "location": "London, England, United Kingdom",
    "period": "yearly",
    "identifier": "04-11",
    "ref_year": 2024,
    "unit": "celsius",
    "latitude": 51.5074,
    "longitude": -0.1278,
}


# ---------------------------------------------------------------------------
# POST /v1/shares
# ---------------------------------------------------------------------------


class TestCreateShare:
    def test_create_share_success(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.create_share.return_value = {
            "id": "aB3xY7qZ",
            "url": "https://temphist.com/s/aB3xY7qZ",
        }
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.post(
                "/v1/shares",
                json=VALID_CREATE_BODY,
                headers={"Authorization": "Bearer firebase-token"},
            )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "aB3xY7qZ"
        assert "url" in data

    def test_create_share_requires_auth(self, client, mock_redis):
        """Unauthenticated request should be rejected."""
        response = client.post("/v1/shares", json=VALID_CREATE_BODY)
        assert response.status_code == 401

    def test_create_share_missing_location(self, client, mock_redis):
        body = {**VALID_CREATE_BODY}
        del body["location"]
        response = client.post(
            "/v1/shares",
            json=body,
            headers={"Authorization": "Bearer firebase-token"},
        )
        assert response.status_code == 422

    def test_create_share_invalid_period(self, client, mock_redis):
        body = {**VALID_CREATE_BODY, "period": "hourly"}
        response = client.post(
            "/v1/shares",
            json=body,
            headers={"Authorization": "Bearer firebase-token"},
        )
        assert response.status_code == 422

    def test_create_share_invalid_identifier_format(self, client, mock_redis):
        """Identifier must be MM-DD."""
        body = {**VALID_CREATE_BODY, "identifier": "2024-04-11"}
        response = client.post(
            "/v1/shares",
            json=body,
            headers={"Authorization": "Bearer firebase-token"},
        )
        assert response.status_code == 422

    def test_create_share_store_unavailable(self, client, mock_redis):
        """503 when the share store returns None (e.g. no Postgres)."""
        store_mock = AsyncMock()
        store_mock.create_share.return_value = None
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.post(
                "/v1/shares",
                json=VALID_CREATE_BODY,
                headers={"Authorization": "Bearer firebase-token"},
            )
        assert response.status_code == 503

    def test_create_share_all_periods(self, client, mock_redis):
        """All valid period values should be accepted."""
        store_mock = AsyncMock()
        store_mock.create_share.return_value = {
            "id": "aB3xY7qZ",
            "url": "https://temphist.com/s/aB3xY7qZ",
        }
        for period in ("daily", "weekly", "monthly", "yearly"):
            body = {**VALID_CREATE_BODY, "period": period}
            with patch("routers.shares.get_share_store", return_value=store_mock):
                response = client.post(
                    "/v1/shares",
                    json=body,
                    headers={"Authorization": "Bearer firebase-token"},
                )
            assert response.status_code == 201, f"Failed for period={period}"

    def test_create_share_fahrenheit_unit(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.create_share.return_value = {
            "id": "aB3xY7qZ",
            "url": "https://temphist.com/s/aB3xY7qZ",
        }
        body = {**VALID_CREATE_BODY, "unit": "fahrenheit"}
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.post(
                "/v1/shares",
                json=body,
                headers={"Authorization": "Bearer firebase-token"},
            )
        assert response.status_code == 201

    def test_create_share_invalid_unit(self, client, mock_redis):
        body = {**VALID_CREATE_BODY, "unit": "kelvin"}
        response = client.post(
            "/v1/shares",
            json=body,
            headers={"Authorization": "Bearer firebase-token"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/shares  (public feed)
# ---------------------------------------------------------------------------

SHARE_LIST = [
    {
        "id": "aB3xY7qZ",
        "location": "London, England, United Kingdom",
        "period": "yearly",
        "identifier": "04-11",
        "ref_year": 2024,
        "unit": "celsius",
        "created_at": "2024-04-11T10:00:00+00:00",
        "og_image_url": "/v1/og/aB3xY7qZ.png",
        "share_url": "/s/aB3xY7qZ",
        "latitude": 51.5074,
        "longitude": -0.1278,
    },
    {
        "id": "cD4eF8gH",
        "location": "Paris, Île-de-France, France",
        "period": "monthly",
        "identifier": "04-11",
        "ref_year": 2024,
        "unit": "celsius",
        "created_at": "2024-04-10T08:00:00+00:00",
        "og_image_url": "/v1/og/cD4eF8gH.png",
        "share_url": "/s/cD4eF8gH",
        "latitude": 48.8566,
        "longitude": 2.3522,
    },
]


class TestListShares:
    def test_list_shares_success(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.list_shares.return_value = SHARE_LIST
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares")
        assert response.status_code == 200
        data = response.json()
        assert "shares" in data
        assert len(data["shares"]) == 2
        assert data["limit"] == 20
        assert data["offset"] == 0

    def test_list_shares_period_filter(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.list_shares.return_value = [SHARE_LIST[0]]
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares?period=yearly")
        assert response.status_code == 200
        store_mock.list_shares.assert_called_once_with(period="yearly", limit=20, offset=0)

    def test_list_shares_pagination(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.list_shares.return_value = []
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares?limit=5&offset=10")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 5
        assert data["offset"] == 10
        store_mock.list_shares.assert_called_once_with(period=None, limit=5, offset=10)

    def test_list_shares_limit_max(self, client, mock_redis):
        response = client.get("/v1/shares?limit=999")
        assert response.status_code == 422

    def test_list_shares_limit_min(self, client, mock_redis):
        response = client.get("/v1/shares?limit=0")
        assert response.status_code == 422

    def test_list_shares_invalid_period(self, client, mock_redis):
        response = client.get("/v1/shares?period=hourly")
        assert response.status_code == 422

    def test_list_shares_store_unavailable(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.list_shares.return_value = None
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares")
        assert response.status_code == 503

    def test_list_shares_no_auth_required(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.list_shares.return_value = SHARE_LIST
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares")
        assert response.status_code == 200

    def test_list_shares_empty_result(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.list_shares.return_value = []
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares")
        assert response.status_code == 200
        assert response.json()["shares"] == []

    def test_list_shares_response_includes_og_url(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.list_shares.return_value = SHARE_LIST
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares")
        first = response.json()["shares"][0]
        assert "og_image_url" in first
        assert "share_url" in first


# ---------------------------------------------------------------------------
# GET /v1/shares/{share_id}
# ---------------------------------------------------------------------------


class TestGetShare:
    def test_get_share_from_cache(self, client, mock_redis):
        """Returns share served from Redis cache."""
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        response = client.get("/v1/shares/aB3xY7qZ")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "aB3xY7qZ"
        assert data["location"] == VALID_SHARE["location"]
        assert data["unit"] == "celsius"

    def test_get_share_from_store(self, client, mock_redis):
        """Cache miss falls back to the Postgres store."""
        mock_redis.get.return_value = None
        store_mock = AsyncMock()
        store_mock.get_share.return_value = VALID_SHARE
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares/aB3xY7qZ")
        assert response.status_code == 200
        assert response.json()["id"] == "aB3xY7qZ"

    def test_get_share_populates_cache(self, client, mock_redis):
        """A Postgres hit should write to Redis for future requests."""
        mock_redis.get.return_value = None
        store_mock = AsyncMock()
        store_mock.get_share.return_value = VALID_SHARE
        with patch("routers.shares.get_share_store", return_value=store_mock):
            client.get("/v1/shares/aB3xY7qZ")
        mock_redis.setex.assert_called_once()

    def test_get_share_not_found(self, client, mock_redis):
        mock_redis.get.return_value = None
        store_mock = AsyncMock()
        store_mock.get_share.return_value = None
        with patch("routers.shares.get_share_store", return_value=store_mock):
            response = client.get("/v1/shares/aB3xY7qZ")
        assert response.status_code == 404

    def test_get_share_invalid_id_too_short(self, client, mock_redis):
        response = client.get("/v1/shares/abc")
        assert response.status_code == 404

    def test_get_share_invalid_id_non_alnum(self, client, mock_redis):
        response = client.get("/v1/shares/aB3xY7q!")
        assert response.status_code == 404

    def test_get_share_no_auth_required(self, client, mock_redis):
        """Public endpoint — no Authorization header needed."""
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        response = client.get("/v1/shares/aB3xY7qZ")
        # Should succeed without any auth header
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# GET /v1/og/{share_id}.png
# ---------------------------------------------------------------------------

# Minimal 1x1 transparent PNG — used as a stand-in when matplotlib is unavailable
_STUB_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestOgImage:
    def _make_bundle_redis_value(self, unit="celsius"):
        """Redis value for the bundle key: {"records": [...]}"""
        base = 10.0
        records = [{"year": y, "temperature": base + (y - 2000) * 0.1} for y in range(2000, 2025)]
        return json.dumps({"records": records}).encode()

    def test_og_image_returns_png(self, client, mock_redis):
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        with (
            patch("routers.og_image._render_chart", return_value=_STUB_PNG),
            patch("routers.og_image._get_bundle_records", return_value=[{"year": 2024, "temperature": 10.0}]),
        ):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:4] == b"\x89PNG"

    def test_og_image_cache_control_header(self, client, mock_redis):
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        with (
            patch("routers.og_image._render_chart", return_value=_STUB_PNG),
            patch("routers.og_image._get_bundle_records", return_value=[{"year": 2024, "temperature": 10.0}]),
        ):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert "public" in response.headers.get("cache-control", "")

    def test_og_image_placeholder_when_no_records(self, client, mock_redis):
        """Returns a placeholder PNG (not 404) when records are missing."""
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        with (
            patch("routers.og_image._render_placeholder", return_value=_STUB_PNG),
            patch("routers.og_image._get_bundle_records", return_value=None),
        ):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_og_image_not_found_for_unknown_share(self, client, mock_redis):
        mock_redis.get.return_value = None
        store_mock = AsyncMock()
        store_mock.get_share.return_value = None
        with patch("routers.og_image.get_share_store", return_value=store_mock):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert response.status_code == 404

    def test_og_image_invalid_id(self, client, mock_redis):
        response = client.get("/v1/og/!!!bad!!.png")
        assert response.status_code == 404

    def test_og_image_fahrenheit_renders(self, client, mock_redis):
        """Fahrenheit share should render without error."""
        share_f = {**VALID_SHARE, "unit": "fahrenheit"}
        mock_redis.get.return_value = json.dumps(share_f).encode()
        with (
            patch("routers.og_image._render_chart", return_value=_STUB_PNG),
            patch("routers.og_image._get_bundle_records", return_value=[{"year": 2024, "temperature": -12.0}]),
        ):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert response.status_code == 200
        assert response.content[:4] == b"\x89PNG"

    def test_og_image_no_auth_required(self, client, mock_redis):
        """Public endpoint — crawlers have no credentials."""
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        with (
            patch("routers.og_image._render_chart", return_value=_STUB_PNG),
            patch("routers.og_image._get_bundle_records", return_value=[{"year": 2024, "temperature": 10.0}]),
        ):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Proximity deduplication — unit tests against ShareStore._haversine_km
# and list_shares dedup logic
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch as _patch  # noqa: E402

from routers.shares import _resolve_location_name  # noqa: E402
from utils.share_store import ShareStore, _haversine_km  # noqa: E402

_FAKE_LOCATIONS = [
    SimpleNamespace(id="cape_town", name="Cape Town", admin1="Western Cape", country_name="South Africa"),
    SimpleNamespace(id="london", name="London", admin1="England", country_name="United Kingdom"),
    SimpleNamespace(id="tokyo", name="Tokyo", admin1=None, country_name="Japan"),
]


class TestResolveLocationName:
    def _resolve(self, location: str) -> str:
        with _patch("routers.shares.locations_data", _FAKE_LOCATIONS):
            return _resolve_location_name(location)

    def test_uppercase_id_resolved(self):
        assert self._resolve("CAPE_TOWN") == "Cape Town, Western Cape, South Africa"

    def test_lowercase_id_resolved(self):
        assert self._resolve("cape_town") == "Cape Town, Western Cape, South Africa"

    def test_mixed_case_id_resolved(self):
        assert self._resolve("Cape_Town") == "Cape Town, Western Cape, South Africa"

    def test_location_without_admin1(self):
        assert self._resolve("tokyo") == "Tokyo, Japan"

    def test_display_name_passthrough(self):
        result = self._resolve("Cape Town, Western Cape, South Africa")
        assert result == "Cape Town, Western Cape, South Africa"

    def test_unknown_id_passthrough(self):
        assert self._resolve("UNKNOWN_PLACE") == "UNKNOWN_PLACE"


class TestCreateShareLocationResolution:
    def test_location_id_is_resolved_before_store(self, client, mock_redis):
        """POST with a raw location ID must store the human-readable display name."""
        store_mock = AsyncMock()
        store_mock.create_share.return_value = {"id": "aB3xY7qZ", "url": "https://temphist.com/s/aB3xY7qZ"}
        body = {**VALID_CREATE_BODY, "location": "CAPE_TOWN"}
        with (
            _patch("routers.shares.get_share_store", return_value=store_mock),
            _patch("routers.shares.locations_data", _FAKE_LOCATIONS),
        ):
            response = client.post(
                "/v1/shares",
                json=body,
                headers={"Authorization": "Bearer firebase-token"},
            )
        assert response.status_code == 201
        store_mock.create_share.assert_called_once()
        assert store_mock.create_share.call_args.kwargs["location"] == "Cape Town, Western Cape, South Africa"

    def test_display_name_stored_unchanged(self, client, mock_redis):
        store_mock = AsyncMock()
        store_mock.create_share.return_value = {"id": "aB3xY7qZ", "url": "https://temphist.com/s/aB3xY7qZ"}
        with (
            _patch("routers.shares.get_share_store", return_value=store_mock),
            _patch("routers.shares.locations_data", _FAKE_LOCATIONS),
        ):
            response = client.post(
                "/v1/shares",
                json=VALID_CREATE_BODY,
                headers={"Authorization": "Bearer firebase-token"},
            )
        assert response.status_code == 201
        assert store_mock.create_share.call_args.kwargs["location"] == "London, England, United Kingdom"


class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine_km(51.5, -0.1, 51.5, -0.1) == 0.0

    def test_london_to_paris_approx(self):
        # London (51.51, -0.13) to Paris (48.86, 2.35) ≈ 340 km
        km = _haversine_km(51.51, -0.13, 48.86, 2.35)
        assert 330 < km < 360

    def test_london_to_greater_london_within_50km(self):
        # City of London vs Greater London centroid — a few km apart
        km = _haversine_km(51.5074, -0.1278, 51.5085, -0.0956)
        assert km < 50

    def test_london_to_brighton_outside_50km(self):
        # Brighton is ~77 km south of London
        km = _haversine_km(51.5074, -0.1278, 50.8225, -0.1372)
        assert km > 50


class TestListSharesProximityDedup:
    """Tests for the Python-side proximity deduplication in ShareStore.list_shares."""

    def _make_row(self, share_id, location, period, identifier, lat, lon, created_at=None):
        """Build a fake asyncpg-like Record dict."""
        return {
            "id": share_id,
            "location": location,
            "period": period,
            "identifier": identifier,
            "ref_year": 2024,
            "unit": "celsius",
            "created_at": created_at or datetime(2024, 4, 11, 10, 0, 0, tzinfo=timezone.utc),
            "latitude": lat,
            "longitude": lon,
        }

    def _run(self, coro):
        return asyncio.run(coro)

    def _mock_store_with_rows(self, rows):
        """Return a ShareStore whose DB fetch returns the given rows."""
        store = ShareStore.__new__(ShareStore)
        store._disabled = False
        store._pool = MagicMock()

        class FakeConn:
            async def fetch(self, *args, **kwargs):
                return rows

        class FakePool:
            def acquire(self):
                return self

            async def __aenter__(self):
                return FakeConn()

            async def __aexit__(self, *args):
                pass

        store._pool = FakePool()
        return store

    def test_nearby_shares_deduplicated(self):
        """Two shares within 50 km with same period+identifier → one result."""
        rows = [
            self._make_row(
                "id000001",
                "London, England, United Kingdom",
                "yearly",
                "04-11",
                51.5074,
                -0.1278,
                datetime(2024, 4, 11, tzinfo=timezone.utc),
            ),
            self._make_row(
                "id000002",
                "Greater London, England, United Kingdom",
                "yearly",
                "04-11",
                51.5085,
                -0.0956,
                datetime(2024, 4, 10, tzinfo=timezone.utc),
            ),
        ]
        store = self._mock_store_with_rows(rows)
        result = self._run(store.list_shares())
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "id000001"  # most-recent kept

    def test_distant_shares_both_returned(self):
        """Two shares >50 km apart with same period+identifier → both returned."""
        rows = [
            self._make_row("id000001", "London, England, United Kingdom", "yearly", "04-11", 51.5074, -0.1278),
            self._make_row("id000002", "Brighton, England, United Kingdom", "yearly", "04-11", 50.8225, -0.1372),
        ]
        store = self._mock_store_with_rows(rows)
        result = self._run(store.list_shares())
        assert result is not None
        assert len(result) == 2

    def test_null_coordinates_exact_string_fallback(self):
        """Shares with NULL coords fall back to exact string deduplication."""
        rows = [
            self._make_row(
                "id000001",
                "London, England, United Kingdom",
                "yearly",
                "04-11",
                None,
                None,
                datetime(2024, 4, 11, tzinfo=timezone.utc),
            ),
            self._make_row(
                "id000002",
                "London, England, United Kingdom",
                "yearly",
                "04-11",
                None,
                None,
                datetime(2024, 4, 10, tzinfo=timezone.utc),
            ),
        ]
        store = self._mock_store_with_rows(rows)
        result = self._run(store.list_shares())
        assert result is not None
        assert len(result) == 1

    def test_null_coords_different_strings_not_deduped(self):
        """NULL-coord shares with different strings are not merged."""
        rows = [
            self._make_row("id000001", "London, England, United Kingdom", "yearly", "04-11", None, None),
            self._make_row("id000002", "Greater London, England, United Kingdom", "yearly", "04-11", None, None),
        ]
        store = self._mock_store_with_rows(rows)
        result = self._run(store.list_shares())
        assert result is not None
        assert len(result) == 2

    def test_private_lat_lon_fields_stripped_from_output(self):
        """The _lat/_lon fields used during dedup must not appear in the response."""
        rows = [
            self._make_row("id000001", "London, England, United Kingdom", "yearly", "04-11", 51.5074, -0.1278),
        ]
        store = self._mock_store_with_rows(rows)
        result = self._run(store.list_shares())
        assert result is not None
        assert len(result) == 1
        assert "_lat" not in result[0]
        assert "_lon" not in result[0]
