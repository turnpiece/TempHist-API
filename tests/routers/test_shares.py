"""Tests for social sharing endpoints.

Covers:
- POST /v1/shares  — create share (requires Firebase auth)
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
    with patch.dict("os.environ", {
        "VISUAL_CROSSING_API_KEY": "test_key",
        "CACHE_ENABLED": "true",
        "API_ACCESS_TOKEN": "test_api_token",
    }):
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
        with patch("routers.og_image._render_chart", return_value=_STUB_PNG), \
             patch("routers.og_image._get_bundle_records", return_value=[{"year": 2024, "temperature": 10.0}]):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content[:4] == b"\x89PNG"

    def test_og_image_cache_control_header(self, client, mock_redis):
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        with patch("routers.og_image._render_chart", return_value=_STUB_PNG), \
             patch("routers.og_image._get_bundle_records", return_value=[{"year": 2024, "temperature": 10.0}]):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert "public" in response.headers.get("cache-control", "")

    def test_og_image_placeholder_when_no_records(self, client, mock_redis):
        """Returns a placeholder PNG (not 404) when records are missing."""
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        with patch("routers.og_image._render_placeholder", return_value=_STUB_PNG), \
             patch("routers.og_image._get_bundle_records", return_value=None):
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
        with patch("routers.og_image._render_chart", return_value=_STUB_PNG), \
             patch("routers.og_image._get_bundle_records", return_value=[{"year": 2024, "temperature": -12.0}]):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert response.status_code == 200
        assert response.content[:4] == b"\x89PNG"

    def test_og_image_no_auth_required(self, client, mock_redis):
        """Public endpoint — crawlers have no credentials."""
        mock_redis.get.return_value = json.dumps(VALID_SHARE).encode()
        with patch("routers.og_image._render_chart", return_value=_STUB_PNG), \
             patch("routers.og_image._get_bundle_records", return_value=[{"year": 2024, "temperature": 10.0}]):
            response = client.get("/v1/og/aB3xY7qZ.png")
        assert response.status_code == 200
