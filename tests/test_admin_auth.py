"""Tests for admin authentication on operational endpoints."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import API_ACCESS_TOKEN, app  # noqa: E402

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def admin_key_env():
    with patch.dict("os.environ", {"ADMIN_API_KEY": ADMIN_KEY}, clear=False):
        with patch("config.ADMIN_API_KEY", ADMIN_KEY):
            with patch("utils.admin_auth.ADMIN_API_KEY", ADMIN_KEY):
                with patch("main.ADMIN_API_KEY", ADMIN_KEY):
                    yield ADMIN_KEY


@pytest.mark.parametrize(
    "path,method",
    [
        ("/usage-stats", "GET"),
        ("/usage-stats/london", "GET"),
        ("/cache-stats", "GET"),
        ("/rate-limit-stats", "GET"),
        ("/cache/clear", "DELETE"),
    ],
)
def test_admin_endpoints_reject_missing_key(client, admin_key_env, path, method):
    response = client.request(method, path)
    assert response.status_code == 401
    assert "admin key" in response.json()["detail"].lower()


def test_admin_endpoints_reject_api_access_token_only(client, admin_key_env):
    headers = {"Authorization": f"Bearer {API_ACCESS_TOKEN}"}
    response = client.get("/usage-stats", headers=headers)
    assert response.status_code == 401


def test_admin_endpoints_reject_firebase_token_only(client, admin_key_env):
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "testuser"}):
        response = client.get(
            "/usage-stats",
            headers={"Authorization": "Bearer firebase-id-token"},
        )
    assert response.status_code == 401


def test_admin_endpoints_accept_admin_key(client, admin_key_env):
    response = client.get("/usage-stats", headers={"X-Admin-Key": ADMIN_KEY})
    assert response.status_code == 200


def test_rate_limit_stats_accepts_admin_key(client, admin_key_env):
    from config import RATE_LIMIT_ENABLED

    response = client.get("/rate-limit-stats", headers={"X-Admin-Key": ADMIN_KEY})
    assert response.status_code == 200
    data = response.json()
    if not RATE_LIMIT_ENABLED:
        pytest.skip("Rate limiting disabled in this environment — content assertions require it enabled")
    assert "whitelisted_ips" in data
    assert "blacklisted_ips" in data


def test_records_endpoint_still_accepts_api_access_token(client, admin_key_env):
    with patch(
        "routers.v1_records.get_temperature_data_v1",
        return_value={"data": [], "metadata": {}},
    ):
        response = client.get(
            "/v1/records/daily/london/06-09",
            headers={"Authorization": f"Bearer {API_ACCESS_TOKEN}"},
        )
    # Auth passes; response depends on mocked handler / validation
    assert response.status_code != 401


def test_admin_endpoint_returns_503_when_not_configured(client):
    with patch("config.ADMIN_API_KEY", None):
        with patch("utils.admin_auth.ADMIN_API_KEY", None):
            with patch("main.ADMIN_API_KEY", None):
                response = client.get("/usage-stats", headers={"X-Admin-Key": "any"})
    assert response.status_code == 503


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/usage-stats", True),
        ("/usage-stats/london", True),
        ("/cache-stats/health", True),
        ("/rate-limit-stats", True),
        ("/cache/clear", True),
        ("/cache/invalidate/pattern", True),
        ("/admin/clear-job-queue", True),
        ("/v1/records/daily/london/06-09", False),
        ("/rate-limit-status", False),
        ("/health", False),
    ],
)
def test_is_admin_path(path, expected):
    from utils.admin_auth import is_admin_path

    method = "DELETE" if path == "/cache/clear" else "GET"
    assert is_admin_path(path, method) is expected
