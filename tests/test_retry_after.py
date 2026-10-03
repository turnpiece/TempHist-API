"""429 and 503 responses carry Retry-After, and the exception handler forwards headers set on an HTTPException."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from exceptions import register_exception_handlers
from main import app as main_app

AUTH = {"Authorization": "Bearer test-token"}
LIMITED = AsyncMock(return_value=(False, "Rate limit exceeded: 60 requests per minute"))
ALLOWED = AsyncMock(return_value=(True, "OK"))


@pytest.fixture(scope="module")
def client():
    with TestClient(main_app) as c:
        yield c


@pytest.fixture(autouse=True)
def _mock_env_and_auth():
    with patch.dict("os.environ", {"CACHE_ENABLED": "true", "API_ACCESS_TOKEN": "test_api_token"}):
        with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "testuser"}):
            yield


class TestExceptionHandlerForwardsHeaders:
    @pytest.fixture
    def tiny_client(self):
        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/slow")
        def slow():
            raise HTTPException(status_code=429, detail="slow down", headers={"Retry-After": "7"})

        @app.get("/plain")
        def plain():
            raise HTTPException(status_code=404, detail="nope")

        return TestClient(app)

    def test_headers_set_on_the_exception_reach_the_client(self, tiny_client):
        response = tiny_client.get("/slow")
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "7"

    def test_body_is_still_the_standard_error_response(self, tiny_client):
        body = tiny_client.get("/slow").json()
        assert body["error"] == "RATE_LIMIT_EXCEEDED"
        assert body["message"] == "slow down"

    def test_exceptions_without_headers_are_unchanged(self, tiny_client):
        response = tiny_client.get("/plain")
        assert response.status_code == 404
        assert "retry-after" not in response.headers

    def test_method_not_allowed_now_reports_allow(self, client):
        response = client.post("/health")
        assert response.status_code == 405
        assert "GET" in response.headers["Allow"]


@pytest.mark.parametrize(
    "method, path, kwargs",
    [
        ("get", "/v1/locations/preapproved", {}),
        ("get", "/v1/locations/search?q=lon", {}),
        ("get", "/v1/locations/popular", {}),
        ("get", "/v1/locations/popular/display-strings", {}),
        ("post", "/v1/locations/selections", {"json": {"location_id": "london"}}),
    ],
)
def test_locations_429_sends_retry_after_for_the_rate_limit_window(client, method, path, kwargs):
    from routers.locations import RATE_LIMIT_WINDOW

    with patch("routers.locations.check_rate_limit", new=LIMITED):
        response = getattr(client, method)(path, headers=AUTH, **kwargs)

    assert response.status_code == 429
    assert response.headers["Retry-After"] == str(RATE_LIMIT_WINDOW)
    assert response.json()["error"] == "RATE_LIMIT_EXCEEDED"


def test_locations_503_while_data_is_loading_sends_retry_after(client):
    with (
        patch("routers.locations.check_rate_limit", new=ALLOWED),
        patch("routers.locations.MAPBOX_TOKEN", None),
        patch("routers.locations.locations_data", []),
    ):
        response = client.get("/v1/locations/search?q=lon", headers=AUTH)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"


class TestShareStoreUnavailable:
    @pytest.fixture
    def store_down(self):
        store = AsyncMock()
        store.list_shares.return_value = None
        store.create_share.return_value = None
        with (
            patch("routers.shares.get_share_store", return_value=store),
            patch("routers.dependencies._redis_client", MagicMock()),
        ):
            yield

    def test_list_shares_503_sends_retry_after(self, client, store_down):
        response = client.get("/v1/shares")
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "30"

    def test_create_share_503_sends_retry_after(self, client, store_down):
        body = {"location": "london", "period": "daily", "identifier": "01-15", "ref_year": 2025}
        response = client.post("/v1/shares", json=body, headers=AUTH)
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "30"
