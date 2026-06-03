"""Tests for Firebase App Check enforcement modes in verify_token_middleware."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app as main_app


@pytest.fixture
def client():
    with TestClient(main_app) as c:
        yield c


def _firebase_headers(app_check_token=None):
    headers = {"Authorization": "Bearer fake-firebase-id-token"}
    if app_check_token is not None:
        headers["X-Firebase-AppCheck"] = app_check_token
    return headers


def _mock_weather():
    return patch(
        "routers.weather.get_weather_for_date",
        new_callable=AsyncMock,
        return_value={"days": [{"temp": 15.0, "tempmax": 17.0, "tempmin": 13.0}]},
    )


def test_off_mode_no_header(client):
    """When APP_CHECK_ENFORCEMENT=off, requests without App Check token are not blocked."""
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "u1"}):
        with patch("main.APP_CHECK_ENFORCEMENT", "off"):
            with _mock_weather():
                response = client.get(
                    "/weather/London/2024-05-15",
                    headers=_firebase_headers(),
                )
    assert response.status_code == 200


def test_monitor_mode_invalid_token(client):
    """When APP_CHECK_ENFORCEMENT=monitor, an invalid App Check token is logged but not blocked."""
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "u1"}):
        with patch("main.APP_CHECK_ENFORCEMENT", "monitor"):
            with patch("firebase_admin.app_check.verify_token", side_effect=Exception("bad token")):
                with _mock_weather():
                    response = client.get(
                        "/weather/London/2024-05-15",
                        headers=_firebase_headers(app_check_token="invalid-token"),
                    )
    assert response.status_code == 200


def test_enforce_mode_no_header(client):
    """When APP_CHECK_ENFORCEMENT=enforce, requests without App Check header are rejected."""
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "u1"}):
        with patch("main.APP_CHECK_ENFORCEMENT", "enforce"):
            response = client.get(
                "/weather/London/2024-05-15",
                headers=_firebase_headers(),
            )
    assert response.status_code == 403
    assert "App Check" in response.json().get("detail", "")


def test_enforce_mode_valid_token(client):
    """When APP_CHECK_ENFORCEMENT=enforce, a valid App Check token allows the request."""
    with patch("firebase_admin.auth.verify_id_token", return_value={"uid": "u1"}):
        with patch("main.APP_CHECK_ENFORCEMENT", "enforce"):
            with patch("firebase_admin.app_check.verify_token", return_value={"app_id": "1:test"}):
                with _mock_weather():
                    response = client.get(
                        "/weather/London/2024-05-15",
                        headers=_firebase_headers(app_check_token="valid-token"),
                    )
    assert response.status_code == 200
