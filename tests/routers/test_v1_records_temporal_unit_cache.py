from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from main import API_ACCESS_TOKEN
from main import app as main_app
from routers import v1_records
from routers.v1_records import get_invalid_location_cache, get_redis_client
from utils.daily_temperature_store import LocationCacheIdentity


class InMemoryTemporalRedis:
    def __init__(self):
        self.values = {}
        self.zsets = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None):
        self.values[key] = value
        return True

    def setex(self, key, _ttl, value):
        self.values[key] = value
        return True

    def zadd(self, key, mapping):
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    def expire(self, _key, _ttl):
        return True

    def pipeline(self):
        return self

    def execute(self):
        return [True]


@pytest.fixture
def fake_redis():
    return InMemoryTemporalRedis()


@pytest.fixture
def client(fake_redis):
    invalid_location_cache = MagicMock()
    invalid_location_cache.is_invalid_location.return_value = False

    main_app.dependency_overrides[get_redis_client] = lambda: fake_redis
    main_app.dependency_overrides[get_invalid_location_cache] = lambda: invalid_location_cache
    try:
        with TestClient(main_app) as test_client:
            yield test_client
    finally:
        main_app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("first_unit", "first_temp", "second_unit", "second_temp"),
    [
        ("fahrenheit", 68.0, "celsius", 20.0),
        ("celsius", 20.0, "fahrenheit", 68.0),
    ],
)
def test_temporal_cache_rebuilds_response_for_requested_unit(
    monkeypatch,
    client,
    first_unit,
    first_temp,
    second_unit,
    second_temp,
):
    year_data = {
        2023: {"date": "2023-06-13", "year": 2023, "temperature": 20.0},
        2024: {"date": "2024-06-13", "year": 2024, "temperature": 20.0},
    }

    monkeypatch.setattr(v1_records, "CACHE_ENABLED", True)
    monkeypatch.setattr(v1_records, "is_location_likely_invalid", lambda _location: False)
    monkeypatch.setattr(v1_records, "get_bundle_with_slug_fallback", lambda *args, **kwargs: (None, None, None))
    monkeypatch.setattr(v1_records, "get_records", AsyncMock(return_value=(year_data, [], False)))
    monkeypatch.setattr(v1_records, "get_year_etags", AsyncMock(return_value={2023: '"2023"', 2024: '"2024"'}))
    monkeypatch.setattr(
        v1_records,
        "assemble_and_cache",
        AsyncMock(return_value=({"records": list(year_data.values())}, '"bundle-etag"')),
    )
    monkeypatch.setattr(v1_records, "_get_location_timezone", lambda *_args, **_kwargs: "Europe/London")
    monkeypatch.setattr(
        v1_records,
        "resolve_location_cache_identity",
        AsyncMock(
            return_value=LocationCacheIdentity(
                redis_slug="test_city",
                canonical_name="Test City",
                lookup_slugs=("test_city",),
            )
        ),
    )

    headers = {"Authorization": f"Bearer {API_ACCESS_TOKEN}"}
    url = "/v1/records/daily/Test%20City/06-13"

    first_response = client.get(f"{url}?unit_group={first_unit}", headers=headers)
    assert first_response.status_code == 200
    first_data = first_response.json()
    assert first_data["unit_group"] == first_unit
    assert first_data["average"]["unit"] == first_unit
    assert first_data["values"][0]["temperature"] == first_temp

    second_response = client.get(f"{url}?unit_group={second_unit}", headers=headers)
    assert second_response.status_code == 200
    assert second_response.headers["X-Cache-Status"] == "HIT"

    second_data = second_response.json()
    assert second_data["unit_group"] == second_unit
    assert second_data["average"]["unit"] == second_unit
    assert second_data["values"][0]["temperature"] == second_temp
