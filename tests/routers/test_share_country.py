"""Tests for working out a share's country (routers/_share_country.py) and the backfill script."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routers._share_country import (
    _pick_geocoder_country,
    normalise_country_code,
    resolve_share_country_code,
)
from scripts.backfill_share_country import backfill


PREAPPROVED = [
    SimpleNamespace(id="london", name="London", country_name="United Kingdom", country_code="GB"),
    SimpleNamespace(id="dublin", name="Dublin", country_name="Ireland", country_code="IE"),
    SimpleNamespace(id="new_york_city", name="New York", country_name="United States", country_code="US"),
]


@pytest.fixture(autouse=True)
def preapproved_locations():
    with patch("routers.locations.locations_data", PREAPPROVED):
        yield


class TestNormaliseCountryCode:
    @pytest.mark.parametrize("raw,expected", [("NO", "NO"), ("no", "NO"), (" gb ", "GB"), ("UK", "GB")])
    def test_valid(self, raw, expected):
        assert normalise_country_code(raw) == expected

    @pytest.mark.parametrize("raw", ["EU", "XX", "NOR", "N", ""])
    def test_invalid(self, raw):
        with pytest.raises(ValueError):
            normalise_country_code(raw)


class TestResolveShareCountryCode:
    @pytest.fixture(autouse=True)
    def no_geocoder(self):
        """Resolution must not need the network unless a test opts in."""
        with patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])) as geocode:
            self.geocode = geocode
            yield

    async def test_client_code_wins(self):
        assert await resolve_share_country_code("Kirkenes, Finnmark, Norway", "SE") == "SE"
        self.geocode.assert_not_called()

    async def test_preapproved_id(self):
        assert await resolve_share_country_code("london") == "GB"

    async def test_preapproved_bare_name(self):
        assert await resolve_share_country_code("Dublin") == "IE"

    async def test_preapproved_bare_name_whose_id_differs(self):
        assert await resolve_share_country_code("New York") == "US"

    async def test_empty_location_resolves_to_nothing(self):
        assert await resolve_share_country_code(" , ") is None
        self.geocode.assert_not_called()

    async def test_preapproved_name_with_matching_country(self):
        assert await resolve_share_country_code("London, England, United Kingdom") == "GB"

    async def test_same_name_in_another_country_is_not_the_preapproved_one(self):
        assert await resolve_share_country_code("London, Ontario, Canada") == "CA"

    async def test_country_at_end_of_text(self):
        assert await resolve_share_country_code("Kirkenes, Finnmark, Norway") == "NO"
        assert await resolve_share_country_code("Greater London, England, United Kingdom") == "GB"
        assert await resolve_share_country_code("Taito, Japan") == "JP"
        self.geocode.assert_not_called()

    @pytest.mark.parametrize(
        "location,expected",
        [("London, UK", "GB"), ("London, uk", "GB"), ("Springfield, USA", "US"), ("Dubai, UAE", "AE")],
    )
    async def test_common_country_abbreviations_at_the_end(self, location, expected):
        assert await resolve_share_country_code(location) == expected
        self.geocode.assert_not_called()

    async def test_region_or_state_code_at_the_end_is_not_a_country(self):
        # "IL" is Israel's code, but here it's a US state; a region name is not a country either
        assert await resolve_share_country_code("Springfield, IL") is None
        assert await resolve_share_country_code("Kirkenes, Finnmark") is None

    async def test_falls_back_to_geocoder_for_bare_names(self):
        self.geocode.return_value = [
            {"name": "Madrid", "country_code": "ES", "latitude": 40.4, "longitude": -3.7},
            {"name": "Madrid", "country_code": "ES", "latitude": 40.9, "longitude": -3.5},
        ]
        assert await resolve_share_country_code("Madrid") == "ES"
        self.geocode.assert_awaited_once()

    async def test_geocoder_failure_leaves_country_unknown(self):
        self.geocode.side_effect = RuntimeError("boom")
        assert await resolve_share_country_code("Madrid") is None

    async def test_unresolvable(self):
        assert await resolve_share_country_code("Nowhereville") is None


class TestPickGeocoderCountry:
    BIRMINGHAMS = [
        {"name": "Birmingham", "country_code": "GB", "latitude": 52.48, "longitude": -1.9},
        {"name": "Birmingham", "country_code": "US", "latitude": 33.52, "longitude": -86.8},
    ]

    def test_ambiguous_name_without_coordinates_is_left_alone(self):
        assert _pick_geocoder_country("Birmingham", self.BIRMINGHAMS, None, None) is None

    def test_coordinates_pick_the_nearby_one(self):
        assert _pick_geocoder_country("Birmingham", self.BIRMINGHAMS, 33.5, -86.8) == "US"

    def test_coordinates_far_from_every_hit_give_nothing(self):
        assert _pick_geocoder_country("Birmingham", self.BIRMINGHAMS, 0.0, 0.0) is None

    def test_only_exact_name_hits_count(self):
        places = [{"name": "Madrid Norte", "country_code": "ES", "latitude": 40.0, "longitude": -3.0}]
        assert _pick_geocoder_country("Madrid", places, None, None) is None


class TestBackfill:
    @staticmethod
    def _store(shares):
        store = MagicMock()
        store.list_shares_missing_country = AsyncMock(return_value=shares)
        store.set_country_code = AsyncMock(return_value=True)
        return store

    SHARES = [
        {"id": "aaaaaaaa", "location": "Kirkenes, Finnmark, Norway", "latitude": None, "longitude": None},
        {"id": "bbbbbbbb", "location": "Nowhereville", "latitude": None, "longitude": None},
    ]

    async def test_dry_run_changes_nothing(self):
        store, r = self._store(self.SHARES), MagicMock()
        with patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])):
            summary = await backfill(store, r, execute=False)
        assert summary == {"missing": 2, "resolved": 1, "unresolved": 1, "updated": 0}
        store.set_country_code.assert_not_called()
        r.delete.assert_not_called()

    async def test_execute_updates_resolved_shares_and_clears_their_cache(self):
        store, r = self._store(self.SHARES), MagicMock()
        with patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])):
            summary = await backfill(store, r, execute=True)
        assert summary["updated"] == 1
        store.set_country_code.assert_awaited_once_with("aaaaaaaa", "NO")
        r.delete.assert_called_once_with("share:aaaaaaaa")
