"""Tests for working out a share's country (routers/_share_country.py) and the backfill script."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from routers._share_country import (
    _pick_geocoder_country,
    normalise_country_code,
    resolve_share_country,
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
        with (
            patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])) as geocode,
            patch("routers._share_country._geocode_mapbox", new=AsyncMock(return_value=[])) as mapbox,
        ):
            self.geocode = geocode
            self.mapbox = mapbox
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

    @pytest.mark.parametrize(
        "location,expected",
        [
            ("Moscow, Russia", "RU"),
            ("Istanbul, Turkey", "TR"),
            ("Abidjan, Ivory Coast", "CI"),
            ("Sarajevo, Bosnia & Herzegovina", "BA"),
            ("Kinshasa, Congo - Kinshasa", "CD"),
            ("Hong Kong, Hong Kong SAR China", "HK"),
            ("Oxford, England", "GB"),
            ("Edinburgh, Scotland", "GB"),
        ],
    )
    async def test_country_names_pycountry_does_not_know(self, location, expected):
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


class TestPopularFallback:
    """Bare names that aren't preapproved resolve from the popular (user-selected) locations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.metadata = {
            "oxford_england_united_kingdom": {"name": "Oxford", "country_code": "GB"},
            "worcester_england_united_kingdom": {"name": "Worcester", "country_code": "GB"},
            "worcester_massachusetts_united_states": {"name": "Worcester", "country_code": "US"},
            "malaga": {"name": "Málaga", "country_code": "ES"},
            "no_country": {"name": "Nowhereville"},
        }
        self.tracker = MagicMock()
        self.tracker.get_popular_from_selections.return_value = [(i, 1) for i in self.metadata]
        self.tracker.get_location_metadata.side_effect = self.metadata.get
        with (
            patch("routers._share_country.get_usage_tracker", return_value=self.tracker),
            patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])) as geocode,
            patch("routers._share_country._geocode_mapbox", new=AsyncMock(return_value=[])) as mapbox,
        ):
            self.geocode = geocode
            self.mapbox = mapbox
            yield

    async def test_bare_name_takes_the_popular_locations_country(self):
        assert await resolve_share_country_code("Oxford") == "GB"
        self.geocode.assert_not_called()
        self.mapbox.assert_not_called()

    async def test_match_ignores_case_and_accents(self):
        assert await resolve_share_country_code("malaga") == "ES"
        assert await resolve_share_country_code("MÁLAGA") == "ES"

    async def test_name_popular_in_two_countries_is_left_alone(self):
        assert await resolve_share_country_code("Worcester") is None

    async def test_name_with_a_region_is_not_looked_up(self):
        # "Oxford, Mississippi" is not the popular Oxford
        assert await resolve_share_country_code("Oxford, Mississippi") is None

    async def test_metadata_without_a_country_is_ignored(self):
        assert await resolve_share_country_code("Nowhereville") is None

    async def test_popular_beats_the_geocoder(self):
        self.geocode.return_value = [{"name": "Oxford", "country_code": "US", "latitude": 34.4, "longitude": -89.5}]
        assert await resolve_share_country_code("Oxford") == "GB"

    async def test_no_tracker_falls_through_to_the_geocoder(self):
        with patch("routers._share_country.get_usage_tracker", return_value=None):
            assert await resolve_share_country_code("Oxford") is None

    async def test_tracker_failure_falls_through(self):
        self.tracker.get_popular_from_selections.side_effect = RuntimeError("redis down")
        assert await resolve_share_country_code("Oxford") is None


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
        with (
            patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])),
            patch("routers._share_country._geocode_mapbox", new=AsyncMock(return_value=[])),
        ):
            summary = await backfill(store, r, execute=False)
        assert summary == {"missing": 2, "resolved": 1, "unresolved": 1, "updated": 0}
        store.set_country_code.assert_not_called()
        r.delete.assert_not_called()

    async def test_execute_updates_resolved_shares_and_clears_their_cache(self):
        store, r = self._store(self.SHARES), MagicMock()
        with (
            patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])),
            patch("routers._share_country._geocode_mapbox", new=AsyncMock(return_value=[])),
        ):
            summary = await backfill(store, r, execute=True)
        assert summary["updated"] == 1
        store.set_country_code.assert_awaited_once_with("aaaaaaaa", "NO")
        r.delete.assert_called_once_with("share:aaaaaaaa")


def _place(name, country, lat=None, lon=None):
    return {"name": name, "country_code": country, "latitude": lat, "longitude": lon}


class TestMapboxTiebreak:
    """When Open-Meteo's exact-name hits span several countries, Mapbox's top exact-name hit decides."""

    @pytest.fixture(autouse=True)
    def geocoders(self):
        with (
            patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])) as open_meteo,
            patch("routers._share_country._geocode_mapbox", new=AsyncMock(return_value=[])) as mapbox,
        ):
            self.open_meteo, self.mapbox = open_meteo, mapbox
            yield

    async def test_breaks_a_tie_between_countries(self):
        self.open_meteo.return_value = [_place("Kyoto", "JP"), _place("Kyoto", "TZ")]
        self.mapbox.return_value = [_place("Kyoto", "JP"), _place("Kyotomanyi", "UG")]
        assert await resolve_share_country("Kyoto") == ("JP", "Mapbox top match")

    async def test_finds_places_open_meteo_does_not_list(self):
        # Macau SAR isn't a populated place in GeoNames, so Open-Meteo only has the villages
        self.open_meteo.return_value = [_place("Macau", "FR"), _place("Macau", "BR")]
        self.mapbox.return_value = [_place("Macau", "MO"), _place("Macaúbas", "BR"), _place("Macau", "BR")]
        assert (await resolve_share_country("Macau")).code == "MO"

    async def test_open_meteo_still_goes_first_and_mapbox_is_not_called(self):
        self.open_meteo.return_value = [_place("Madrid", "ES")]
        assert await resolve_share_country("Madrid") == ("ES", "Open-Meteo geocoder")
        self.mapbox.assert_not_called()

    async def test_name_comparison_ignores_case_and_accents(self):
        self.mapbox.return_value = [_place("Ōta", "JP")]
        assert (await resolve_share_country("ota")).code == "JP"

    async def test_no_exact_name_match_is_left_unresolved_with_a_reason(self):
        self.open_meteo.return_value = [_place("Ota", c) for c in ("NG", "FR", "PT")]
        self.mapbox.return_value = [_place("Otaru", "JP"), _place("Ota-shi", "JP")]
        result = await resolve_share_country("Ota")
        assert result.code is None
        assert "ambiguous without coordinates (Open-Meteo matches in FR, NG, PT)" in result.detail
        assert "Mapbox has no exact-name match (top result: Otaru)" in result.detail

    async def test_coordinates_reject_a_mapbox_match_in_another_place(self):
        self.mapbox.return_value = [_place("Birmingham", "US", 33.52, -86.8)]
        result = await resolve_share_country("Birmingham", latitude=52.48, longitude=-1.9)
        assert result.code is None
        assert "Mapbox's match is more than 100 km from the share's coordinates" in result.detail

    async def test_coordinates_accept_a_nearby_mapbox_match(self):
        self.mapbox.return_value = [_place("Birmingham", "GB", 52.48, -1.9)]
        assert (await resolve_share_country("Birmingham", latitude=52.5, longitude=-1.9)).code == "GB"

    async def test_reason_when_neither_geocoder_knows_the_name(self):
        result = await resolve_share_country("Nowhereville")
        assert result == (None, "Open-Meteo has no exact-name match; Mapbox returned nothing")

    async def test_reason_when_coordinates_match_nothing_nearby(self):
        self.open_meteo.return_value = [_place("Springfield", "US", 39.8, -89.6)]
        result = await resolve_share_country("Springfield", latitude=51.5, longitude=-0.1)
        assert "no Open-Meteo match within 100 km of the share's coordinates" in result.detail

    async def test_open_meteo_failure_falls_through_to_mapbox(self):
        self.open_meteo.side_effect = RuntimeError("boom")
        self.mapbox.return_value = [_place("Kyoto", "JP")]
        assert (await resolve_share_country("Kyoto")).code == "JP"

    async def test_both_geocoders_failing_is_reported_not_raised(self):
        self.open_meteo.side_effect = RuntimeError("boom")
        self.mapbox.side_effect = RuntimeError("boom")
        assert await resolve_share_country("Kyoto") == (None, "Open-Meteo lookup failed; Mapbox lookup failed")


class TestResolutionDetail:
    @pytest.fixture(autouse=True)
    def no_geocoder(self):
        with (
            patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])),
            patch("routers._share_country._geocode_mapbox", new=AsyncMock(return_value=[])),
        ):
            yield

    @pytest.mark.parametrize(
        "location,client_code,expected",
        [
            ("Kirkenes", "NO", ("NO", "supplied by the client")),
            ("Dublin", None, ("IE", "preapproved location")),
            ("Kirkenes, Finnmark, Norway", None, ("NO", "country named in the location")),
            (" , ", None, (None, "no place name in the location")),
        ],
    )
    async def test_detail_says_where_the_code_came_from(self, location, client_code, expected):
        assert await resolve_share_country(location, client_code) == expected


class TestBackfillReport:
    async def test_report_says_where_each_country_came_from_or_why_it_is_missing(self, capsys):
        store = MagicMock()
        store.list_shares_missing_country = AsyncMock(
            return_value=[
                {"id": "aaaaaaaa", "location": "Kirkenes, Finnmark, Norway", "latitude": None, "longitude": None},
                {"id": "bbbbbbbb", "location": "Nowhereville", "latitude": None, "longitude": None},
            ]
        )
        with (
            patch("routers._share_country._geocode_open_meteo", new=AsyncMock(return_value=[])),
            patch("routers._share_country._geocode_mapbox", new=AsyncMock(return_value=[])),
        ):
            await backfill(store, None, execute=False)
        out = capsys.readouterr().out
        assert "NO aaaaaaaa  'Kirkenes, Finnmark, Norway'  (country named in the location)" in out
        assert "?  bbbbbbbb  'Nowhereville'  (left NULL: Open-Meteo has no exact-name match; Mapbox returned nothing)" in out
