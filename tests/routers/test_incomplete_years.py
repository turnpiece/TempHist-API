"""Incomplete year sets are rejected, never cached as a bundle or drawn as the full series (P1-190)."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.responses import Response

from routers import og_image, v1_records
from utils.weather import get_year_range, missing_past_years

CURRENT_YEAR = datetime.now(timezone.utc).year
SHARE = {
    "id": "FlDfMx6o",
    "location": "Taito, Japan",
    "period": "yearly",
    "identifier": "10-24",
    "ref_year": CURRENT_YEAR,
    "unit": "celsius",
}


def _records(years):
    return [{"year": y, "temperature": 15.0} for y in years]


FULL = _records(get_year_range(CURRENT_YEAR))
# What the broken share on develop looked like: every fourth year, nothing recent.
SPARSE = _records(range(CURRENT_YEAR - 3, CURRENT_YEAR - 48, -4))


class TestMissingPastYears:
    def test_full_series_is_complete(self):
        assert missing_past_years((r["year"] for r in FULL), CURRENT_YEAR) == []

    def test_current_year_is_not_required(self):
        years = [y for y in get_year_range(CURRENT_YEAR) if y != CURRENT_YEAR]
        assert missing_past_years(years, CURRENT_YEAR) == []

    def test_sparse_series_reports_the_gaps(self):
        missing = missing_past_years((r["year"] for r in SPARSE), CURRENT_YEAR)
        assert CURRENT_YEAR - 1 in missing
        assert len(missing) == len(get_year_range(CURRENT_YEAR)) - 1 - len(SPARSE)

    def test_empty_set_is_missing_every_past_year(self):
        assert len(missing_past_years([], CURRENT_YEAR)) == 50


class TestBundleCacheRejectsPartialBundles:
    def test_partial_bundle_is_flagged(self):
        assert v1_records._is_incomplete_bundle({"records": SPARSE}, CURRENT_YEAR, "bundle:k")

    def test_full_and_empty_bundles_are_not_flagged(self):
        assert not v1_records._is_incomplete_bundle({"records": FULL}, CURRENT_YEAR, "bundle:k")
        assert not v1_records._is_incomplete_bundle({"records": []}, CURRENT_YEAR, "bundle:k")

    def test_partial_bundle_is_not_served_even_on_etag_match(self):
        response = Response()
        with patch.object(
            v1_records,
            "get_bundle_with_slug_fallback",
            return_value=(json.dumps({"records": SPARSE}).encode(), b'"etag"', "taito"),
        ):
            result = v1_records._try_serve_bundle_cache(
                redis_client=MagicMock(), period="yearly", location="Taito, Japan", identifier="10-24",
                slug="taito", lookup_slugs=("taito",), month=10, day=24, current_year=CURRENT_YEAR,
                years=get_year_range(CURRENT_YEAR), unit_group="celsius", if_none_match='"etag"',
                invalid_location_cache=MagicMock(), end_date=None, response=response,
            )
        assert result is None
        assert response.status_code != 304

    async def test_assemble_does_not_cache_a_partial_bundle(self):
        year_data = {r["year"]: r for r in SPARSE}
        with (
            patch.object(v1_records, "get_year_etags", AsyncMock(return_value={})),
            patch.object(v1_records, "assemble_and_cache", AsyncMock()) as assemble,
            patch.object(v1_records, "_rebuild_full_response_from_values", return_value={}),
        ):
            _, status, _, etag = await v1_records._assemble_cached_record(
                redis_client=MagicMock(), period="yearly", location="Taito, Japan", identifier="10-24",
                slug="taito", lookup_slugs=("taito",), year_data=year_data, missing_past=[], missing_current=False,
                month=10, day=24, current_year=CURRENT_YEAR, years=get_year_range(CURRENT_YEAR),
                unit_group="celsius", if_none_match=None, response=Response(),
            )
        assemble.assert_not_awaited()
        assert etag  # still computed, so the response carries a valid ETag

    async def test_assemble_caches_a_complete_bundle(self):
        year_data = {r["year"]: r for r in FULL}
        with (
            patch.object(v1_records, "get_year_etags", AsyncMock(return_value={})),
            patch.object(v1_records, "assemble_and_cache", AsyncMock(return_value=({}, "etag"))) as assemble,
            patch.object(v1_records, "_rebuild_full_response_from_values", return_value={}),
        ):
            await v1_records._assemble_cached_record(
                redis_client=MagicMock(), period="yearly", location="Taito, Japan", identifier="10-24",
                slug="taito", lookup_slugs=("taito",), year_data=year_data, missing_past=[], missing_current=False,
                month=10, day=24, current_year=CURRENT_YEAR, years=get_year_range(CURRENT_YEAR),
                unit_group="celsius", if_none_match=None, response=Response(),
            )
        assemble.assert_awaited_once()


class TestDroppedYearLogging:
    def test_dropped_historical_years_are_logged_with_reasons(self, caplog):
        missing = [
            {"year": 2010, "reason": "coverage_below_threshold"},
            {"year": 2014, "reason": "coverage_below_threshold"},
            {"year": CURRENT_YEAR, "reason": "no_data_current_year"},
        ]
        with caplog.at_level("WARNING", logger=v1_records.logger.name):
            v1_records._log_dropped_historical_years("Taito, Japan", "yearly", 10, 24, CURRENT_YEAR, missing)
        assert "coverage_below_threshold=[2010, 2014]" in caplog.text
        assert "no_data_current_year" not in caplog.text

    def test_nothing_logged_when_only_the_current_year_is_missing(self, caplog):
        missing = [{"year": CURRENT_YEAR, "reason": "no_data_current_year"}]
        with caplog.at_level("WARNING", logger=v1_records.logger.name):
            v1_records._log_dropped_historical_years("Taito, Japan", "yearly", 10, 24, CURRENT_YEAR, missing)
        assert caplog.text == ""


class TestOgImageRejectsPartialSets:
    @pytest.fixture
    def stubs(self):
        with (
            patch.object(og_image, "_lookup_share", AsyncMock(return_value=SHARE)),
            patch.object(og_image, "_render_chart", return_value=b"CHART") as chart,
            patch.object(og_image, "_render_placeholder", return_value=b"PLACEHOLDER") as placeholder,
        ):
            yield chart, placeholder

    async def _call(self):
        return await og_image.og_image(share_id="FlDfMx6o", redis_client=MagicMock(), show_title=False)

    async def test_complete_bundle_renders_a_chart_cached_for_a_day(self, stubs):
        chart, placeholder = stubs
        with patch.object(og_image, "_get_bundle_records", AsyncMock(return_value=FULL)):
            response = await self._call()
        assert response.body == b"CHART"
        assert response.headers["cache-control"] == "public, max-age=86400"
        placeholder.assert_not_called()

    async def test_partial_cache_falls_through_to_live_fetch(self, stubs):
        chart, _ = stubs
        live = AsyncMock(return_value=FULL)
        with (
            patch.object(og_image, "_get_bundle_records", AsyncMock(return_value=None)),
            patch.object(og_image, "_fetch_records_live", live),
        ):
            response = await self._call()
        live.assert_awaited_once()
        assert response.body == b"CHART"
        assert chart.call_args.args[1] == FULL

    async def test_partial_live_result_gets_a_short_lived_placeholder(self, stubs):
        chart, placeholder = stubs
        with (
            patch.object(og_image, "_get_bundle_records", AsyncMock(return_value=None)),
            patch.object(og_image, "_fetch_records_live", AsyncMock(return_value=SPARSE)),
        ):
            response = await self._call()
        assert response.body == b"PLACEHOLDER"
        assert response.headers["cache-control"] == "public, max-age=300"
        chart.assert_not_called()

    async def test_no_data_gets_a_short_lived_placeholder(self, stubs):
        with (
            patch.object(og_image, "_get_bundle_records", AsyncMock(return_value=None)),
            patch.object(og_image, "_fetch_records_live", AsyncMock(return_value=None)),
        ):
            response = await self._call()
        assert response.headers["cache-control"] == "public, max-age=300"

    async def test_render_failure_gets_a_short_lived_placeholder(self, stubs):
        chart, _ = stubs
        chart.side_effect = RuntimeError("boom")
        with patch.object(og_image, "_get_bundle_records", AsyncMock(return_value=FULL)):
            response = await self._call()
        assert response.body == b"PLACEHOLDER"
        assert response.headers["cache-control"] == "public, max-age=300"


class TestOgBundleRecords:
    """_get_bundle_records must not hand back a partial set from either the bundle or the per-year keys."""

    @staticmethod
    def _redis_with_years(years):
        redis_client = MagicMock()
        by_year = {og_image.rec_key("yearly", "taito", "10-24", r["year"]): json.dumps(r) for r in _records(years)}

        def mget(keys):
            return [by_year.get(k) for k in keys]

        redis_client.mget.side_effect = mget
        return redis_client

    async def _get(self, redis_client, bundle):
        identity = MagicMock(lookup_slugs=("taito",), redis_slug="taito")
        with (
            patch.object(og_image, "resolve_location_cache_identity", AsyncMock(return_value=identity)),
            patch.object(og_image, "get_bundle_with_slug_fallback", return_value=(bundle, None, None)),
        ):
            return await og_image._get_bundle_records(SHARE, redis_client)

    async def test_partial_bundle_and_partial_per_year_keys_are_a_miss(self):
        bundle = json.dumps({"records": SPARSE}).encode()
        redis_client = self._redis_with_years([r["year"] for r in SPARSE])
        assert await self._get(redis_client, bundle) is None

    async def test_complete_per_year_keys_are_used_when_the_bundle_is_partial(self):
        bundle = json.dumps({"records": SPARSE}).encode()
        redis_client = self._redis_with_years([r["year"] for r in FULL])
        records = await self._get(redis_client, bundle)
        assert records is not None
        assert {r["year"] for r in records} == {r["year"] for r in FULL}
