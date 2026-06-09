"""Tests for Phase 4 legacy slug fallback and lazy migration."""

import json
from datetime import date
from unittest.mock import MagicMock

from app.cache_utils import cache_get
from cache.keys import (
    bundle_key,
    get_bundle_with_slug_fallback,
    migrate_redis_key,
)
from utils.daily_temperature_store import LocationCacheIdentity, _build_lookup_slugs


class TestBuildLookupSlugs:
    def test_canonical_first_then_request_normalized(self):
        slugs = _build_lookup_slugs("london", "greater_london__england__united_kingdom", ())
        assert slugs == ("london", "greater_london__england__united_kingdom")

    def test_includes_pg_aliases_without_duplicates(self):
        slugs = _build_lookup_slugs(
            "london",
            "london",
            ["greater_london__england__united_kingdom", "lambeth__england__united_kingdom"],
        )
        assert slugs[0] == "london"
        assert "greater_london__england__united_kingdom" in slugs
        assert "lambeth__england__united_kingdom" in slugs
        assert len(slugs) == len(set(slugs))


class TestMigrateRedisKey:
    def test_renames_when_target_absent(self):
        r = MagicMock()
        r.exists.side_effect = lambda key: key == "old:key"
        r.rename.return_value = True

        assert migrate_redis_key(r, "old:key", "new:key") is True
        r.rename.assert_called_once_with("old:key", "new:key")

    def test_deletes_source_when_target_exists(self):
        r = MagicMock()
        r.exists.return_value = True

        assert migrate_redis_key(r, "old:key", "new:key") is False
        r.delete.assert_called_once_with("old:key")


class TestBundleSlugFallback:
    def test_hits_legacy_slug_and_migrates(self):
        r = MagicMock()
        canonical = "london"
        legacy = "greater_london__england__united_kingdom"
        identifier = "06-09"

        def get_side_effect(key):
            if key == bundle_key("daily", legacy, identifier):
                return b'{"records": []}'
            if key == f"{bundle_key('daily', legacy, identifier)}:etag":
                return b'"etag123"'
            return None

        r.get.side_effect = get_side_effect
        r.exists.side_effect = lambda key: key.startswith("records:v1:daily:greater_london")

        data, etag, hit_slug = get_bundle_with_slug_fallback(
            r, "daily", (canonical, legacy), identifier
        )

        assert hit_slug == legacy
        assert data is not None
        assert etag is not None
        r.rename.assert_called()


class TestTemporalLegacyFallback:
    def test_legacy_lexical_hit_with_canonical_name(self):
        """A legacy lexical temporal key is found when canonical_name is supplied."""
        r = MagicMock()

        legacy_wrapped = {
            "data": {"records": [{"year": 2024}]},
            "meta": {
                "requested": {"location": "London", "end_date": "2024-06-09"},
                "served_from": {
                    "canonical_location": "london",
                    "end_date": "2024-06-09",
                    "temporal_delta_days": 0,
                },
                "approximate": {"temporal": False},
            },
        }

        def get_side_effect(key):
            if key == "thv1:monthly:london:2024-06-09":
                return _legacy_compress(legacy_wrapped)
            return None

        r.get.side_effect = get_side_effect
        r.exists.return_value = False

        result = cache_get(
            r,
            agg="monthly",
            original_location="london",
            end_date=date(2024, 6, 9),
            canonical_name="London, England, United Kingdom",
        )

        assert result is not None
        assert result["data"]["records"][0]["year"] == 2024


def _legacy_compress(wrapped: dict) -> str:
    import base64
    import gzip

    raw = json.dumps(wrapped, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


class TestLocationCacheIdentityLookupSlugs:
    def test_default_lookup_slugs_tuple(self):
        identity = LocationCacheIdentity(
            redis_slug="london",
            canonical_name="London, England, United Kingdom",
            lookup_slugs=("london", "greater_london__england__united_kingdom"),
        )
        assert identity.lookup_slugs[0] == "london"
