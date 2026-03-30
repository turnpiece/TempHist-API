"""Tests for app/cache_utils.py — canonicalization, temporal tolerance, and metadata."""

import gzip
import json
import pytest
from datetime import date
from unittest.mock import MagicMock, call

from app.cache_utils import (
    canonicalize_location,
    cache_get,
    cache_set,
    cache_invalidate,
    _val_key,
    _tindex_key,
    _epoch,
    TEMPORAL_TOLERANCE,
    KEY_NS,
)


# ---------------------------------------------------------------------------
# canonicalize_location
# ---------------------------------------------------------------------------

class TestCanonicalizeLocation:

    def test_basic_normalization(self):
        assert canonicalize_location("London") == "london"

    def test_commas_and_spaces(self):
        assert canonicalize_location("London, England, United Kingdom") == "london_england_united_kingdom"

    def test_resolved_address_takes_precedence(self):
        result = canonicalize_location("London", "London, England, United Kingdom")
        assert result == "london_england_united_kingdom"

    def test_extra_whitespace(self):
        assert canonicalize_location("  New   York  ,  USA  ") == "new_york_usa"

    def test_empty_string(self):
        assert canonicalize_location("") == ""

    def test_none_original_with_resolved(self):
        assert canonicalize_location("", "Paris, France") == "paris_france"

    def test_consistency(self):
        """Different user inputs for the same location produce the same canonical key
        when resolved_address is available."""
        ra = "London, England, United Kingdom"
        assert canonicalize_location("London", ra) == canonicalize_location("London, UK", ra)
        assert canonicalize_location("london", ra) == canonicalize_location("LONDON", ra)

    def test_case_insensitive(self):
        assert canonicalize_location("NEW YORK") == canonicalize_location("new york")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis_mock():
    """Create a MagicMock that behaves like redis.Redis."""
    r = MagicMock()
    pipe = MagicMock()
    pipe.execute = MagicMock(return_value=[True, True, True])
    r.pipeline.return_value = pipe
    return r, pipe


def _compress(obj: dict) -> bytes:
    """Gzip-compress a dict (same as cache_set internals)."""
    return gzip.compress(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


# ---------------------------------------------------------------------------
# cache_set
# ---------------------------------------------------------------------------

class TestCacheSet:

    def test_stores_compressed_payload_with_metadata(self):
        r, pipe = _make_redis_mock()
        payload = {"records": [{"year": 2024, "temp": 15.0}]}

        result = cache_set(
            r,
            agg="daily",
            original_location="London, UK",
            end_date=date(2024, 3, 25),
            payload=payload,
        )

        assert result is True
        pipe.set.assert_called_once()
        # Verify the stored value is gzip-compressed JSON containing the payload + meta
        stored_gz = pipe.set.call_args[0][1]
        stored = json.loads(gzip.decompress(stored_gz))
        assert stored["data"] == payload
        assert stored["meta"]["requested"]["location"] == "London, UK"
        assert stored["meta"]["served_from"]["canonical_location"] == "london_uk"
        assert stored["meta"]["approximate"]["temporal"] is False
        assert stored["meta"]["served_from"]["temporal_delta_days"] == 0

    def test_uses_pipeline_for_atomicity(self):
        r, pipe = _make_redis_mock()

        cache_set(
            r,
            agg="monthly",
            original_location="Paris",
            end_date=date(2024, 6, 15),
            payload={"data": 1},
        )

        r.pipeline.assert_called_once()
        pipe.execute.assert_called_once()
        # Should call set, zadd, expire
        assert pipe.set.call_count == 1
        assert pipe.zadd.call_count == 1
        assert pipe.expire.call_count == 1

    def test_custom_ttl(self):
        r, pipe = _make_redis_mock()

        cache_set(
            r,
            agg="daily",
            original_location="Berlin",
            end_date=date(2024, 1, 1),
            payload={},
            ttl_seconds=3600,
        )

        # set() should be called with ex=3600
        set_call = pipe.set.call_args
        assert set_call[1].get("ex") == 3600 or set_call[0][2:] == () and set_call[1]["ex"] == 3600

    def test_returns_false_on_error(self):
        r, pipe = _make_redis_mock()
        pipe.execute.side_effect = Exception("connection lost")

        result = cache_set(
            r,
            agg="daily",
            original_location="Tokyo",
            end_date=date(2024, 7, 1),
            payload={},
        )

        assert result is False

    def test_resolved_address_used_for_canonical(self):
        r, pipe = _make_redis_mock()

        cache_set(
            r,
            agg="yearly",
            original_location="London",
            end_date=date(2024, 3, 25),
            payload={},
            resolved_address="London, England, United Kingdom",
        )

        stored_gz = pipe.set.call_args[0][1]
        stored = json.loads(gzip.decompress(stored_gz))
        assert stored["meta"]["served_from"]["canonical_location"] == "london_england_united_kingdom"


# ---------------------------------------------------------------------------
# cache_get — exact match
# ---------------------------------------------------------------------------

class TestCacheGetExact:

    def test_exact_hit(self):
        r, _ = _make_redis_mock()
        wrapped = {
            "data": {"records": [1, 2, 3]},
            "meta": {
                "requested": {"location": "London", "end_date": "2024-03-25"},
                "served_from": {
                    "canonical_location": "london",
                    "end_date": "2024-03-25",
                    "temporal_delta_days": 0,
                },
                "approximate": {"temporal": False},
            },
        }
        r.get.return_value = _compress(wrapped)

        result = cache_get(
            r, agg="daily", original_location="London", end_date=date(2024, 3, 25)
        )

        assert result is not None
        assert result["data"] == {"records": [1, 2, 3]}
        assert result["meta"]["approximate"]["temporal"] is False

    def test_exact_miss_for_daily(self):
        """Daily has zero tolerance — should not try temporal index."""
        r, _ = _make_redis_mock()
        r.get.return_value = None

        result = cache_get(
            r, agg="daily", original_location="London", end_date=date(2024, 3, 25)
        )

        assert result is None
        # Should not call zrangebyscore for daily
        r.zrangebyscore.assert_not_called()


# ---------------------------------------------------------------------------
# cache_get — temporal tolerance
# ---------------------------------------------------------------------------

class TestCacheGetTemporal:

    def _setup_temporal_hit(self, r, agg, stored_date_iso, requested_location):
        """Set up mock for a temporal tolerance hit."""
        canonical = canonicalize_location(requested_location)

        # Exact match returns None
        r.get.side_effect = lambda key: {
            _val_key(agg, canonical, stored_date_iso): _compress({
                "data": {"records": ["stored_data"]},
                "meta": {
                    "requested": {"location": requested_location, "end_date": stored_date_iso},
                    "served_from": {
                        "canonical_location": canonical,
                        "end_date": stored_date_iso,
                        "temporal_delta_days": 0,
                    },
                    "approximate": {"temporal": False},
                },
            }),
        }.get(key)

        # Sorted set returns the stored date
        r.zrangebyscore.return_value = [stored_date_iso.encode()]

    def test_yearly_tolerance_within_7_days(self):
        r, _ = _make_redis_mock()
        self._setup_temporal_hit(r, "yearly", "2024-03-20", "London")

        result = cache_get(
            r, agg="yearly", original_location="London", end_date=date(2024, 3, 25)
        )

        assert result is not None
        assert result["meta"]["approximate"]["temporal"] is True
        assert result["meta"]["served_from"]["end_date"] == "2024-03-20"
        assert result["meta"]["served_from"]["temporal_delta_days"] == 5
        assert result["meta"]["requested"]["end_date"] == "2024-03-25"

    def test_monthly_tolerance_within_2_days(self):
        r, _ = _make_redis_mock()
        self._setup_temporal_hit(r, "monthly", "2024-06-14", "Paris")

        result = cache_get(
            r, agg="monthly", original_location="Paris", end_date=date(2024, 6, 15)
        )

        assert result is not None
        assert result["meta"]["approximate"]["temporal"] is True
        assert result["meta"]["served_from"]["temporal_delta_days"] == 1

    def test_no_candidates_returns_none(self):
        r, _ = _make_redis_mock()
        r.get.return_value = None
        r.zrangebyscore.return_value = []

        result = cache_get(
            r, agg="yearly", original_location="London", end_date=date(2024, 3, 25)
        )

        assert result is None

    def test_candidate_exists_but_data_expired(self):
        """Sorted set has the date but the actual data key was evicted."""
        r, _ = _make_redis_mock()
        r.get.return_value = None  # Both exact and candidate data miss
        r.zrangebyscore.return_value = [b"2024-03-20"]

        result = cache_get(
            r, agg="yearly", original_location="London", end_date=date(2024, 3, 25)
        )

        assert result is None

    def test_picks_nearest_candidate(self):
        """When multiple candidates exist, picks the one closest to the requested date."""
        r, _ = _make_redis_mock()
        canonical = canonicalize_location("London")

        stored_data = {
            "data": {"records": ["nearest"]},
            "meta": {
                "requested": {"location": "London", "end_date": "2024-03-23"},
                "served_from": {
                    "canonical_location": canonical,
                    "end_date": "2024-03-23",
                    "temporal_delta_days": 0,
                },
                "approximate": {"temporal": False},
            },
        }

        def get_side_effect(key):
            if key == _val_key("yearly", canonical, "2024-03-23"):
                return _compress(stored_data)
            return None

        r.get.side_effect = get_side_effect
        r.zrangebyscore.return_value = [b"2024-03-19", b"2024-03-23"]

        result = cache_get(
            r, agg="yearly", original_location="London", end_date=date(2024, 3, 25)
        )

        assert result is not None
        assert result["meta"]["served_from"]["end_date"] == "2024-03-23"
        assert result["meta"]["served_from"]["temporal_delta_days"] == 2


# ---------------------------------------------------------------------------
# cache_invalidate
# ---------------------------------------------------------------------------

class TestCacheInvalidate:

    def test_invalidate_specific_date(self):
        r, pipe = _make_redis_mock()

        result = cache_invalidate(
            r, agg="daily", original_location="London", end_date=date(2024, 3, 25)
        )

        assert result is True
        pipe.delete.assert_called_once()
        pipe.zrem.assert_called_once()
        pipe.execute.assert_called_once()

    def test_invalidate_all_dates(self):
        r, pipe = _make_redis_mock()
        r.zrange.return_value = [b"2024-03-20", b"2024-03-25"]

        result = cache_invalidate(
            r, agg="yearly", original_location="London"
        )

        assert result is True
        # Should delete both data keys + the sorted set
        assert pipe.delete.call_count == 3


# ---------------------------------------------------------------------------
# Key format
# ---------------------------------------------------------------------------

class TestKeyFormat:

    def test_val_key_format(self):
        assert _val_key("daily", "london", "2024-03-25") == f"{KEY_NS}:daily:london:2024-03-25"

    def test_tindex_key_format(self):
        assert _tindex_key("yearly", "london") == f"{KEY_NS}:idx:yearly:london"

    def test_epoch_consistency(self):
        d = date(2024, 3, 25)
        assert _epoch(d) == _epoch(d)  # deterministic
        assert _epoch(date(2024, 3, 26)) > _epoch(date(2024, 3, 25))


# ---------------------------------------------------------------------------
# Temporal tolerance config
# ---------------------------------------------------------------------------

class TestTemporalToleranceConfig:

    def test_daily_is_exact(self):
        assert TEMPORAL_TOLERANCE["daily"] == 0

    def test_weekly_is_exact(self):
        assert TEMPORAL_TOLERANCE["weekly"] == 0

    def test_monthly_tolerance(self):
        assert TEMPORAL_TOLERANCE["monthly"] == 2

    def test_yearly_tolerance(self):
        assert TEMPORAL_TOLERANCE["yearly"] == 7
