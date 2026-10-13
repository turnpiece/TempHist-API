"""Tied years share a rank in the /meta ranking block (P1-176)."""

from routers.v1_records import _build_ranking_block


def _series(*temps, start_year=2020):
    """Build values one per year from start_year; None leaves a gap in the data."""
    return [{"year": start_year + i, "temperature": t} for i, t in enumerate(temps)]


def test_no_ties_ranks_are_complementary():
    r = _build_ranking_block(_series(14.5, 15.0, 15.5))

    assert r == {"warm": 1, "cold": 3, "total": 3, "ties": 0}
    assert r["warm"] + r["cold"] == r["total"] + 1


def test_tie_with_the_current_year_counts_tied_years_once():
    # Current year (2026) shares 8.6 with three earlier years; two years are warmer and one is colder.
    values = [
        {"year": 1989, "temperature": 8.6},
        {"year": 1990, "temperature": 9.4},
        {"year": 1998, "temperature": 8.6},
        {"year": 2005, "temperature": 7.0},
        {"year": 2010, "temperature": 10.1},
        {"year": 2025, "temperature": 8.6},
        {"year": 2026, "temperature": 8.6},
    ]

    r = _build_ranking_block(values)

    assert r == {"warm": 3, "cold": 2, "total": 7, "ties": 3}
    assert r["warm"] + r["cold"] == r["total"] + 1 - r["ties"]


def test_tie_elsewhere_in_the_series_does_not_change_the_current_years_rank():
    # 2021 and 2022 tie with each other (both colder than 2024); 2024 is unique.
    r = _build_ranking_block(_series(10.0, 9.0, 9.0, 12.0, 11.0, start_year=2020))

    assert r == {"warm": 2, "cold": 4, "total": 5, "ties": 0}
    assert r["warm"] + r["cold"] == r["total"] + 1


def test_current_year_tied_for_warmest_shares_rank_one():
    r = _build_ranking_block(_series(10.0, 12.0, 12.0))

    assert r == {"warm": 1, "cold": 2, "total": 3, "ties": 1}
    assert r["warm"] + r["cold"] == r["total"] + 1 - r["ties"]


def test_empty_series():
    assert _build_ranking_block([]) == {"warm": 1, "cold": 1, "total": 0, "ties": 0}


def test_missing_temperatures_are_skipped():
    r = _build_ranking_block(_series(10.0, None, 12.0, None))

    # The latest year with data (2022) is warmest of the two scored years.
    assert r == {"warm": 1, "cold": 2, "total": 2, "ties": 0}


def test_single_year():
    assert _build_ranking_block(_series(10.0)) == {"warm": 1, "cold": 1, "total": 1, "ties": 0}
