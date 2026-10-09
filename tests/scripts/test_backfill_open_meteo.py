"""Unit tests for scripts/backfill_open_meteo.py (P1-173)."""

import json
from datetime import date, timedelta

import pytest

from scripts import backfill_open_meteo as backfill


def _daily(start: date, end: date, value_for) -> dict:
    out, day = {}, start
    while day <= end:
        out[day] = value_for(day)
        day += timedelta(days=1)
    return out


class TestArchiveEndDate:
    def test_stops_one_day_before_forecast_boundary(self):
        # fetch_days sends the last 7 days to the forecast endpoint, which cannot serve the pinned model.
        assert backfill.archive_end_date(date(2026, 10, 7)) == date(2026, 9, 29)


class TestEstimateCalls:
    def test_matches_pricing_page_examples(self):
        assert backfill.estimate_open_meteo_calls(14) == 1
        assert backfill.estimate_open_meteo_calls(28) == 3

    def test_full_history_is_thousands_of_calls(self):
        assert 2500 < backfill.estimate_open_meteo_calls(18628) < 2800

    def test_never_below_one(self):
        assert backfill.estimate_open_meteo_calls(1) == 1


class TestRollingYearMeans:
    def test_recovers_a_known_trend(self):
        anchor = date(2026, 10, 6)
        daily = _daily(date(1975, 1, 1), anchor, lambda d: 10 + 0.03 * (d.year - 1975))
        slope, r2, _err = backfill.calculate_trend_slope(backfill.rolling_year_means(daily, anchor))
        assert slope == pytest.approx(0.3, abs=0.02)
        assert r2 > 0.95

    def test_year_with_too_few_days_is_dropped(self):
        anchor = date(2026, 10, 6)
        daily = _daily(date(2000, 1, 1), anchor, lambda d: 10.0)
        for offset in range(100):  # leave only 265 days in the 2010 window
            daily.pop(date(2010, 10, 6) - timedelta(days=offset))
        years = [p["x"] for p in backfill.rolling_year_means(daily, anchor)]
        assert 2010 not in years
        assert 2009 in years

    def test_leap_day_anchor_does_not_raise(self):
        anchor = date(2024, 2, 29)
        daily = _daily(date(1990, 1, 1), anchor, lambda d: 5.0)
        assert backfill.rolling_year_means(daily, anchor)


class TestBuildRows:
    def test_skips_days_without_a_mean(self):
        days = [
            {"datetime": "2020-01-01", "temp": 3.5, "tempmax": 5.0, "tempmin": 1.0},
            {"datetime": "2020-01-02", "temp": None, "tempmax": 5.0, "tempmin": 1.0},
        ]
        rows = backfill.build_rows(7, days, "open-meteo:era5_land")
        assert len(rows) == 1
        location_id, day, temp, tmax, tmin, payload, source = rows[0]
        assert (location_id, day, temp, tmax, tmin) == (7, date(2020, 1, 1), 3.5, 5.0, 1.0)
        assert source == "open-meteo:era5_land"

    def test_payload_matches_the_shape_the_store_writes(self):
        rows = backfill.build_rows(1, [{"datetime": "2020-01-01", "temp": 3.5, "tempmax": 5.0, "tempmin": 1.0}], "s")
        assert json.loads(rows[0][5]) == {"datetime": "2020-01-01", "temp": 3.5, "tempmax": 5.0, "tempmin": 1.0}


class TestCheckCoverage:
    def test_complete_fetch_passes(self):
        backfill.check_coverage(366, date(2020, 1, 1), date(2020, 12, 31))

    def test_empty_fetch_is_an_error_because_fetch_days_swallows_failures(self):
        with pytest.raises(backfill.FetchIncomplete):
            backfill.check_coverage(0, date(2020, 1, 1), date(2020, 12, 31))

    def test_just_below_threshold_fails(self):
        with pytest.raises(backfill.FetchIncomplete):
            backfill.check_coverage(300, date(2020, 1, 1), date(2020, 12, 31))


class TestCompareSeries:
    def test_step_in_stored_series_is_removed(self):
        anchor = date(2026, 10, 6)
        start = date(1975, 1, 1)
        true = _daily(start, anchor, lambda d: 10 + 0.03 * (d.year - 1975))
        # Stored copy cools by 1 degC from 2017 on, like best_match in Hong Kong.
        stored = {d: v - (1.0 if d.year >= 2017 else 0.0) for d, v in true.items()}
        result = backfill.compare_series(stored, true, anchor)
        assert result["trend_before"] < 0.15
        assert result["trend_after"] == pytest.approx(0.3, abs=0.02)
        assert result["days_changed"] > 0

    def test_identical_series_changes_nothing(self):
        anchor = date(2026, 10, 6)
        series = _daily(date(1975, 1, 1), anchor, lambda d: 10 + 0.02 * (d.year - 1975))
        result = backfill.compare_series(series, series, anchor)
        assert result["trend_before"] == result["trend_after"]
        assert result["days_changed"] == 0
        assert result["mean_abs_diff"] == 0.0


class TestFetchSeries:
    async def test_raises_when_open_meteo_returns_too_little(self, monkeypatch):
        async def fake_fetch_days(lat, lon, start, end):
            return [{"datetime": "2020-01-01", "temp": 1.0, "tempmax": 2.0, "tempmin": 0.0}], {}

        monkeypatch.setattr("utils.open_meteo_client.fetch_days", fake_fetch_days)
        loc = backfill.LocationSummary(1, "x", 51.5, -0.1, 10, date(2020, 1, 1), date(2020, 12, 31), 0, 10, 0)
        with pytest.raises(backfill.FetchIncomplete):
            await backfill.fetch_series(loc, date(2020, 12, 31))


class TestParseArgs:
    def test_execute_requires_backup_dir(self):
        with pytest.raises(SystemExit):
            backfill.parse_args(["--execute"])

    def test_default_is_read_only(self):
        args = backfill.parse_args([])
        assert not args.execute and not args.compare

    def test_execute_with_backup_dir_is_accepted(self):
        assert backfill.parse_args(["--execute", "--backup-dir", "backups"]).execute


# --- Database-facing code, exercised with a fake connection (no Postgres needed) -------------------------------------


class _FakeTransaction:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.transactions += 1
        return self

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, summary_rows=(), stored_rows=(), backup_rows=()):
        self.summary_rows = list(summary_rows)
        self.stored_rows = list(stored_rows)
        self.backup_rows = list(backup_rows)
        self.queries = []
        self.executemany_calls = []
        self.transactions = 0
        self.closed = False

    async def fetch(self, sql, *params):
        self.queries.append((sql, params))
        if "GROUP BY l.id" in sql:
            return self.summary_rows
        if "payload::text" in sql:
            return self.backup_rows
        return self.stored_rows

    async def executemany(self, sql, rows):
        self.executemany_calls.append((sql, list(rows)))

    def transaction(self):
        return _FakeTransaction(self)

    async def close(self):
        self.closed = True


def _summary_row(location_id=1, name="london__england__united_kingdom", lat=51.5, lon=-0.1, pending=100):
    return {
        "id": location_id,
        "normalized_name": name,
        "latitude": lat,
        "longitude": lon,
        "total_rows": 120,
        "first_day": date(2020, 1, 1),
        "last_day": date(2020, 4, 30),
        "legacy_rows": 120,
        "pending_rows": pending,
        "future_rows": 0,
        "first_pending_day": date(2020, 1, 1),
    }


def _loc(**overrides):
    return backfill.LocationSummary(**{**_summary_row_as_kwargs(), **overrides})


def _summary_row_as_kwargs():
    row = _summary_row()
    return {
        "id": row["id"],
        "name": row["normalized_name"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "total_rows": row["total_rows"],
        "first_day": row["first_day"],
        "last_day": row["last_day"],
        "legacy_rows": row["legacy_rows"],
        "pending_rows": row["pending_rows"],
        "future_rows": row["future_rows"],
        "first_pending_day": row["first_pending_day"],
    }


def _args(**overrides):
    values = {
        "execute": False,
        "compare": False,
        "ids": None,
        "limit": None,
        "delay": 0.0,
        "force": False,
        "backup_dir": None,
    }
    values.update(overrides)
    return backfill.argparse.Namespace(**values)


def _fetched_days(start=date(2020, 1, 1), n=3):
    return [
        {"datetime": (start + timedelta(days=i)).isoformat(), "temp": 10.0 + i, "tempmax": 12.0, "tempmin": 8.0}
        for i in range(n)
    ]


class TestLoadSummaries:
    async def test_maps_rows_and_passes_query_parameters(self):
        conn = FakeConn(summary_rows=[_summary_row(), _summary_row(location_id=2, name="x", pending=0)])
        end, today = date(2026, 9, 29), date(2026, 10, 7)

        result = await backfill.load_summaries(conn, "open-meteo:era5_land", today, end, [1, 2])

        assert [s.id for s in result] == [1, 2]
        assert result[0].name == "london__england__united_kingdom"
        assert result[0].pending_rows == 100
        _sql, params = conn.queries[0]
        assert params[1:] == ("open-meteo:era5_land", end, today, [1, 2])
        assert params[0].isoformat().startswith(backfill.LEGACY_CUTOFF)


class TestSelectWork:
    def _summaries(self):
        return [_loc(id=1, pending_rows=5), _loc(id=2, pending_rows=0), _loc(id=3, pending_rows=9)]

    def test_skips_locations_already_on_the_pinned_source(self):
        todo, skipped = backfill.select_work(self._summaries(), _args())
        assert [s.id for s in todo] == [1, 3]
        assert skipped == 1

    def test_force_includes_everything(self):
        todo, skipped = backfill.select_work(self._summaries(), _args(force=True))
        assert [s.id for s in todo] == [1, 2, 3]
        assert skipped == 0

    def test_limit_applies_after_filtering(self):
        todo, skipped = backfill.select_work(self._summaries(), _args(limit=1))
        assert [s.id for s in todo] == [1]
        assert skipped == 1


class TestPrintReport:
    def test_lists_locations_with_estimated_calls_and_a_total(self, capsys):
        backfill.print_report([_loc()], 3, date(2026, 9, 29), "open-meteo:era5_land")
        out = capsys.readouterr().out
        assert "est-calls=" in out
        assert "1 locations need work (3 already on 'open-meteo:era5_land')" in out


class TestBackupPath:
    def test_stays_inside_the_backup_directory(self, tmp_path):
        path = backfill.backup_path(tmp_path, 7, "../../etc/passwd")
        assert path.parent == tmp_path.resolve()
        assert path.name == "location_7_______etc_passwd.csv"

    def test_ordinary_name_is_kept(self, tmp_path):
        assert backfill.backup_path(tmp_path, 1, "london__england__united_kingdom").name == (
            "location_1_london__england__united_kingdom.csv"
        )

    def test_long_names_are_truncated(self, tmp_path):
        assert len(backfill.backup_path(tmp_path, 1, "a" * 500).name) < 120


class TestWriteBackup:
    async def test_writes_header_and_rows(self, tmp_path):
        row = {
            "location_id": 1,
            "day": date(2020, 1, 1),
            "temp_c": 3.5,
            "temp_max_c": 5.0,
            "temp_min_c": 1.0,
            "payload": "{}",
            "source": "timeline",
            "updated_at": "2025-11-24",
        }
        conn = FakeConn(backup_rows=[row])

        path = await backfill.write_backup(conn, _loc(), date(2020, 4, 30), tmp_path / "bk")

        lines = path.read_text().splitlines()
        assert lines[0] == ",".join(backfill.BACKUP_COLUMNS)
        assert lines[1].startswith("1,2020-01-01,3.5,5.0,1.0")
        assert path.parent == (tmp_path / "bk").resolve()


class TestProcessLocation:
    async def test_compare_reports_trend_and_writes_nothing(self, monkeypatch, capsys):
        async def fake_fetch_series(loc, end):
            return _fetched_days()

        monkeypatch.setattr(backfill, "fetch_series", fake_fetch_series)
        conn = FakeConn(stored_rows=[{"day": date(2020, 1, 1), "temp_c": 9.0}])

        await backfill.process_location(conn, _loc(), _args(compare=True), date(2026, 10, 7), date(2020, 4, 30), "s")

        assert "trend" in capsys.readouterr().out
        assert conn.executemany_calls == []
        assert conn.transactions == 0

    async def test_execute_backs_up_then_upserts_in_a_transaction(self, monkeypatch, tmp_path, capsys):
        async def fake_fetch_series(loc, end):
            return _fetched_days()

        monkeypatch.setattr(backfill, "fetch_series", fake_fetch_series)
        conn = FakeConn(stored_rows=[{"day": date(2020, 1, 1), "temp_c": 9.0}])
        args = _args(execute=True, backup_dir=str(tmp_path))

        await backfill.process_location(
            conn, _loc(), args, date(2026, 10, 7), date(2020, 4, 30), "open-meteo:era5_land"
        )

        assert conn.transactions == 1
        ((sql, rows),) = conn.executemany_calls
        assert "ON CONFLICT (location_id, day) DO UPDATE" in sql
        assert len(rows) == 3
        assert all(r[6] == "open-meteo:era5_land" for r in rows)
        assert list(tmp_path.glob("location_1_*.csv"))
        assert "upserted 3 rows" in capsys.readouterr().out


class TestProcessAll:
    async def test_skips_locations_without_coordinates(self, monkeypatch, capsys):
        processed = []

        async def fake_process(conn, loc, args, today, end, source):
            processed.append(loc.id)

        monkeypatch.setattr(backfill, "process_location", fake_process)
        todo = [_loc(id=1, latitude=None, longitude=None), _loc(id=2)]

        code = await backfill.process_all(FakeConn(), todo, 0, _args(), date(2026, 10, 7), date(2026, 9, 29), "s")

        assert code == 0
        assert processed == [2]
        assert "no coordinates, skipped" in capsys.readouterr().out

    async def test_stops_with_exit_code_2_when_a_fetch_is_incomplete(self, monkeypatch, capsys):
        async def fake_process(conn, loc, args, today, end, source):
            raise backfill.FetchIncomplete("got 0 of 100 days")

        monkeypatch.setattr(backfill, "process_location", fake_process)

        code = await backfill.process_all(
            FakeConn(), [_loc(id=1), _loc(id=2)], 0, _args(), date(2026, 10, 7), date(2026, 9, 29), "s"
        )

        assert code == 2
        assert "Stopping; re-run later to resume" in capsys.readouterr().err

    async def test_waits_between_fetches_and_reminds_about_caches_after_execute(self, monkeypatch, capsys):
        sleeps = []

        async def fake_sleep(delay):
            sleeps.append(delay)

        async def fake_process(conn, loc, args, today, end, source):
            return None

        monkeypatch.setattr(backfill, "process_location", fake_process)
        monkeypatch.setattr(backfill.asyncio, "sleep", fake_sleep)

        code = await backfill.process_all(
            FakeConn(),
            [_loc(id=1), _loc(id=2), _loc(id=3)],
            0,
            _args(execute=True, backup_dir="x", delay=4.0),
            date(2026, 10, 7),
            date(2026, 9, 29),
            "s",
        )

        assert code == 0
        assert sleeps == [4.0, 4.0]  # between locations, not before the first
        assert "/cache/invalidate/location/" in capsys.readouterr().out


class TestRun:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.setenv("TEMPHIST_PG_DSN", "postgresql://example/db")

    def _connect_to(self, monkeypatch, conn):
        async def fake_connect(dsn):
            return conn

        monkeypatch.setattr(backfill.asyncpg, "connect", fake_connect)

    async def test_requires_a_dsn(self, monkeypatch, capsys):
        monkeypatch.delenv("TEMPHIST_PG_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert await backfill.run(_args()) == 1
        assert "TEMPHIST_PG_DSN" in capsys.readouterr().err

    async def test_refuses_to_fetch_when_the_model_is_not_pinned(self, monkeypatch, capsys):
        monkeypatch.setattr(backfill, "configured_model", lambda: "")
        assert await backfill.run(_args(compare=True)) == 1
        assert "refusing to fetch unpinned data" in capsys.readouterr().err

    async def test_report_mode_reads_only_and_closes_the_connection(self, monkeypatch, capsys):
        conn = FakeConn(summary_rows=[_summary_row()])
        self._connect_to(monkeypatch, conn)

        assert await backfill.run(_args()) == 0

        out = capsys.readouterr().out
        assert "REPORT (read-only)" in out
        assert "1 locations need work" in out
        assert conn.closed
        assert conn.executemany_calls == []

    async def test_report_mode_names_the_target_source_after_the_configured_model(self, monkeypatch, capsys):
        conn = FakeConn(summary_rows=[_summary_row()])
        self._connect_to(monkeypatch, conn)
        monkeypatch.setattr(backfill, "configured_model", lambda: "era5")

        assert await backfill.run(_args()) == 0

        assert "open-meteo:era5" in capsys.readouterr().out
        assert conn.queries[0][1][1] == "open-meteo:era5"

    async def test_report_mode_still_works_when_no_model_is_configured(self, monkeypatch, capsys):
        conn = FakeConn(summary_rows=[_summary_row()])
        self._connect_to(monkeypatch, conn)
        monkeypatch.setattr(backfill, "configured_model", lambda: "")

        assert await backfill.run(_args()) == 0

        assert "model=(unset)" in capsys.readouterr().out

    async def test_execute_mode_processes_locations_and_closes_the_http_client(self, monkeypatch, tmp_path):
        conn = FakeConn(summary_rows=[_summary_row()], stored_rows=[{"day": date(2020, 1, 1), "temp_c": 9.0}])
        self._connect_to(monkeypatch, conn)
        monkeypatch.setattr(backfill, "configured_model", lambda: "era5_land")
        closed = []

        async def fake_close_client():
            closed.append(True)

        async def fake_fetch_series(loc, end):
            return _fetched_days()

        monkeypatch.setattr("utils.open_meteo_client.close_client", fake_close_client)
        monkeypatch.setattr(backfill, "fetch_series", fake_fetch_series)

        assert await backfill.run(_args(execute=True, backup_dir=str(tmp_path))) == 0

        assert len(conn.executemany_calls) == 1
        assert conn.closed and closed == [True]


class TestMain:
    def test_returns_the_exit_code_from_run(self, monkeypatch):
        monkeypatch.delenv("TEMPHIST_PG_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert backfill.main(["--ids", "1"]) == 1


class TestModeLabel:
    def test_labels(self):
        assert backfill._mode_label(_args(execute=True)) == "EXECUTE"
        assert backfill._mode_label(_args(compare=True)) == "COMPARE (read-only)"
        assert backfill._mode_label(_args()) == "REPORT (read-only)"


class TestRefreshWindow:
    def test_starts_at_the_oldest_row_not_on_the_target_source(self):
        assert _loc(first_day=date(1975, 1, 1), first_pending_day=date(2026, 9, 1)).refresh_from == date(2026, 9, 1)

    def test_falls_back_to_the_whole_history_when_nothing_is_pending(self):
        assert _loc(first_day=date(1975, 1, 1), first_pending_day=None).refresh_from == date(1975, 1, 1)

    async def test_fetch_series_requests_only_the_refresh_window(self, monkeypatch):
        seen = {}

        async def fake_fetch_days(lat, lon, start, end):
            seen["window"] = (start, end)
            return _fetched_days(start, n=(end - start).days + 1), {}

        monkeypatch.setattr("utils.open_meteo_client.fetch_days", fake_fetch_days)
        loc = _loc(first_day=date(1975, 1, 1), first_pending_day=date(2026, 9, 1))

        await backfill.fetch_series(loc, date(2026, 9, 30))

        assert seen["window"] == (date(2026, 9, 1), date(2026, 9, 30))

    def test_report_estimate_follows_the_window(self, capsys):
        recent = _loc(first_day=date(1975, 1, 1), first_pending_day=date(2026, 9, 1))
        backfill.print_report([recent], 0, date(2026, 9, 30), "s")
        assert "est-calls=3\n" in capsys.readouterr().out
