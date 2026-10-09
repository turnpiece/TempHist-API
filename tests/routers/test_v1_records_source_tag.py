"""Stored daily records carry the provider tag fetched with them (P1-173)."""

from datetime import date

from routers.v1_records import _timeline_days_to_records


def _day(**extra):
    return {"datetime": "2024-06-01", "temp": 15.0, "tempmax": 20.0, "tempmin": 10.0, **extra}


def test_record_uses_the_source_the_provider_reported():
    (record,) = _timeline_days_to_records([_day(source="open-meteo:era5")], date(2024, 6, 1), date(2024, 6, 1))
    assert record.source == "open-meteo:era5"


def test_record_falls_back_to_timeline_when_the_provider_gave_no_source():
    (record,) = _timeline_days_to_records([_day()], date(2024, 6, 1), date(2024, 6, 1))
    assert record.source == "timeline"
