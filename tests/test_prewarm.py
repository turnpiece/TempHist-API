"""Tests for the input validation in prewarm.py (SonarQube S8703 / S8707 hardening)."""

import argparse
import json
import os

import pytest

import prewarm


class TestValidatedBaseUrl:
    @pytest.mark.parametrize("url", ["http://localhost:8000", "https://api.example.com/"])
    def test_accepts_http_and_https(self, url):
        assert prewarm._validated_base_url(url) == url.rstrip("/")

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "http://", "not-a-url", ""])
    def test_rejects_other_schemes_and_missing_host(self, url):
        with pytest.raises(argparse.ArgumentTypeError):
            prewarm._validated_base_url(url)


class TestValidatedLocationsFile:
    def test_accepts_file_inside_project(self):
        path = os.path.join(prewarm.PROJECT_ROOT, "data", "preapproved_locations.json")
        assert prewarm._validated_locations_file(path) == os.path.realpath(path)

    def test_rejects_file_outside_allowed_roots(self, tmp_path, monkeypatch):
        outside = tmp_path / "locations.json"
        outside.write_text("[]")
        # tmp_path is outside both the project root and the (monkeypatched) cwd.
        monkeypatch.chdir(prewarm.PROJECT_ROOT)
        with pytest.raises(argparse.ArgumentTypeError, match="must be inside"):
            prewarm._validated_locations_file(str(outside))

    def test_rejects_parent_directory_traversal(self):
        with pytest.raises(argparse.ArgumentTypeError, match="must be inside"):
            prewarm._validated_locations_file(os.path.join(prewarm.PROJECT_ROOT, "..", "..", "etc", "hosts"))

    def test_accepts_file_inside_cwd(self, tmp_path, monkeypatch):
        inside = tmp_path / "locations.json"
        inside.write_text("[]")
        monkeypatch.chdir(tmp_path)
        assert prewarm._validated_locations_file("locations.json") == os.path.realpath(inside)

    def test_rejects_missing_file(self):
        with pytest.raises(argparse.ArgumentTypeError, match="does not exist"):
            prewarm._validated_locations_file(os.path.join(prewarm.PROJECT_ROOT, "no_such_file.json"))


class TestLoadPreapprovedLocations:
    def test_loads_and_dedupes_from_cwd_file(self, tmp_path, monkeypatch):
        data = [
            {"name": "London", "admin1": "England", "country_name": "United Kingdom"},
            {"name": "london", "admin1": "england", "country_name": "united kingdom"},
            {"name": "Paris", "country_name": "France"},
            {"admin1": "No name entry"},
        ]
        (tmp_path / "locs.json").write_text(json.dumps(data))
        monkeypatch.chdir(tmp_path)
        assert prewarm.load_preapproved_locations("locs.json") == [
            "London, England, United Kingdom",
            "Paris, France",
        ]

    def test_returns_empty_for_path_outside_allowed_roots(self):
        assert prewarm.load_preapproved_locations("/etc/hosts") == []

    def test_returns_empty_for_invalid_json(self, tmp_path, monkeypatch):
        (tmp_path / "bad.json").write_text("{not json")
        monkeypatch.chdir(tmp_path)
        assert prewarm.load_preapproved_locations("bad.json") == []

    def test_default_file_loads(self):
        assert prewarm.load_preapproved_locations()


class TestLoadLocationsToPrewarm:
    def test_rejects_non_http_base_url_and_falls_back(self, monkeypatch):
        monkeypatch.setattr(prewarm, "load_preapproved_locations", lambda _f=None: ["fallback"])
        assert prewarm.load_locations_to_prewarm("file:///etc/passwd", "token", 5) == ["fallback"]

    def test_uses_popular_locations_from_api(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"locations": ["London", "Paris"]}).encode()

        def fake_urlopen(req, timeout):
            captured["url"] = req.full_url
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        result = prewarm.load_locations_to_prewarm("http://localhost:8000/", "token", 2)
        assert result == ["London", "Paris"]
        assert captured["url"] == "http://localhost:8000/v1/locations/popular/display-strings?limit=2"

    def test_falls_back_when_api_returns_too_few_locations(self, monkeypatch):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"locations": ["London"]}).encode()

        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: FakeResponse())
        monkeypatch.setattr(prewarm, "load_preapproved_locations", lambda _f=None: ["fallback"])
        assert prewarm.load_locations_to_prewarm("http://localhost:8000", "token", 2) == ["fallback"]

    def test_without_token_uses_preapproved_list(self, monkeypatch):
        monkeypatch.setattr(prewarm, "load_preapproved_locations", lambda _f=None: ["fallback"])
        assert prewarm.load_locations_to_prewarm("http://localhost:8000", None, 5) == ["fallback"]
