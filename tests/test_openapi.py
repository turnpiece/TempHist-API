"""Contract tests for the generated OpenAPI spec that backs the public developer docs."""

import asyncio
import json
from collections import Counter

import pytest
from fastapi.testclient import TestClient

from config import BASE_URL
from main import app, is_public_route
from models import (
    ErrorResponse,
    MetaResponse,
    MiddlewareErrorResponse,
    RateLimitErrorResponse,
    RecordResponse,
    SimpleErrorResponse,
)
from openapi_docs import servers_for
from routers.health import HealthResponse, health_check
from routers.locations import LocationsStatusResponse, get_locations_status, get_popular_locations_status
from routers.shares import ShareCreate
from version import __version__

HTTP_METHODS = {"get", "put", "post", "delete", "options", "head", "patch"}

# Every operation intended for the public docs. A new route shows up in the spec unless it is excluded with
# include_in_schema=False, and this test then fails until the route is added here (or hidden) on purpose.
DOCUMENTED_OPERATIONS = {
    ("GET", "/forecast/{location}"),
    ("GET", "/health"),
    ("GET", "/v1/jobs/{job_id}"),
    ("GET", "/v1/locations/popular"),
    ("GET", "/v1/locations/popular/status"),
    ("GET", "/v1/locations/preapproved"),
    ("GET", "/v1/locations/preapproved/status"),
    ("GET", "/v1/locations/search"),
    ("POST", "/v1/locations/selections"),
    ("GET", "/v1/og/{share_id}.png"),
    ("GET", "/v1/records/{period}/{location}/{identifier}"),
    ("POST", "/v1/records/{period}/{location}/{identifier}/async"),
    ("GET", "/v1/records/{period}/{location}/{identifier}/average"),
    ("GET", "/v1/records/{period}/{location}/{identifier}/meta"),
    ("GET", "/v1/records/{period}/{location}/{identifier}/summary"),
    ("GET", "/v1/records/{period}/{location}/{identifier}/trend"),
    ("GET", "/v1/records/{period}/{location}/{identifier}/updated"),
    ("GET", "/v1/shares"),
    ("POST", "/v1/shares"),
    ("GET", "/v1/shares/{share_id}"),
    ("GET", "/weather/{location}/{date}"),
}

# Routes that must stay out of the docs but keep working.
HIDDEN_PATHS = {
    "/",
    "/admin/clear-job-queue",
    "/analytics",
    "/average/{location}/{month_day}",
    "/cache/info",
    "/cache-stats",
    "/cache-warm",
    "/data/{location}/{month_day}",
    "/debug/jobs",
    "/health/detailed",
    "/protected-endpoint",
    "/rate-limit-stats",
    "/rate-limit-status",
    "/test-cors",
    "/test-cors-rolling",
    "/test-redis",
    "/usage-stats",
    "/v1/jobs/diagnostics/worker-status",
    "/v1/locations/popular/display-strings",
    "/v1/locations/popular/stats",
    "/v1/records/rolling-bundle/{location}/{anchor}/async",
}

RATE_LIMITED_PREFIXES = ("/weather/", "/forecast/", "/v1/records/")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def raw_spec(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.content


@pytest.fixture(scope="module")
def spec(raw_spec):
    return json.loads(raw_spec)


def operations(spec):
    for path, item in spec["paths"].items():
        for method, operation in item.items():
            if method in HTTP_METHODS:
                yield method.upper(), path, operation


def test_info_block(spec):
    assert spec["info"]["title"] == "TempHist API"
    assert spec["info"]["version"] == __version__
    assert spec["info"]["description"].strip()


def test_only_intended_operations_are_documented(spec):
    documented = {(method, path) for method, path, _ in operations(spec)}
    assert documented == DOCUMENTED_OPERATIONS


def test_hidden_routes_are_registered_but_not_documented(spec):
    registered = {route.path for route in app.routes}
    assert HIDDEN_PATHS <= registered, f"no longer routable: {sorted(HIDDEN_PATHS - registered)}"
    assert not HIDDEN_PATHS & set(spec["paths"])


def test_hidden_routes_still_answer(client):
    assert client.get("/test-cors").status_code == 200
    assert client.options("/").status_code == 200
    removed = client.get("/data/london/01-15")
    assert removed.status_code == 410
    assert removed.headers["X-New-Endpoint"] == "/v1/records/daily/{location}/{month_day}"


def test_every_operation_has_a_declared_tag(spec):
    declared = {tag["name"] for tag in spec["tags"]}
    assert declared == {"Records", "Locations", "Shares", "Jobs", "Weather", "Health"}
    for method, path, operation in operations(spec):
        assert len(operation.get("tags", [])) == 1, f"{method} {path}"
        assert operation["tags"][0] in declared, f"{method} {path}"
    assert all(tag["description"] for tag in spec["tags"])


def test_operation_ids_are_unique(spec):
    counts = Counter(operation["operationId"] for _, _, operation in operations(spec))
    assert [op_id for op_id, n in counts.items() if n > 1] == []


def test_no_response_has_an_empty_schema(spec):
    empty = [
        (method, path, status, media_type)
        for method, path, operation in operations(spec)
        for status, response in operation["responses"].items()
        for media_type, content in response.get("content", {}).items()
        if not content.get("schema")
    ]
    assert empty == []


def test_every_success_response_declares_a_body_unless_no_content(spec):
    for method, path, operation in operations(spec):
        success = [status for status in operation["responses"] if status.startswith("2")]
        assert success, f"{method} {path} documents no success response"
        for status in success:
            if status != "204":
                assert operation["responses"][status].get("content"), f"{method} {path} {status}"


def test_security_matches_what_the_middleware_enforces(spec):
    schemes = spec["components"]["securitySchemes"]
    assert set(schemes) == {"FirebaseBearer"}
    assert schemes["FirebaseBearer"]["type"] == "http"
    assert schemes["FirebaseBearer"]["scheme"] == "bearer"

    for method, path, operation in operations(spec):
        needs_auth = not is_public_route(path, method)
        assert ("security" in operation) == needs_auth, f"{method} {path}"
        assert ("401" in operation["responses"]) == needs_auth, f"{method} {path}"
        assert ("403" in operation["responses"]) == needs_auth, f"{method} {path}"
        if needs_auth:
            assert operation["security"] == [{"FirebaseBearer": []}]


def test_specific_routes_are_public_or_protected(spec):
    by_route = {(method, path): operation for method, path, operation in operations(spec)}
    for route in [
        ("POST", "/v1/shares"),
        ("POST", "/v1/locations/selections"),
        ("GET", "/v1/records/{period}/{location}/{identifier}"),
        ("GET", "/weather/{location}/{date}"),
    ]:
        assert "security" in by_route[route], route
    for route in [
        ("GET", "/health"),
        ("GET", "/v1/shares"),
        ("GET", "/v1/shares/{share_id}"),
        ("GET", "/v1/og/{share_id}.png"),
    ]:
        assert "security" not in by_route[route], route


def test_admin_key_is_not_documented(raw_spec):
    assert b"X-Admin-Key" not in raw_spec
    assert b"/admin/" not in raw_spec


def test_validation_errors_are_documented_with_the_real_body(spec):
    schemas = spec["components"]["schemas"]
    assert "HTTPValidationError" not in schemas
    for method, path, operation in operations(spec):
        if "422" in operation["responses"]:
            body = operation["responses"]["422"]["content"]["application/json"]["schema"]
            assert body == {"$ref": "#/components/schemas/ErrorResponse"}, f"{method} {path}"


def test_rate_limited_routes_document_429_with_retry_after(spec):
    for method, path, operation in operations(spec):
        if path.startswith(RATE_LIMITED_PREFIXES):
            response = operation["responses"]["429"]
            assert "Retry-After" in response["headers"], f"{method} {path}"
            body = response["content"]["application/json"]["schema"]
            assert body == {"$ref": "#/components/schemas/RateLimitErrorResponse"}, f"{method} {path}"


def test_every_documented_429_and_503_declares_retry_after(spec):
    """Each route that documents one of these must send the header (tests/test_retry_after.py checks it does)."""
    checked = 0
    for method, path, operation in operations(spec):
        for status in ("429", "503"):
            if status in operation["responses"]:
                checked += 1
                assert "Retry-After" in operation["responses"][status]["headers"], f"{method} {path} {status}"
    assert checked


def test_async_job_documents_202_not_200(spec):
    responses = spec["paths"]["/v1/records/{period}/{location}/{identifier}/async"]["post"]["responses"]
    assert "202" in responses
    assert "200" not in responses
    assert "Retry-After" in responses["503"]["headers"]


def test_og_image_declares_png(spec):
    ok = spec["paths"]["/v1/og/{share_id}.png"]["get"]["responses"]["200"]
    assert list(ok["content"]) == ["image/png"]
    assert ok["content"]["image/png"]["schema"] == {"type": "string", "format": "binary"}


def test_record_parameters_are_documented(spec):
    parameters = {
        p["name"]: p["schema"]
        for p in spec["paths"]["/v1/records/{period}/{location}/{identifier}"]["get"]["parameters"]
    }
    assert parameters["identifier"]["pattern"] == r"^\d{2}-\d{2}$"
    assert parameters["identifier"]["examples"] == ["01-15"]
    assert parameters["period"]["examples"] == ["daily"]
    assert parameters["location"]["examples"] == ["london"]
    assert parameters["unit_group"]["enum"] == ["celsius", "fahrenheit"]


def test_weather_and_forecast_advertise_the_unit_enum_and_date_pattern(spec):
    for path in ("/weather/{location}/{date}", "/forecast/{location}"):
        parameters = {p["name"]: p["schema"] for p in spec["paths"][path]["get"]["parameters"]}
        assert parameters["unit_group"]["enum"] == ["celsius", "fahrenheit"], path
    date = next(p for p in spec["paths"]["/weather/{location}/{date}"]["get"]["parameters"] if p["name"] == "date")
    assert date["schema"]["pattern"] == r"^\d{4}-\d{2}-\d{2}$"


def test_popular_locations_do_not_require_image_fields(spec):
    schemas = spec["components"]["schemas"]
    assert set(schemas["PopularLocationItem"]["required"]) == {"id", "slug", "name"}
    # /preapproved always includes images, so its item model keeps requiring them.
    assert {"imageUrl", "imageAlt"} <= set(schemas["LocationItem"]["required"])


def test_identifier_description_no_longer_claims_other_formats(spec):
    description = spec["components"]["schemas"]["RecordResponse"]["properties"]["identifier"]["description"]
    assert "YYYY-MM" not in description


@pytest.mark.parametrize(
    "model",
    [
        RecordResponse,
        MetaResponse,
        ErrorResponse,
        MiddlewareErrorResponse,
        RateLimitErrorResponse,
        SimpleErrorResponse,
        ShareCreate,
    ],
)
def test_schema_examples_validate_against_their_model(model):
    examples = model.model_json_schema()["examples"]
    assert examples
    for example in examples:
        model.model_validate(example)


def test_key_schemas_carry_examples(spec):
    schemas = spec["components"]["schemas"]
    for name in ("RecordResponse", "MetaResponse", "ErrorResponse"):
        assert schemas[name]["examples"], name


def test_served_spec_is_ascii_so_no_consumer_can_misdecode_it(client, raw_spec):
    assert client.get("/openapi.json").headers["content-type"].startswith("application/json")
    assert raw_spec.isascii()

    as_utf8 = json.loads(raw_spec.decode("utf-8"))
    as_latin1 = json.loads(raw_spec.decode("latin-1"))
    assert as_latin1 == as_utf8
    assert as_utf8["components"]["schemas"]["TrendData"]["properties"]["unit"]["default"] == "°C/decade"

    mojibake = ["â€", "Â°", "Ã\u0097", "âˆ"]  # what UTF-8 looks like as cp1252
    text = raw_spec.decode("latin-1")
    assert not [marker for marker in mojibake if marker in text]


def test_served_spec_matches_the_generated_one(spec):
    assert spec == app.openapi()


def test_docs_pages_still_render(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200


def test_documented_inline_responses_match_what_the_handlers_return():
    health = HealthResponse.model_validate(asyncio.run(health_check()))
    assert health.status == "healthy"
    for handler in (get_locations_status, get_popular_locations_status):
        LocationsStatusResponse.model_validate(asyncio.run(handler()))


# Implementation detail that belongs in code comments, not in public documentation.
INTERNAL_TERMS = [
    "SSRF",
    "MAPBOX",
    "Render",
    "dev / CI",
    "Response shape",
    "usage tracker",
    "validate_location",
    "Redis",
]


def test_every_operation_has_a_public_facing_description(spec):
    for method, path, operation in operations(spec):
        description = operation.get("description", "")
        assert description.strip(), f"{method} {path} has no description"
        leaked = [term for term in INTERNAL_TERMS if term in description]
        assert not leaked, f"{method} {path} exposes internal detail: {leaked}"


def test_og_image_summary_is_readable(spec):
    assert spec["paths"]["/v1/og/{share_id}.png"]["get"]["summary"] == "Get share preview image"


@pytest.mark.parametrize(
    "base_url, expected",
    [
        ("https://api.temphist.com", [{"url": "https://api.temphist.com"}]),
        ("https://devapi.temphist.com/", [{"url": "https://devapi.temphist.com"}]),
        ("http://localhost:8000", None),
        ("http://127.0.0.1:8000", None),
        ("http://[::1]:8000", None),
        ("", None),
    ],
)
def test_servers_come_from_base_url_and_are_omitted_for_local_runs(base_url, expected):
    assert servers_for(base_url) == expected


def test_app_declares_the_server_for_its_own_environment(spec):
    expected = servers_for(BASE_URL)
    assert spec.get("servers") == expected
