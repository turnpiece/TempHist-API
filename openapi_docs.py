"""Presentation and post-processing for the generated OpenAPI spec.

The spec served at ``/openapi.json`` backs the public developer docs. Routing decides *what* is documented (see the
``include_in_schema`` flags in ``main.py`` and the routers); this module owns how it is presented: the title,
description and tags, the Firebase security scheme, and how the JSON is serialised.
"""

import json
from typing import Callable, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response
from starlette.routing import Route

from models import MiddlewareErrorResponse

API_TITLE = "TempHist API"

API_DESCRIPTION = """\
Historical temperature data for any day of the year, going back 50 years.

TempHist answers questions such as *"how does today compare with the same day over the last 50 years?"* It returns
one temperature per year for a chosen location and date, together with the average, the warming or cooling trend and
a plain-language summary.

## Authentication

Most endpoints require a Firebase ID token sent as `Authorization: Bearer <token>`. Anonymous Firebase users are
accepted. Endpoints that need a token declare the `FirebaseBearer` security scheme; endpoints without it are public.
When App Check enforcement is enabled for a deployment, requests must also carry a valid `X-Firebase-AppCheck` header.

## Periods and identifiers

Records are addressed by a `period` (`daily`, `weekly`, `monthly`, `yearly`), a `location` and an `identifier`. The
identifier is always an `MM-DD` date, for example `01-15`: the rolling window for the chosen period ends on that date
in every year of the record.

## Units

Temperatures are in Celsius unless `unit_group=fahrenheit` is requested.

## Rate limits and errors

Requests to `/weather`, `/forecast` and `/v1/records` are rate limited per client. A rejected request receives
`429 Too Many Requests` with a `Retry-After` header. Most errors use the `ErrorResponse` body. Authentication and rate
limiting are enforced before a request reaches an endpoint and use their own bodies, which are documented on each
operation.

## Data source

Historical temperatures come from the Open-Meteo ERA5 archive, licensed under CC BY 4.0.
"""

OPENAPI_TAGS = [
    {
        "name": "Records",
        "description": (
            "Temperature history for a location and date: one value per year, plus the average, trend, "
            "summary and ranking."
        ),
    },
    {
        "name": "Locations",
        "description": "Find locations to query: the curated list, popular locations and free-text search.",
    },
    {
        "name": "Shares",
        "description": "Create and read shareable snapshots of a record, and fetch their preview images.",
    },
    {
        "name": "Jobs",
        "description": (
            "Asynchronous record computation. Start a job with the `/async` endpoint, which returns `202 Accepted` "
            "and a `job_id`, then poll the job until its status is `ready` or `error`."
        ),
    },
    {
        "name": "Weather",
        "description": "Observed weather for a single date, and the forecast for the current day.",
    },
    {"name": "Health", "description": "Service liveness."},
]

FIREBASE_BEARER = "FirebaseBearer"

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_FIREBASE_BEARER_SCHEME = {
    "type": "http",
    "scheme": "bearer",
    "bearerFormat": "JWT",
    "description": (
        "A Firebase ID token for the signed-in or anonymous user, sent as `Authorization: Bearer <token>`. "
        "Requests with a missing or malformed header receive 401; a token that fails verification receives 403."
    ),
}

_HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})


def servers_for(base_url: str) -> Optional[list]:
    """The ``servers`` entry for the spec, taken from the per-environment ``BASE_URL`` setting.

    Docs hosted on another origin (Scalar, Redoc) call whatever server the spec names, so a deployed spec has to name
    its own API: production declares ``https://api.temphist.com`` and the dev deployment its own host. Hard-coding
    production would send "Try it out" on the dev docs to production.

    Returns None for an unset or localhost ``BASE_URL`` (the default). The spec then names no server, and OpenAPI
    tools resolve paths against the host that served it, which is right for local runs.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url or urlparse(url).hostname in _LOCAL_HOSTS:
        return None
    return [{"url": url}]


def _component_ref(name: str) -> dict:
    return {"$ref": f"#/components/schemas/{name}"}


def _json_content(name: str) -> dict:
    return {"application/json": {"schema": _component_ref(name)}}


def _use_error_response_for_422(responses: dict) -> None:
    """Point 422 at ErrorResponse: the app's exception handler replaces FastAPI's default ``{"detail": [...]}``."""
    validation = responses.get("422")
    if validation is not None:
        validation["content"] = _json_content("ErrorResponse")


def _postprocess(schema: dict, requires_auth: Callable[[str, str], bool]) -> None:
    components = schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})
    components.setdefault("securitySchemes", {})[FIREBASE_BEARER] = _FIREBASE_BEARER_SCHEME

    uses_auth = False
    for path, path_item in schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method not in _HTTP_METHODS:
                continue
            responses = operation.setdefault("responses", {})
            if "ErrorResponse" in schemas:
                _use_error_response_for_422(responses)
            if requires_auth(method.upper(), path):
                uses_auth = True
                operation["security"] = [{FIREBASE_BEARER: []}]
                responses.setdefault(
                    "401",
                    {
                        "description": "Missing or invalid Authorization header",
                        "content": _json_content("MiddlewareErrorResponse"),
                    },
                )
                responses.setdefault(
                    "403",
                    {
                        "description": "The Firebase token could not be verified",
                        "content": _json_content("MiddlewareErrorResponse"),
                    },
                )

    if uses_auth:
        schemas.setdefault("MiddlewareErrorResponse", MiddlewareErrorResponse.model_json_schema())

    # FastAPI's default 422 body is no longer referenced once every operation points at ErrorResponse.
    if "#/components/schemas/HTTPValidationError" not in json.dumps(schema.get("paths", {})):
        schemas.pop("HTTPValidationError", None)
        schemas.pop("ValidationError", None)


def _serve_ascii_openapi(app: FastAPI) -> None:
    """Serve the spec with every non-ASCII character escaped as ``\\uXXXX``.

    FastAPI answers with ``Content-Type: application/json`` and raw UTF-8, which is correct, but some consumers
    (importers, fetch tools, editors) decode a response with no declared charset as Latin-1 and turn ``°C`` into
    ``Â°C``. The escaped form is the same JSON to any parser and cannot be misread, whatever encoding is assumed.
    The spec legitimately contains non-ASCII text, such as the ``°C/decade`` trend unit, so rewording descriptions
    would not remove the problem.
    """
    url = app.openapi_url
    builtin = next((r for r in app.router.routes if isinstance(r, Route) and r.path == url), None)
    if builtin is None:
        return

    async def openapi_ascii(request: Request) -> Response:
        # Delegate to FastAPI's own handler so root_path/servers handling stays identical, then re-encode.
        body = (await builtin.endpoint(request)).body
        text = json.dumps(json.loads(body), ensure_ascii=True, separators=(",", ":"))
        return Response(content=text, media_type="application/json")

    app.router.routes[app.router.routes.index(builtin)] = Route(url, openapi_ascii, include_in_schema=False)


def install_openapi(app: FastAPI, *, requires_auth: Callable[[str, str], bool]) -> None:
    """Post-process the generated spec and serve it ASCII-escaped.

    ``requires_auth(method, path)`` must answer the same question the auth middleware does, so the documented
    security requirements cannot drift from what is enforced.
    """
    processed: dict | None = None

    def openapi() -> dict:
        nonlocal processed
        schema = FastAPI.openapi(app)
        if schema is not processed:  # FastAPI caches the dict; rebuild means a new object to process
            _postprocess(schema, requires_auth)
            processed = schema
        return schema

    app.openapi = openapi
    _serve_ascii_openapi(app)
