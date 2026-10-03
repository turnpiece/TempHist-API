"""Shared, documented path and query parameters for the public routes.

``pattern`` and ``enum`` entries passed through ``json_schema_extra`` are documentation only: FastAPI does not enforce
them. That is deliberate. The handlers already reject malformed input with their own 400 responses (and accept some
looser forms, such as ``1-5`` for an identifier), so enforcing the regex here would turn those responses into 422s.
"""

from typing import Annotated, Literal

from fastapi import Path, Query

PeriodParam = Annotated[
    Literal["daily", "weekly", "monthly", "yearly"],
    Path(
        description=(
            "Aggregation period. Each period is a rolling window that ends on the identifier date: "
            "`daily` is that single day, `weekly` the 7 days ending on it, `monthly` the 31 days ending on it "
            "and `yearly` the 365 days ending on it."
        ),
        examples=["daily"],
    ),
]

LocationParam = Annotated[
    str,
    Path(
        description=(
            "Location name or canonical location ID, for example `london` or "
            "`Cape Town, Western Cape, South Africa`. Use the Locations endpoints to find valid values."
        ),
        max_length=200,
        examples=["london"],
    ),
]

IdentifierParam = Annotated[
    str,
    Path(
        description=(
            "Period end date as `MM-DD` with a zero-padded month and day, for example `01-15`. The format is the "
            "same for every period; the window for the chosen `period` ends on this date in each year of the record."
        ),
        examples=["01-15"],
        json_schema_extra={"pattern": r"^\d{2}-\d{2}$"},
    ),
]

UnitGroupParam = Annotated[
    Literal["celsius", "fahrenheit"],
    Query(description="Temperature unit for the response.", examples=["celsius"]),
]

# /weather and /forecast take a free string and treat anything other than fahrenheit (and the legacy alias "us") as
# celsius. The spec advertises only the two supported values; the leniency is kept for backwards compatibility.
LenientUnitGroupParam = Annotated[
    str,
    Query(
        description="Temperature unit: `celsius` (default) or `fahrenheit`.",
        examples=["celsius"],
        json_schema_extra={"enum": ["celsius", "fahrenheit"]},
    ),
]

ShareIdParam = Annotated[
    str,
    Path(
        description="Share ID: 8 alphanumeric characters, as returned when the share was created.",
        examples=["aB3dE5gH"],
        json_schema_extra={"pattern": r"^[A-Za-z0-9]{8}$"},  # documentation only; the handler returns 404
    ),
]
