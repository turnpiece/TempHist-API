"""Pydantic models for API requests and responses."""

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# Example payloads rendered in the OpenAPI docs. tests/test_openapi.py validates each one against its model.
_RECORD_EXAMPLE = {
    "period": "daily",
    "location": "london",
    "identifier": "01-15",
    "range": {"start": "2021-01-15", "end": "2025-01-15", "years": 5},
    "unit_group": "celsius",
    "values": [
        {"date": "2021-01-15", "year": 2021, "temperature": 8.2, "anomaly": -0.25},
        {"date": "2022-01-15", "year": 2022, "temperature": 9.1, "anomaly": 0.65},
        {"date": "2023-01-15", "year": 2023, "temperature": 7.6, "anomaly": -0.85},
        {"date": "2024-01-15", "year": 2024, "temperature": 8.9, "anomaly": 0.45},
        {"date": "2025-01-15", "year": 2025, "temperature": 8.8, "anomaly": 0.35},
    ],
    "average": {"mean": 8.45, "unit": "celsius", "data_points": 5, "standard_deviation": 0.55},
    "trend": {
        "slope": 1.0,
        "unit": "°C/decade",
        "data_points": 5,
        "r_squared": 0.07,
        "slope_error": 2.17,
        "gradient_factor": 0.0,
    },
    "summary": "8.8°C. It's not as warm as last year but warmer than 2023. "
    "It was 0.4°C warmer than average for the time of year.",
    "metadata": {"total_years": 5, "available_years": 5, "missing_years": [], "completeness": 100.0},
    "updated": "2025-01-15T09:30:00+00:00",
    "timezone": "Europe/London",
}

_META_EXAMPLE = {
    "period": "daily",
    "location": "london",
    "identifier": "01-15",
    "data": {
        "summary": _RECORD_EXAMPLE["summary"],
        "average": _RECORD_EXAMPLE["average"],
        "trend": _RECORD_EXAMPLE["trend"],
        "ranking": {"warm": 3, "cold": 3, "total": 5},
        "current_anomaly": 0.35,
    },
    "metadata": _RECORD_EXAMPLE["metadata"],
    "timezone": "Europe/London",
}

_ERROR_EXAMPLE = {
    "error": "NOT_FOUND",
    "message": "Job not found",
    "code": "NOT_FOUND",
    "details": None,
    "path": "/v1/jobs/record_computation_1768469400000_ab12cd34",
    "method": "GET",
    "request_id": "3f2b8c1e-5d4a-4e8b-9a77-1c2d3e4f5a6b",
    "timestamp": "2025-01-15T09:30:00.000000",
}


# Pydantic Models for v1 API
class TemperatureValue(BaseModel):
    """Individual temperature data point."""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    year: int = Field(..., description="Year")
    temperature: float = Field(..., description="Temperature value")
    anomaly: Optional[float] = Field(None, description="Deviation from the historical mean (temperature − mean)")


class DateRange(BaseModel):
    """Date range for the data."""

    start: str = Field(..., description="Start date in YYYY-MM-DD format")
    end: str = Field(..., description="End date in YYYY-MM-DD format")
    years: int = Field(..., description="Number of years in range")


class AverageData(BaseModel):
    """Average temperature statistics."""

    mean: float = Field(..., description="Mean temperature")
    unit: str = Field("celsius", description="Temperature unit (celsius or fahrenheit)")
    data_points: int = Field(..., description="Number of data points used")
    standard_deviation: Optional[float] = Field(
        None, description="Population standard deviation of all values in the series"
    )


class TrendData(BaseModel):
    """Temperature trend analysis."""

    slope: float = Field(..., description="Temperature change per decade")
    unit: str = Field("°C/decade", description="Trend unit (changes based on temperature unit)")
    data_points: int = Field(..., description="Number of data points used")
    r_squared: Optional[float] = Field(None, description="R-squared value for trend fit")
    slope_error: Optional[float] = Field(
        None, description="Standard error of the slope (one SE); 95% CI is approximately slope ± 2 × slope_error"
    )
    gradient_factor: Optional[float] = Field(
        None,
        description="Normalised trend intensity [-1.0 cooling … 1.0 warming] adjusted for slope uncertainty; intended for frontend colour gradients.",
    )


class RankingData(BaseModel):
    """Year ranking within the historical record."""

    warm: int = Field(..., description="Rank by warmth (1 = warmest on record)")
    cold: int = Field(..., description="Rank by coldness (1 = coldest on record)")
    total: int = Field(..., description="Total number of years with data")


class UpdatedResponse(BaseModel):
    """Response model for updated timestamp endpoint."""

    period: str = Field(..., description="Data period")
    location: str = Field(..., description="Location name")
    identifier: str = Field(..., description="Date identifier")
    updated: Optional[str] = Field(None, description="ISO timestamp when data was last updated, null if not cached")
    cached: bool = Field(..., description="Whether the data is currently cached")
    cache_key: str = Field(..., description="Cache key used for this endpoint")


class RecordResponse(BaseModel):
    """Main record response for v1 API."""

    model_config = ConfigDict(json_schema_extra={"examples": [_RECORD_EXAMPLE]})

    period: Literal["daily", "weekly", "monthly", "yearly"] = Field(..., description="Data period")
    location: str = Field(..., description="Location name")
    identifier: str = Field(..., description="Period end date as MM-DD (the same format for every period)")
    range: DateRange = Field(..., description="Date range covered")
    unit_group: str = Field("celsius", description="Temperature unit used")
    values: List[TemperatureValue] = Field(..., description="Temperature data points")
    average: AverageData = Field(..., description="Average temperature statistics")
    trend: TrendData = Field(..., description="Temperature trend analysis")
    summary: str = Field(..., description="Human-readable summary")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    updated: Optional[str] = Field(None, description="ISO timestamp when data was last updated (if cached)")
    timezone: Optional[str] = Field(
        None, description="IANA timezone identifier for the location (e.g., 'America/New_York', 'Europe/London')"
    )


class SubResourceResponse(BaseModel):
    """Response for subresource endpoints."""

    period: Literal["daily", "weekly", "monthly", "yearly"] = Field(..., description="Data period")
    location: str = Field(..., description="Location name")
    identifier: str = Field(..., description="Date identifier")
    data: Union[AverageData, TrendData, str] = Field(..., description="Subresource data")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    timezone: Optional[str] = Field(
        None, description="IANA timezone identifier for the location (e.g., 'America/New_York', 'Europe/London')"
    )


class MetaData(BaseModel):
    """Combined summary, average and trend payload for the /meta sub-resource."""

    summary: str = Field(..., description="Human-readable summary")
    average: AverageData = Field(..., description="Average temperature statistics")
    trend: TrendData = Field(..., description="Temperature trend analysis")
    ranking: RankingData = Field(..., description="Rank of the most recent year within the historical record")
    current_anomaly: Optional[float] = Field(
        None, description="Current year's temperature deviation from the historical mean"
    )


class MetaResponse(BaseModel):
    """Response model for the /meta sub-resource endpoint."""

    model_config = ConfigDict(json_schema_extra={"examples": [_META_EXAMPLE]})

    period: Literal["daily", "weekly", "monthly", "yearly"] = Field(..., description="Data period")
    location: str = Field(..., description="Location name")
    identifier: str = Field(..., description="Date identifier")
    data: MetaData = Field(..., description="Combined summary, average and trend data")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")
    timezone: Optional[str] = Field(
        None, description="IANA timezone identifier for the location (e.g., 'America/New_York', 'Europe/London')"
    )


# Analytics Models
class ErrorDetail(BaseModel):
    """Individual error detail."""

    timestamp: str = Field(..., description="Error timestamp in ISO format")
    error_type: str = Field(..., description="Type of error (network, api, validation, etc.)")
    message: str = Field(..., description="Error message")
    location: Optional[str] = Field(None, description="Location where error occurred")
    endpoint: Optional[str] = Field(None, description="API endpoint that failed")
    status_code: Optional[int] = Field(None, description="HTTP status code if applicable")


class AnalyticsData(BaseModel):
    """Analytics data from client applications."""

    session_duration: int = Field(..., ge=0, description="Session duration in seconds")
    api_calls: int = Field(..., ge=0, description="Total number of API calls made")
    api_failure_rate: str = Field(..., description="API failure rate as percentage (e.g., '0%', '15%')")
    retry_attempts: int = Field(..., ge=0, description="Number of retry attempts made")
    location_failures: int = Field(..., ge=0, description="Number of location-related failures")
    error_count: int = Field(..., ge=0, description="Total number of errors encountered")
    recent_errors: List[ErrorDetail] = Field(default_factory=list, description="Recent error details")
    app_version: Optional[str] = Field(None, description="Client application version")
    platform: Optional[str] = Field(None, description="Platform (web, mobile, desktop)")
    user_agent: Optional[str] = Field(None, description="User agent string")
    session_id: Optional[str] = Field(None, description="Unique session identifier")
    response_time_ms: Optional[int] = Field(None, ge=0, description="Client-measured response time in milliseconds")
    cache_hit: Optional[bool] = Field(
        None, description="Whether the response was served from cache (derived from X-Cache header)"
    )
    canonical_location: Optional[str] = Field(None, description="Canonical location name resolved by the API")
    requested_location: Optional[str] = Field(None, description="Location as originally entered by the user")
    selection_method: Optional[Literal["own_location", "carousel", "recent", "popular", "search"]] = Field(
        None, description="How the location was selected"
    )


class AnalyticsResponse(BaseModel):
    """Response for analytics submission."""

    status: str = Field(..., description="Submission status")
    message: str = Field(..., description="Response message")
    analytics_id: str = Field(..., description="Unique analytics record ID")
    timestamp: str = Field(..., description="Submission timestamp")


# Error Response Model
class ErrorResponse(BaseModel):
    """Standardized error response format for consistent API error handling."""

    model_config = ConfigDict(json_schema_extra={"examples": [_ERROR_EXAMPLE]})

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    code: Optional[str] = Field(None, description="Error code for programmatic handling")
    details: Optional[Union[List[Dict], Dict, str]] = Field(None, description="Additional error details")
    path: Optional[str] = Field(None, description="Request path where error occurred")
    method: Optional[str] = Field(None, description="HTTP method")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")
    timestamp: str = Field(
        default_factory=lambda: __import__("datetime").datetime.now().isoformat(), description="Error timestamp"
    )


# The shapes below are what the request middleware and a few endpoints return directly, bypassing the
# exception handlers that produce ErrorResponse. They exist so the OpenAPI docs describe what a client sees.
class MiddlewareErrorResponse(BaseModel):
    """Error body returned by the authentication middleware (401 and 403)."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"detail": "Missing or invalid Authorization header."}]},
    )

    detail: str = Field(..., description="Human-readable reason the request was rejected")
    reason: Optional[str] = Field(None, description="Additional context, present on some 403 responses")


class RateLimitErrorResponse(BaseModel):
    """Error body returned by the request-rate and location-diversity limiter (429)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detail": "Rate limit exceeded",
                    "reason": "Too many requests (401 > 400) in 1 hour(s)",
                    "retry_after": 3600,
                }
            ]
        },
    )

    detail: str = Field(..., description="Which limit was exceeded")
    reason: str = Field(..., description="Details of the limit that was hit")
    retry_after: int = Field(..., description="Seconds to wait before retrying (also sent as the Retry-After header)")


class SimpleErrorResponse(BaseModel):
    """Bare ``{"error": ...}`` body returned by /weather and /forecast when no data could be produced."""

    model_config = ConfigDict(json_schema_extra={"examples": [{"error": "No temperature data available"}]})

    error: str = Field(..., description="Description of what went wrong")
