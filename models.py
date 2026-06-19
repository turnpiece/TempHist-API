"""Pydantic models for API requests and responses."""

from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


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

    period: Literal["daily", "weekly", "monthly", "yearly"] = Field(..., description="Data period")
    location: str = Field(..., description="Location name")
    identifier: str = Field(..., description="Date identifier (MM-DD for daily, YYYY-MM for monthly, etc.)")
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


class MetaResponse(BaseModel):
    """Response model for the /meta sub-resource endpoint."""

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
