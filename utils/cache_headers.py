"""Cache header utilities."""
import hashlib
from datetime import datetime, timezone, date as dt_date, timedelta
from fastapi import Response


def set_weather_cache_headers(response: Response, *, req_date: dt_date, key_parts: str):
    """Set appropriate cache headers based on the age of the weather data."""
    # Strong long-term cache for any day before (UTC) today - 2 days
    today_utc = datetime.now(timezone.utc).date()
    if req_date <= today_utc - timedelta(days=2):
        # Historical data - very unlikely to change
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, s-maxage=31536000, immutable, stale-if-error=604800"
        )
    else:
        # Safer policy for the last ~48h (recent data might be revised)
        response.headers["Cache-Control"] = (
            "public, max-age=21600, stale-while-revalidate=86400, stale-if-error=86400"
        )

    # Deterministic weak ETag: location|date|unit_group|schema_version
    etag = hashlib.sha256(key_parts.encode("utf-8")).hexdigest()[:32]  # Use 32 chars (128-bit security)
    response.headers["ETag"] = f'W/"{etag}"'
    
    # Use the requested calendar day as Last-Modified (UTC midnight)
    response.headers["Last-Modified"] = f"{req_date.isoformat()}T00:00:00Z"
