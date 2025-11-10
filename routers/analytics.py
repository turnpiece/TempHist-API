"""Analytics endpoints."""
import json
import logging
import redis
from datetime import datetime
from typing import Any, Dict, List, Tuple
from fastapi import APIRouter, Request, Query, HTTPException, Depends
from models import AnalyticsData, AnalyticsResponse
from utils.ip_utils import get_client_ip
from config import ANALYTICS_RATE_LIMIT, DEBUG
from routers.dependencies import get_redis_client, get_analytics_storage
from analytics_storage import AnalyticsStorage

logger = logging.getLogger(__name__)

router = APIRouter()


INT_FIELDS_DEFAULTS: Tuple[Tuple[str, int], ...] = (
    ("session_duration", 0),
    ("api_calls", 0),
    ("retry_attempts", 0),
    ("location_failures", 0),
    ("error_count", 0),
)


def _coerce_non_negative_int(value: Any, field: str, default: int, warnings: List[str]) -> int:
    """Best-effort coercion of analytics numeric fields to non-negative ints."""
    if value is None:
        return default
    if isinstance(value, bool):
        coerced = int(value)
    elif isinstance(value, (int, float)):
        coerced = int(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        cleaned = cleaned.replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            coerced = int(float(cleaned))
        except ValueError:
            warnings.append(f"{field}: could not parse '{value}', defaulted to {default}")
            return default
    else:
        warnings.append(f"{field}: unsupported type {type(value).__name__}, defaulted to {default}")
        return default
    if coerced < 0:
        warnings.append(f"{field}: negative value {coerced} reset to {default}")
        return default
    return coerced


def _normalize_failure_rate(value: Any, warnings: List[str]) -> str:
    """Normalize API failure rate into a percentage string."""
    default = "0%"
    if value is None:
        return default
    if isinstance(value, (int, float)):
        numeric = max(float(value), 0.0)
        return f"{numeric:.1f}%" if numeric % 1 else f"{int(numeric)}%"
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return default
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        cleaned = cleaned.replace(",", "")
        try:
            numeric = max(float(cleaned), 0.0)
            return f"{numeric:.1f}%" if numeric % 1 else f"{int(numeric)}%"
        except ValueError:
            warnings.append(f"api_failure_rate: could not parse '{value}', defaulted to {default}")
            return default
    warnings.append(f"api_failure_rate: unsupported type {type(value).__name__}, defaulted to {default}")
    return default


def _sanitize_recent_errors(raw_errors: Any, warnings: List[str]) -> List[Dict[str, Any]]:
    """Return only well-formed recent error records."""
    if raw_errors is None:
        return []
    if not isinstance(raw_errors, list):
        warnings.append("recent_errors: expected list, defaulted to empty list")
        return []
    sanitized: List[Dict[str, Any]] = []
    required_keys = {"timestamp", "error_type", "message"}
    for idx, item in enumerate(raw_errors):
        if not isinstance(item, dict):
            warnings.append(f"recent_errors[{idx}]: skipped non-dict entry")
            continue
        if not required_keys.issubset(item.keys()):
            missing = required_keys.difference(item.keys())
            warnings.append(f"recent_errors[{idx}]: missing keys {sorted(missing)}, entry skipped")
            continue
        sanitized.append(item)
    return sanitized


def _sanitize_analytics_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Sanitize raw analytics payload to prevent validation failures."""
    sanitized: Dict[str, Any] = {}
    warnings: List[str] = []
    data = payload or {}

    for field_name, default in INT_FIELDS_DEFAULTS:
        sanitized[field_name] = _coerce_non_negative_int(data.get(field_name), field_name, default, warnings)

    sanitized["api_failure_rate"] = _normalize_failure_rate(data.get("api_failure_rate"), warnings)
    sanitized["recent_errors"] = _sanitize_recent_errors(data.get("recent_errors"), warnings)

    # Optional string metadata fields
    for optional_field in ("app_version", "platform", "user_agent", "session_id"):
        value = data.get(optional_field)
        if value is None:
            sanitized[optional_field] = None
        else:
            sanitized[optional_field] = str(value)

    # Preserve any additional metadata fields not explicitly handled
    extra_fields = set(data.keys()) - {name for name, _ in INT_FIELDS_DEFAULTS} - {
        "api_failure_rate",
        "recent_errors",
        "app_version",
        "platform",
        "user_agent",
        "session_id",
    }
    for extra_field in extra_fields:
        sanitized[extra_field] = data.get(extra_field)

    return sanitized, warnings


@router.post("/analytics", response_model=AnalyticsResponse)
async def submit_analytics(
    request: Request,
    redis_client: redis.Redis = Depends(get_redis_client),
    analytics_storage: AnalyticsStorage = Depends(get_analytics_storage)
):
    """Submit client analytics data for monitoring and error tracking (MED-007: Rate limited)."""
    client_ip = get_client_ip(request)
    
    # MED-007: Add rate limiting for analytics endpoint to prevent spam/DoS
    # Skip rate limiting in test environment or if limit is very high (indicates test mode)
    if ANALYTICS_RATE_LIMIT < 10000:  # Normal production limit
        analytics_key = f"analytics_limit:{client_ip}"
        try:
            current_count = redis_client.get(analytics_key)
            if current_count:
                current_count = int(current_count) if isinstance(current_count, (str, bytes)) else current_count
                if current_count >= ANALYTICS_RATE_LIMIT:
                    logger.warning(f"⚠️  ANALYTICS RATE LIMIT EXCEEDED: {client_ip} ({current_count}/{ANALYTICS_RATE_LIMIT})")
                    raise HTTPException(
                        status_code=429,
                        detail=f"Analytics submission rate limit exceeded ({ANALYTICS_RATE_LIMIT} per hour). Please try again later.",
                        headers={"Retry-After": "3600"}
                    )
            # Increment counter
            redis_client.incr(analytics_key)
            redis_client.expire(analytics_key, 3600)  # 1 hour window
        except redis.exceptions.RedisError as e:
            # Fail open if Redis unavailable (don't block analytics)
            logger.warning(f"⚠️  Analytics rate limiting unavailable (Redis error): {e}")
    elif DEBUG:
        logger.debug(f"📊 Analytics rate limiting bypassed (test mode): limit={ANALYTICS_RATE_LIMIT}")
    
    try:
        # Log request details for debugging
        content_type = request.headers.get("content-type", "")
        content_length = request.headers.get("content-length", "unknown")
        
        logger.info(f"📊 ANALYTICS REQUEST: IP={client_ip} | Content-Type={content_type} | Length={content_length}")
        
        # Validate content type
        if not content_type.startswith("application/json"):
            logger.warning(f"⚠️  ANALYTICS INVALID CONTENT-TYPE: {content_type} | IP={client_ip}")
            raise HTTPException(
                status_code=415, 
                detail="Content-Type must be application/json"
            )
        
        # Check content length (limit to 1MB for analytics data)
        max_content_length = 1024 * 1024  # 1MB
        try:
            content_length_int = int(content_length) if content_length != "unknown" else None
            if content_length_int and content_length_int > max_content_length:
                logger.warning(f"⚠️  ANALYTICS REQUEST TOO LARGE: {content_length_int} bytes | IP={client_ip}")
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Maximum size is {max_content_length} bytes"
                )
        except ValueError:
            # If we can't parse content-length, continue but log it
            logger.warning(f"⚠️  ANALYTICS INVALID CONTENT-LENGTH: {content_length} | IP={client_ip}")
        
        # Read and parse request body
        try:
            body = await request.body()
            if not body:
                logger.warning(f"⚠️  ANALYTICS EMPTY BODY: IP={client_ip}")
                raise HTTPException(
                    status_code=400,
                    detail="Request body cannot be empty"
                )
            
            # Check actual body size
            if len(body) > max_content_length:
                logger.warning(f"⚠️  ANALYTICS BODY TOO LARGE: {len(body)} bytes | IP={client_ip}")
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body too large. Maximum size is {max_content_length} bytes"
                )
            
            # Log request body for debugging (truncated for security)
            body_str = body.decode('utf-8')
            body_preview = body_str[:500] + "..." if len(body_str) > 500 else body_str
            logger.info(f"📊 ANALYTICS BODY PREVIEW: {body_preview}")
            
            # Parse JSON
            try:
                json_data = json.loads(body_str)
            except json.JSONDecodeError as e:
                logger.error(f"❌ ANALYTICS JSON PARSE ERROR: {e} | IP={client_ip} | Body: {body_preview}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON format: {str(e)}"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ ANALYTICS BODY READ ERROR: {e} | IP={client_ip}")
            raise HTTPException(
                status_code=400,
                detail="Failed to read request body"
            )
        
        # Sanitize analytics payload before validation
        sanitized_payload, sanitization_warnings = _sanitize_analytics_payload(json_data)

        # Validate data using Pydantic model
        try:
            analytics_data = AnalyticsData(**sanitized_payload)
        except Exception as e:
            logger.error(f"❌ ANALYTICS VALIDATION ERROR: {e} | IP={client_ip} | Data: {sanitized_payload}")
            
            # Provide detailed validation error information
            if hasattr(e, 'errors'):
                error_details = []
                for error in e.errors():
                    field = " -> ".join(str(loc) for loc in error['loc'])
                    error_details.append(f"{field}: {error['msg']}")
                error_message = f"Validation failed after sanitization: {'; '.join(error_details)}"
            else:
                error_message = f"Validation failed after sanitization: {str(e)}"
            
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Validation Error",
                    "message": error_message,
                    "details": str(e) if hasattr(e, 'errors') else None
                }
            )
        
        # Store analytics data
        try:
            analytics_id = analytics_storage.store_analytics(analytics_data, client_ip)
            logger.info(f"📊 ANALYTICS STORED: {analytics_id} | IP={client_ip}")
        except Exception as e:
            logger.error(f"❌ ANALYTICS STORAGE ERROR: {e} | IP={client_ip}")
            raise HTTPException(
                status_code=500,
                detail="Failed to store analytics data"
            )
        
        if sanitization_warnings:
            logger.warning(
                f"⚠️  ANALYTICS SANITIZATION WARNINGS: {sanitization_warnings} | IP={client_ip}"
            )

        response_message = "Analytics data submitted successfully"
        if sanitization_warnings:
            response_message += f" (sanitized {len(sanitization_warnings)} field(s))"

        return AnalyticsResponse(
            status="success",
            message=response_message,
            analytics_id=analytics_id,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper status codes)
        raise
    except Exception as e:
        logger.error(f"❌ ANALYTICS UNEXPECTED ERROR: {e} | IP={client_ip}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while processing analytics data"
        )


@router.get("/analytics/summary")
async def get_analytics_summary(
    analytics_storage: AnalyticsStorage = Depends(get_analytics_storage)
):
    """Get analytics summary statistics."""
    try:
        summary = analytics_storage.get_analytics_summary()
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics summary")


@router.get("/analytics/recent")
async def get_recent_analytics(
    limit: int = Query(100, ge=1, le=1000),
    analytics_storage: AnalyticsStorage = Depends(get_analytics_storage)
):
    """Get recent analytics records."""
    try:
        analytics_records = analytics_storage.get_recent_analytics(limit)
        return {
            "status": "success",
            "data": analytics_records,
            "count": len(analytics_records),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting recent analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent analytics")


@router.get("/analytics/session/{session_id}")
async def get_analytics_by_session(
    session_id: str,
    analytics_storage: AnalyticsStorage = Depends(get_analytics_storage)
):
    """Get analytics records for a specific session."""
    try:
        session_analytics = analytics_storage.get_analytics_by_session(session_id)
        return {
            "status": "success",
            "data": session_analytics,
            "count": len(session_analytics),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting session analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session analytics")
