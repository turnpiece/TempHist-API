"""Analytics endpoints."""
import json
import logging
import redis
from datetime import datetime
from fastapi import APIRouter, Request, Query, HTTPException
from fastapi.responses import JSONResponse
from models import AnalyticsData, AnalyticsResponse
from utils.ip_utils import get_client_ip
from config import ANALYTICS_RATE_LIMIT, DEBUG

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analytics", response_model=AnalyticsResponse)
async def submit_analytics(
    request: Request,
    redis_client: redis.Redis,
    analytics_storage
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
        
        # Validate data using Pydantic model
        try:
            analytics_data = AnalyticsData(**json_data)
        except Exception as e:
            logger.error(f"❌ ANALYTICS VALIDATION ERROR: {e} | IP={client_ip} | Data: {json_data}")
            
            # Provide detailed validation error information
            if hasattr(e, 'errors'):
                error_details = []
                for error in e.errors():
                    field = " -> ".join(str(loc) for loc in error['loc'])
                    error_details.append(f"{field}: {error['msg']}")
                error_message = f"Validation failed: {'; '.join(error_details)}"
            else:
                error_message = f"Validation failed: {str(e)}"
            
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
        
        return AnalyticsResponse(
            status="success",
            message="Analytics data submitted successfully",
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
async def get_analytics_summary(analytics_storage):
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
    analytics_storage=None
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
    analytics_storage=None
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
