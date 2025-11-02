"""Analytics storage and management."""
import json
import time
import logging
from datetime import datetime
from typing import List, Dict
from fastapi import HTTPException
from models import AnalyticsData
import redis
from config import DEBUG

logger = logging.getLogger(__name__)


class AnalyticsStorage:
    """Store and manage client analytics data."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.analytics_prefix = "analytics_"
        self.retention_seconds = 7 * 24 * 3600  # 7 days retention
        self.max_errors_per_session = 50  # Limit errors per session
    
    def store_analytics(self, analytics_data: AnalyticsData, client_ip: str) -> str:
        """Store analytics data and return unique ID."""
        analytics_id = f"analytics_{int(time.time() * 1000)}_{hash(client_ip) % 10000}"
        
        # Prepare data for storage
        analytics_record = {
            "id": analytics_id,
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_ip,
            "session_duration": analytics_data.session_duration,
            "api_calls": analytics_data.api_calls,
            "api_failure_rate": analytics_data.api_failure_rate,
            "retry_attempts": analytics_data.retry_attempts,
            "location_failures": analytics_data.location_failures,
            "error_count": analytics_data.error_count,
            "recent_errors": [error.model_dump() for error in analytics_data.recent_errors[:self.max_errors_per_session]],
            "app_version": analytics_data.app_version,
            "platform": analytics_data.platform,
            "user_agent": analytics_data.user_agent,
            "session_id": analytics_data.session_id
        }
        
        try:
            # Store in Redis with expiration
            self.redis.setex(
                f"{self.analytics_prefix}{analytics_id}",
                self.retention_seconds,
                json.dumps(analytics_record)
            )
            
            # Add to analytics index for easy retrieval
            self.redis.lpush("analytics_index", analytics_id)
            self.redis.expire("analytics_index", self.retention_seconds)
            
            # Update analytics summary stats
            self._update_analytics_summary(analytics_record)
            
            if DEBUG:
                logger.debug(f"📊 ANALYTICS STORED: {analytics_id} | Errors: {analytics_data.error_count} | Duration: {analytics_data.session_duration}s")
            
            return analytics_id
            
        except Exception as e:
            logger.error(f"Failed to store analytics data: {e}")
            raise HTTPException(status_code=500, detail="Failed to store analytics data")
    
    def _update_analytics_summary(self, analytics_record: dict):
        """Update analytics summary statistics."""
        try:
            # Get current summary
            summary_key = "analytics_summary"
            summary = self.redis.get(summary_key)
            if summary:
                summary_data = json.loads(summary)
            else:
                summary_data = {
                    "total_sessions": 0,
                    "total_api_calls": 0,
                    "total_errors": 0,
                    "avg_session_duration": 0,
                    "avg_failure_rate": 0,
                    "platforms": {},
                    "error_types": {},
                    "last_updated": datetime.now().isoformat()
                }
            
            # Update counters
            summary_data["total_sessions"] += 1
            summary_data["total_api_calls"] += analytics_record["api_calls"]
            summary_data["total_errors"] += analytics_record["error_count"]
            
            # Update averages
            total_sessions = summary_data["total_sessions"]
            summary_data["avg_session_duration"] = (
                (summary_data["avg_session_duration"] * (total_sessions - 1) + analytics_record["session_duration"]) / total_sessions
            )
            
            # Update platform stats
            platform = analytics_record.get("platform", "unknown")
            summary_data["platforms"][platform] = summary_data["platforms"].get(platform, 0) + 1
            
            # Update error type stats
            for error in analytics_record["recent_errors"]:
                error_type = error.get("error_type", "unknown")
                summary_data["error_types"][error_type] = summary_data["error_types"].get(error_type, 0) + 1
            
            # Store updated summary
            self.redis.setex(summary_key, self.retention_seconds, json.dumps(summary_data))
            
        except Exception as e:
            logger.error(f"Failed to update analytics summary: {e}")
    
    def get_analytics_summary(self) -> dict:
        """Get analytics summary statistics."""
        try:
            summary_key = "analytics_summary"
            summary = self.redis.get(summary_key)
            if summary:
                return json.loads(summary)
            else:
                return {
                    "total_sessions": 0,
                    "total_api_calls": 0,
                    "total_errors": 0,
                    "avg_session_duration": 0,
                    "avg_failure_rate": 0,
                    "platforms": {},
                    "error_types": {},
                    "last_updated": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {"error": "Failed to retrieve analytics summary"}
    
    def get_recent_analytics(self, limit: int = 100) -> List[dict]:
        """Get recent analytics records."""
        try:
            # Get recent analytics IDs
            analytics_ids = self.redis.lrange("analytics_index", 0, limit - 1)
            analytics_records = []
            
            for analytics_id in analytics_ids:
                analytics_id = analytics_id.decode('utf-8') if isinstance(analytics_id, bytes) else analytics_id
                record = self.redis.get(f"{self.analytics_prefix}{analytics_id}")
                if record:
                    record_str = record.decode('utf-8') if isinstance(record, bytes) else record
                    analytics_records.append(json.loads(record_str))
            
            return analytics_records
            
        except Exception as e:
            logger.error(f"Failed to get recent analytics: {e}")
            return []
    
    def get_analytics_by_session(self, session_id: str) -> List[dict]:
        """Get analytics records for a specific session."""
        try:
            # This would require a more sophisticated indexing system
            # For now, we'll search through recent analytics
            recent_analytics = self.get_recent_analytics(1000)  # Get more records
            session_analytics = [
                record for record in recent_analytics 
                if record.get("session_id") == session_id
            ]
            return session_analytics
            
        except Exception as e:
            logger.error(f"Failed to get analytics by session: {e}")
            return []
