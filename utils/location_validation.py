"""Location validation utilities."""
import json
import logging
from typing import Optional, Tuple
from datetime import datetime, timezone
import redis

logger = logging.getLogger(__name__)


class InvalidLocationCache:
    """Cache for invalid locations to avoid repeated API calls."""
    
    def __init__(self, redis_client: redis.Redis, ttl_hours: int = 24):
        self.redis_client = redis_client
        self.ttl_seconds = ttl_hours * 3600
        self.invalid_key_prefix = "invalid_location:"
    
    def is_invalid_location(self, location: str) -> bool:
        """Check if a location is known to be invalid."""
        if not self.redis_client:
            return False
        try:
            key = f"{self.invalid_key_prefix}{location.lower()}"
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking invalid location cache: {e}")
            return False
    
    def mark_location_invalid(self, location: str, reason: str = "no_data"):
        """Mark a location as invalid with a reason."""
        if not self.redis_client:
            return
        try:
            key = f"{self.invalid_key_prefix}{location.lower()}"
            data = {
                "location": location,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.redis_client.setex(key, self.ttl_seconds, json.dumps(data))
            logger.info(f"Marked location as invalid: {location} (reason: {reason})")
        except Exception as e:
            logger.error(f"Error marking location as invalid: {e}")
    
    def get_invalid_location_info(self, location: str) -> Optional[dict]:
        """Get information about why a location was marked as invalid."""
        if not self.redis_client:
            return None
        try:
            key = f"{self.invalid_key_prefix}{location.lower()}"
            data = self.redis_client.get(key)
            if data:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting invalid location info: {e}")
        return None


def validate_location_response(data: dict, location: str) -> Tuple[bool, str]:
    """
    Validate that the response contains meaningful data.
    Returns (is_valid, error_message).
    """
    # Check if values array is empty
    if not data.get("values") or len(data["values"]) == 0:
        return False, f"No temperature data found for location '{location}'. The location may be invalid or not supported by the weather service."
    
    # Check if all temperature values are None or 0
    values = data.get("values", [])
    valid_temps = [v for v in values if v.get("temperature") is not None and v.get("temperature") != 0]
    if not valid_temps:
        return False, f"No valid temperature data found for location '{location}'. The location may be invalid or not supported by the weather service."
    
    # Check if data points count is 0
    if data.get("average", {}).get("data_points", 0) == 0:
        return False, f"No data points available for location '{location}'. The location may be invalid or not supported by the weather service."
    
    return True, ""


def is_location_likely_invalid(location: str) -> bool:
    """
    Check if a location string looks obviously invalid.
    This is a quick check before making API calls.
    Enhanced with security checks for SSRF prevention.
    """
    if not location or not isinstance(location, str):
        return True
    
    # Check for common invalid patterns
    invalid_patterns = [
        "[object Object]",
        "undefined",
        "null",
        "NaN",
        "object Object",
        "Object",
        "[object",
        "object]"
    ]
    
    location_lower = location.lower().strip()
    if any(pattern.lower() in location_lower for pattern in invalid_patterns):
        return True
    
    # Security checks for SSRF prevention
    # Length check
    if len(location) > 200:
        return True
    
    # Check for dangerous patterns that could indicate SSRF attempts
    dangerous_patterns = [
        '://',      # URL scheme
        '@',        # URL auth
        'localhost',
        '127.0.0.1',
        '0.0.0.0',
        '..',       # Path traversal
        '/',        # Path separator
        '\\',       # Windows path separator
        '//',       # Protocol-relative
    ]
    
    for pattern in dangerous_patterns:
        if pattern in location_lower:
            return True
    
    # Check for private IP ranges
    private_ip_patterns = ['10.', '172.16', '172.17', '172.18', '172.19', '172.20',
                          '172.21', '172.22', '172.23', '172.24', '172.25', '172.26',
                          '172.27', '172.28', '172.29', '172.30', '172.31', '192.168']
    
    for pattern in private_ip_patterns:
        if pattern in location_lower:
            return True
    
    return False
