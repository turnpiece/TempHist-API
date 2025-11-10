"""Rate limiting classes and monitoring."""
import time
import logging
from typing import Dict, Set, Tuple
from collections import defaultdict
import redis
from config import (
    SERVICE_TOKEN_RATE_LIMITS,
    MAX_LOCATIONS_PER_HOUR,
    MAX_REQUESTS_PER_HOUR,
    RATE_LIMIT_WINDOW_HOURS,
    RATE_LIMIT_ENABLED,
    IP_WHITELIST,
    IP_BLACKLIST,
    DEBUG
)

logger = logging.getLogger(__name__)


class ServiceTokenRateLimiter:
    """Redis-based rate limiter for service tokens to prevent abuse while allowing legitimate cache warming.
    
    Uses Redis for distributed rate limiting across multiple worker instances.
    """
    
    def __init__(self, redis_client: redis.Redis, 
                 requests_per_hour: int = SERVICE_TOKEN_RATE_LIMITS["requests_per_hour"],
                 locations_per_hour: int = SERVICE_TOKEN_RATE_LIMITS["locations_per_hour"],
                 window_hours: int = SERVICE_TOKEN_RATE_LIMITS["window_hours"]):
        self.redis = redis_client
        self.requests_per_hour = requests_per_hour
        self.locations_per_hour = locations_per_hour
        self.window_seconds = window_hours * 3600
        self.requests_key_prefix = "service_rate:requests:"
        self.locations_key_prefix = "service_rate:locations:"
    
    def check_request_rate(self, client_ip: str) -> Tuple[bool, str]:
        """Check if service token request rate is within limits.
        
        Returns:
            tuple: (is_allowed, reason)
        """
        try:
            key = f"{self.requests_key_prefix}{client_ip}"
            now = time.time()
            window_start = now - self.window_seconds
            
            # Use Redis sorted set for sliding window
            # Remove old entries
            self.redis.zremrangebyscore(key, 0, window_start)
            
            # Count requests in window
            count = self.redis.zcard(key)
            
            if count >= self.requests_per_hour:
                return False, f"Service token rate limit exceeded: {count}/{self.requests_per_hour} requests per hour"
            
            # Add current request with current timestamp
            self.redis.zadd(key, {str(now): now})
            self.redis.expire(key, self.window_seconds)
            
            return True, "OK"
        except Exception as e:
            logger.error(f"Error checking service token request rate: {e}")
            # Fail open - allow request if Redis fails (to prevent DoS from Redis issues)
            return True, "OK"
    
    def check_location_diversity(self, client_ip: str, location: str) -> Tuple[bool, str]:
        """Check if service token location diversity is within limits.
        
        Returns:
            tuple: (is_allowed, reason)
        """
        try:
            key = f"{self.locations_key_prefix}{client_ip}"
            now = time.time()
            window_start = now - self.window_seconds
            
            # Use Redis sorted set for sliding window
            # Remove old entries
            self.redis.zremrangebyscore(key, 0, window_start)
            
            # Check if location already in window
            location_exists = self.redis.zscore(key, location) is not None
            
            if not location_exists:
                # Count unique locations in window
                count = self.redis.zcard(key)
                
                if count >= self.locations_per_hour:
                    return False, f"Service token location limit exceeded: {count}/{self.locations_per_hour} unique locations per hour"
            
            # Add/update location with current timestamp
            self.redis.zadd(key, {location: now})
            self.redis.expire(key, self.window_seconds)
            
            return True, "OK"
        except Exception as e:
            logger.error(f"Error checking service token location diversity: {e}")
            # Fail open - allow request if Redis fails
            return True, "OK"
    
    def get_stats(self, client_ip: str) -> Dict:
        """Get service token rate limiting stats."""
        try:
            requests_key = f"{self.requests_key_prefix}{client_ip}"
            locations_key = f"{self.locations_key_prefix}{client_ip}"
            now = time.time()
            window_start = now - self.window_seconds
            
            # Count requests
            self.redis.zremrangebyscore(requests_key, 0, window_start)
            request_count = self.redis.zcard(requests_key)
            
            # Count locations
            self.redis.zremrangebyscore(locations_key, 0, window_start)
            location_count = self.redis.zcard(locations_key)
            
            return {
                "requests_count": request_count,
                "requests_limit": self.requests_per_hour,
                "locations_count": location_count,
                "locations_limit": self.locations_per_hour,
                "window_hours": self.window_seconds / 3600,
                "remaining_requests": max(0, self.requests_per_hour - request_count),
                "remaining_locations": max(0, self.locations_per_hour - location_count)
            }
        except Exception as e:
            logger.error(f"Error getting service token stats: {e}")
            return {
                "error": str(e),
                "requests_count": 0,
                "requests_limit": self.requests_per_hour,
                "locations_count": 0,
                "locations_limit": self.locations_per_hour
            }


class LocationDiversityMonitor:
    """Monitor and limit location diversity per IP address to prevent API abuse."""
    
    def __init__(self, max_locations: int = 10, window_hours: int = 1):
        self.max_locations = max_locations
        self.window_hours = window_hours
        self.window_seconds = window_hours * 3600
        
        # Track unique locations per IP over time windows
        self.ip_locations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.suspicious_ips: Set[str] = set()
        self.last_cleanup = time.time()
        # LOW-002: Use background task for cleanup instead of on every request
        self.cleanup_interval = 300  # Clean up every 5 minutes
        self._cleanup_task = None  # Background cleanup task
        
    def _cleanup_old_entries(self):
        """Clean up old entries to prevent memory bloat."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        window_start = current_time - self.window_seconds
        
        for ip_addr in list(self.ip_locations.keys()):
            for timestamp in list(self.ip_locations[ip_addr].keys()):
                if float(timestamp) < window_start:
                    del self.ip_locations[ip_addr][timestamp]
            
            # Remove IP if no timestamps remain
            if not self.ip_locations[ip_addr]:
                del self.ip_locations[ip_addr]
        
        self.last_cleanup = current_time
    
    def check_location_diversity(self, ip: str, location: str) -> Tuple[bool, str]:
        """Check if IP is requesting too many different locations.
        
        Returns:
            tuple: (is_allowed, reason)
        """
        self._cleanup_old_entries()
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Add current request
        timestamp_str = str(current_time)
        self.ip_locations[ip][timestamp_str].add(location)
        
        # Count unique locations in time window
        unique_locations = set()
        for timestamp, locations in self.ip_locations[ip].items():
            if float(timestamp) >= window_start:
                unique_locations.update(locations)
        
        # Check if suspicious
        if len(unique_locations) > self.max_locations:
            self.suspicious_ips.add(ip)
            return False, f"Too many different locations ({len(unique_locations)} > {self.max_locations}) in {self.window_hours} hour(s)"
        
        return True, "OK"
    
    def is_suspicious(self, ip: str) -> bool:
        """Check if an IP has been flagged as suspicious."""
        return ip in self.suspicious_ips
    
    def get_stats(self, ip: str) -> Dict:
        """Get rate limiting stats for an IP address."""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        unique_locations = set()
        total_requests = 0
        
        for timestamp, locations in self.ip_locations[ip].items():
            if float(timestamp) >= window_start:
                unique_locations.update(locations)
                total_requests += len(locations)
        
        return {
            "unique_locations": len(unique_locations),
            "total_requests": total_requests,
            "max_locations": self.max_locations,
            "window_hours": self.window_hours,
            "is_suspicious": self.is_suspicious(ip)
        }


class RequestRateMonitor:
    """Monitor and limit total request rate per IP address."""
    
    def __init__(self, max_requests: int = 100, window_hours: int = 1):
        self.max_requests = max_requests
        self.window_hours = window_hours
        self.window_seconds = window_hours * 3600
        
        # Track request counts per IP over time windows
        self.ip_requests: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.last_cleanup = time.time()
        # LOW-002: Use background task for cleanup instead of on every request
        self.cleanup_interval = 300  # Clean up every 5 minutes
        self._cleanup_task = None  # Background cleanup task
        
    def _cleanup_old_entries(self):
        """Clean up old entries to prevent memory bloat."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        window_start = current_time - self.window_seconds
        
        for ip_addr in list(self.ip_requests.keys()):
            for timestamp in list(self.ip_requests[ip_addr].keys()):
                if float(timestamp) < window_start:
                    del self.ip_requests[ip_addr][timestamp]
            
            # Remove IP if no timestamps remain
            if not self.ip_requests[ip_addr]:
                del self.ip_requests[ip_addr]
        
        self.last_cleanup = current_time
    
    def check_request_rate(self, ip: str) -> Tuple[bool, str]:
        """Check if IP is making too many requests.
        
        Returns:
            tuple: (is_allowed, reason)
        """
        self._cleanup_old_entries()
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Add current request
        timestamp_str = str(int(current_time / 60) * 60)  # Round to minute for better grouping
        self.ip_requests[ip][timestamp_str] += 1
        
        # Count total requests in time window
        total_requests = sum(
            count for timestamp, count in self.ip_requests[ip].items()
            if float(timestamp) >= window_start
        )
        
        # Check if rate limit exceeded
        if total_requests > self.max_requests:
            return False, f"Too many requests ({total_requests} > {self.max_requests}) in {self.window_hours} hour(s)"
        
        return True, "OK"
    
    def get_stats(self, ip: str) -> Dict:
        """Get rate limiting stats for an IP address."""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        total_requests = sum(
            count for timestamp, count in self.ip_requests[ip].items()
            if float(timestamp) >= window_start
        )
        
        return {
            "total_requests": total_requests,
            "max_requests": self.max_requests,
            "window_hours": self.window_hours,
            "remaining_requests": max(0, self.max_requests - total_requests)
        }


# Initialize rate limiting monitors
def initialize_rate_limiting():
    """Initialize rate limiting monitors."""
    if RATE_LIMIT_ENABLED:
        location_monitor = LocationDiversityMonitor(MAX_LOCATIONS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS)
        request_monitor = RequestRateMonitor(MAX_REQUESTS_PER_HOUR, RATE_LIMIT_WINDOW_HOURS)
        
        if DEBUG:
            logger.info(f"🛡️  RATE LIMITING INITIALIZED: {MAX_LOCATIONS_PER_HOUR} locations/hour, {MAX_REQUESTS_PER_HOUR} requests/hour, {RATE_LIMIT_WINDOW_HOURS}h window")
            if IP_WHITELIST:
                logger.debug(f"⭐ WHITELISTED IPS: {', '.join(IP_WHITELIST)}")
            if IP_BLACKLIST:
                logger.debug(f"🚫 BLACKLISTED IPS: {', '.join(IP_BLACKLIST)}")
        else:
            logger.debug(f"Rate limiting enabled: max {MAX_LOCATIONS_PER_HOUR} locations, max {MAX_REQUESTS_PER_HOUR} requests per {RATE_LIMIT_WINDOW_HOURS} hour(s)")
            if IP_WHITELIST:
                logger.debug(f"Whitelisted IPs: {len(IP_WHITELIST)} configured")
            if IP_BLACKLIST:
                logger.debug(f"Blacklisted IPs: {len(IP_BLACKLIST)} configured")
        
        return location_monitor, request_monitor
    else:
        if DEBUG:
            logger.info("⚠️  RATE LIMITING DISABLED")
        else:
            logger.info("Rate limiting disabled")
        return None, None
