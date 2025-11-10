"""IP address utility functions."""
from fastapi import Request
from config import IP_WHITELIST, IP_BLACKLIST


def get_client_ip(request: Request) -> str:
    """Get the client IP address from the request."""
    # Check for forwarded headers first (for proxy/load balancer scenarios)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"


def is_ip_whitelisted(ip: str) -> bool:
    """Check if an IP address is whitelisted (exempt from rate limiting)."""
    return ip in IP_WHITELIST


def is_ip_blacklisted(ip: str) -> bool:
    """Check if an IP address is blacklisted (blocked entirely)."""
    return ip in IP_BLACKLIST
