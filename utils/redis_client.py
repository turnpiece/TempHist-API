"""Redis client creation and management."""
import os
import logging
import ssl
from urllib.parse import urlparse
import redis
from config import ENVIRONMENT, DEBUG

logger = logging.getLogger(__name__)


def create_redis_client(url: str) -> redis.Redis:
    """Create Redis client with security validation."""
    parsed = urlparse(url)
    env = ENVIRONMENT
    
    # Enforce password in production
    if env == "production" and not parsed.password:
        logger.error("❌ Redis password required in production")
        raise ValueError("Redis password required in production environment")
    
    # Enforce SSL in production
    ssl_context = None
    if parsed.scheme == "rediss":
        ssl_context = ssl.create_default_context()
    elif env == "production":
        logger.warning("⚠️  Redis not using SSL (rediss://) in production! Consider using rediss:// for encrypted connections.")
        # In production, warn but don't fail (some providers handle SSL at network level)
    
    # Create Redis client with SSL if needed
    # Note: from_url handles SSL automatically when using rediss:// scheme
    client = redis.from_url(
        url,
        decode_responses=True
    )
    
    # Test connection
    try:
        client.ping()
        if DEBUG:
            logger.info("✅ Redis connection validated successfully")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        raise
    
    return client
