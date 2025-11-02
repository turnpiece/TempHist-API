"""Sanitization utilities for logging and security."""
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import re


def sanitize_url(url: str) -> str:
    """Remove credentials and API keys from URL for logging to prevent sensitive data exposure."""
    try:
        parsed = urlparse(url)
        
        # Redact password from netloc
        if parsed.password:
            username = parsed.username or ""
            netloc = f"{username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            parsed = parsed._replace(netloc=netloc)
        
        # Redact API keys from query parameters
        if parsed.query:
            query_params = parse_qs(parsed.query, keep_blank_values=True)
            # Redact sensitive parameters
            sensitive_params = ['key', 'api_key', 'apikey', 'token', 'password', 'secret']
            for param in sensitive_params:
                if param in query_params:
                    query_params[param] = ['[REDACTED]']
            # Reconstruct query string
            sanitized_query = urlencode(query_params, doseq=True)
            parsed = parsed._replace(query=sanitized_query)
        
        return urlunparse(parsed)
    except Exception:
        # If parsing fails, return a safe placeholder
        return "[REDACTED_URL]"


def sanitize_for_logging(data: str, max_length: int = 100) -> str:
    """Sanitize user input data before logging to prevent log injection and sensitive data exposure (MED-005).
    
    Args:
        data: The input string to sanitize
        max_length: Maximum length to keep (default 100 chars)
        
    Returns:
        Sanitized string safe for logging
    """
    if not data or not isinstance(data, str):
        return str(data)[:max_length] if data else ""
    
    # Truncate to max length
    if len(data) > max_length:
        data = data[:max_length] + "..."
    
    # Remove or replace control characters that could be used for log injection
    # Replace newlines, tabs, and other control chars with spaces
    data = re.sub(r'[\r\n\t\x00-\x1f\x7f-\x9f]', ' ', data)
    
    # Remove potential sensitive patterns
    # Redact bearer tokens
    data = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', data, flags=re.IGNORECASE)
    # Redact API keys in URLs
    data = re.sub(r'key=[^&\s]+', 'key=[REDACTED]', data, flags=re.IGNORECASE)
    data = re.sub(r'api_key=[^&\s]+', 'api_key=[REDACTED]', data, flags=re.IGNORECASE)
    # Redact tokens
    data = re.sub(r'token=[^&\s]+', 'token=[REDACTED]', data, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    data = re.sub(r'\s+', ' ', data).strip()
    
    return data
