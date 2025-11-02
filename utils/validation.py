"""Input validation utilities for security."""
import re
from datetime import datetime
from config import VISUAL_CROSSING_UNIT_GROUP, VISUAL_CROSSING_INCLUDE_PARAMS, VISUAL_CROSSING_REMOTE_DATA
from config import API_KEY, VISUAL_CROSSING_BASE_URL


def clean_location_string(location: str) -> str:
    """Clean location string by removing non-printable ASCII characters."""
    # Remove any non-printable ASCII characters (keep only printable ASCII + common Unicode)
    cleaned = ''.join(char for char in location if char.isprintable() or char in [' ', ','])
    # Remove any multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


def validate_location_for_ssrf(location: str) -> str:
    """Validate location string to prevent SSRF attacks.
    
    Args:
        location: The location string to validate
        
    Returns:
        Validated location string
        
    Raises:
        ValueError: If location is invalid or potentially dangerous
    """
    if not location or not isinstance(location, str):
        raise ValueError("Location must be a non-empty string")
    
    # Length validation
    if len(location) > 200:
        raise ValueError(f"Location string too long (max 200 characters, got {len(location)})")
    
    # Check for null bytes
    if '\x00' in location:
        raise ValueError("Location contains null bytes")
    
    # Check for control characters
    if any(ord(c) < 32 and c not in ['\t', '\n', '\r'] for c in location):
        raise ValueError("Location contains control characters")
    
    # Allow printable characters (letters, numbers, spaces, common punctuation)
    # This includes Unicode letters (accents, non-Latin scripts) which are common in location names
    # Block only control characters and specific dangerous patterns
    if not all(c.isprintable() or c in ['\t', '\n', '\r'] for c in location):
        raise ValueError("Location contains non-printable or control characters")
    
    # Prevent path traversal attempts
    dangerous_patterns = ['..', '/', '\\', '//']
    for pattern in dangerous_patterns:
        if pattern in location:
            raise ValueError(f"Location contains dangerous pattern: {pattern}")
    
    # Prevent SSRF patterns - block URLs, IP addresses, and special schemes
    location_lower = location.lower()
    ssrf_patterns = [
        '://',           # URL scheme
        '@',             # URL auth separator
        'localhost',
        '127.0.0.1',
        '0.0.0.0',
        '169.254',       # Link-local
        '10.',           # Private IP range start
        '172.16',        # Private IP range start
        '172.17',
        '172.18',
        '172.19',
        '172.20',
        '172.21',
        '172.22',
        '172.23',
        '172.24',
        '172.25',
        '172.26',
        '172.27',
        '172.28',
        '172.29',
        '172.30',
        '172.31',
        '192.168',       # Private IP range start
        '[::1]',         # IPv6 localhost
        '[fc00:',        # IPv6 private range
        '[fe80:',        # IPv6 link-local
    ]
    
    for pattern in ssrf_patterns:
        if pattern in location_lower:
            raise ValueError(f"Location contains potentially dangerous SSRF pattern: {pattern}")
    
    # Additional validation: block URL-encoded dangerous characters
    if '%' in location:
        encoded_patterns = [
            '%2f',   # /
            '%5c',   # \
            '%2e',   # .
            '%40',   # @
            '%3a',   # :
        ]
        for enc_pattern in encoded_patterns:
            if enc_pattern in location_lower:
                raise ValueError(f"Location contains encoded dangerous character: {enc_pattern}")
    
    return location.strip()


def validate_date_format(date: str) -> str:
    """Validate date format to prevent injection attacks.
    
    Args:
        date: Date string in YYYY-MM-DD format
        
    Returns:
        Validated date string
        
    Raises:
        ValueError: If date format is invalid
    """
    if not date or not isinstance(date, str):
        raise ValueError("Date must be a non-empty string")
    
    # Strict date format validation
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
        raise ValueError(f"Invalid date format. Must be YYYY-MM-DD, got: {date}")
    
    # Validate date values are reasonable
    try:
        year, month, day = map(int, date.split('-'))
        # Check year range (reasonable bounds)
        if year < 1800 or year > 2100:
            raise ValueError(f"Year out of range: {year}")
        if month < 1 or month > 12:
            raise ValueError(f"Month out of range: {month}")
        if day < 1 or day > 31:
            raise ValueError(f"Day out of range: {day}")
        
        # Try to create date to validate it's a real date
        datetime(year, month, day)
    except ValueError as e:
        if "out of range" in str(e):
            raise
        raise ValueError(f"Invalid date: {date}")
    
    return date


def build_visual_crossing_url(location: str, date: str, remote: bool = True) -> str:
    """Build Visual Crossing API URL with consistent parameters and SSRF protection.
    
    Args:
        location: The location to get weather data for (will be validated)
        date: The date in YYYY-MM-DD format (will be validated)
        remote: Whether to include remote data parameters (default: True)
        
    Returns:
        URL string for Visual Crossing API
        
    Raises:
        ValueError: If location or date validation fails
    """
    from urllib.parse import quote
    import logging
    from config import (
        VISUAL_CROSSING_BASE_URL, VISUAL_CROSSING_UNIT_GROUP,
        VISUAL_CROSSING_INCLUDE_PARAMS, VISUAL_CROSSING_REMOTE_DATA, API_KEY
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate inputs to prevent SSRF and injection attacks
    try:
        validated_location = validate_location_for_ssrf(location)
        validated_date = validate_date_format(date)
    except ValueError as e:
        # Log the error but don't expose the exact validation failure to prevent information disclosure
        logger.error(f"Location or date validation failed: {str(e)}")
        raise ValueError("Invalid location or date format") from e
    
    # Clean and URL-encode the validated location
    cleaned_location = clean_location_string(validated_location)
    encoded_location = quote(cleaned_location, safe='')
    
    # Final validation: ensure encoded location doesn't reintroduce dangerous patterns
    encoded_lower = encoded_location.lower()
    dangerous_encoded = ['%2f', '%5c', '%40', '%3a%3a%2f', 'localhost', '127.0.0.1']
    for pattern in dangerous_encoded:
        if pattern in encoded_lower:
            logger.error(f"Encoded location contains dangerous pattern after encoding: {pattern}")
            raise ValueError("Invalid location format")
    
    base_params = f"unitGroup={VISUAL_CROSSING_UNIT_GROUP}&include={VISUAL_CROSSING_INCLUDE_PARAMS}&key={API_KEY}"
    if remote:
        return f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}/{validated_date}?{base_params}&{VISUAL_CROSSING_REMOTE_DATA}"
    else:
        return f"{VISUAL_CROSSING_BASE_URL}/{encoded_location}/{validated_date}?{base_params}"
