"""Admin authentication for operational and monitoring endpoints."""

from __future__ import annotations

from fastapi import Header, HTTPException

from config import ADMIN_API_KEY


def is_admin_path(path: str, method: str) -> bool:
    """Return True if the path is an admin-only route (X-Admin-Key required)."""
    if path.startswith("/admin/"):
        return True
    if path.startswith("/usage-stats"):
        return True
    if path.startswith("/cache-stats"):
        return True
    if path == "/rate-limit-stats":
        return True
    if path.startswith("/cache/invalidate/"):
        return True
    if path == "/cache/clear" and method == "DELETE":
        return True
    if path == "/cache/info":
        return True
    if path == "/cache-warm" and method == "POST":
        return True
    return False


def verify_admin_key(x_admin_key: str | None = Header(None, alias="X-Admin-Key")) -> bool:
    """FastAPI dependency: require valid X-Admin-Key header."""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API not configured (ADMIN_API_KEY not set)")
    if not x_admin_key or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    return True


def admin_key_is_valid(admin_key: str | None) -> bool:
    """Return True when the header matches ADMIN_API_KEY."""
    if not ADMIN_API_KEY:
        return False
    return bool(admin_key) and admin_key == ADMIN_API_KEY
