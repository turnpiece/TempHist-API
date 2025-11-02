"""Firebase initialization and authentication utilities."""
import os
import json
import logging
import firebase_admin
from firebase_admin import credentials
from config import DEBUG

logger = logging.getLogger(__name__)


def initialize_firebase():
    """Initialize Firebase Admin SDK."""
    try:
        # Try to load from environment variable first (for Railway/production)
        firebase_creds_json = os.getenv("FIREBASE_SERVICE_ACCOUNT") or os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if firebase_creds_json:
            firebase_creds = json.loads(firebase_creds_json)
            cred = credentials.Certificate(firebase_creds)
        else:
            # Fall back to file (for local development)
            if os.path.exists("firebase-service-account.json"):
                cred = credentials.Certificate("firebase-service-account.json")
            else:
                logger.warning("⚠️ No Firebase credentials found - Firebase features will be disabled")
                cred = None
        
        if cred:
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase initialized successfully")
    except ValueError:
        # Firebase app already initialized, skip
        logger.info("Firebase app already initialized")
        pass
    except Exception as e:
        logger.error(f"❌ Error initializing Firebase: {e}")
        # Continue without Firebase - the app can still work without it


def verify_firebase_token(request):
    """Verify Firebase authentication token."""
    from fastapi import HTTPException, Request
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    id_token = auth_header.split(" ")[1]
    try:
        from firebase_admin import auth
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=403, detail="Invalid token")
