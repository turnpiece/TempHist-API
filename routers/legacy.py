"""Legacy endpoints (410 Gone)."""
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from utils.firebase import verify_firebase_token

router = APIRouter()


@router.get("/data/{location}/{month_day}")
async def removed_data_endpoint():
    """Legacy data endpoint has been removed. Use /v1/records/daily/{location}/{month_day} instead."""
    return JSONResponse(
        content={
            "error": "Endpoint removed",
            "message": "This endpoint has been removed. Please use the v1 API instead.",
            "new_endpoint": "/v1/records/daily/{location}/{month_day}",
            "migration_guide": "/"
        },
        status_code=410,
        headers={
            "X-Removed": "true",
            "X-New-Endpoint": "/v1/records/daily/{location}/{month_day}",
            "Cache-Control": "no-cache"
        }
    )


@router.get("/average/{location}/{month_day}")
async def removed_average_endpoint():
    """Legacy average endpoint has been removed. Use /v1/records/daily/{location}/{month_day}/average instead."""
    return JSONResponse(
        content={
            "error": "Endpoint removed",
            "message": "This endpoint has been removed. Please use the v1 API instead.",
            "new_endpoint": "/v1/records/daily/{location}/{month_day}/average",
            "migration_guide": "/"
        },
        status_code=410,
        headers={
            "X-Removed": "true",
            "X-New-Endpoint": "/v1/records/daily/{location}/{month_day}/average",
            "Cache-Control": "no-cache"
        }
    )


@router.get("/trend/{location}/{month_day}")
async def removed_trend_endpoint():
    """Legacy trend endpoint has been removed. Use /v1/records/daily/{location}/{month_day}/trend instead."""
    return JSONResponse(
        content={
            "error": "Endpoint removed",
            "message": "This endpoint has been removed. Please use the v1 API instead.",
            "new_endpoint": "/v1/records/daily/{location}/{month_day}/trend",
            "migration_guide": "/"
        },
        status_code=410,
        headers={
            "X-Removed": "true",
            "X-New-Endpoint": "/v1/records/daily/{location}/{month_day}/trend",
            "Cache-Control": "no-cache"
        }
    )


@router.get("/summary/{location}/{month_day}")
async def removed_summary_endpoint():
    """Legacy summary endpoint has been removed. Use /v1/records/daily/{location}/{month_day}/summary instead."""
    return JSONResponse(
        content={
            "error": "Endpoint removed",
            "message": "This endpoint has been removed. Please use the v1 API instead.",
            "new_endpoint": "/v1/records/daily/{location}/{month_day}/summary",
            "migration_guide": "/"
        },
        status_code=410,
        headers={
            "X-Removed": "true",
            "X-New-Endpoint": "/v1/records/daily/{location}/{month_day}/summary",
            "Cache-Control": "no-cache"
        }
    )


@router.get("/protected-endpoint")
def protected_route(user=Depends(verify_firebase_token)):
    """Protected endpoint that requires Firebase authentication."""
    return {"message": "You are authenticated!", "user": user}
