"""Health check endpoint."""

from fastapi import APIRouter
from app.schemas.forecast import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status, identity, and region metadata.
    """
    return HealthResponse(
        status="ok",
        service="ArgoCast-Forecasting-API",
        region="Arabian Sea / Northern Indian Ocean",
        version="2.0.0"
    )