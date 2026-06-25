"""Health check endpoint — mirrors HealthController.java."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/actuator/health")
async def health():
    """Standard health check endpoint."""
    return {"status": "UP"}


@router.get("/actuator/info")
async def info():
    return {"app": {"name": "honghu-ai", "version": "0.0.1"}}
