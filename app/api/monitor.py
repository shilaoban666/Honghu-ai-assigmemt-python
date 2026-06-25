"""Monitor API endpoints — mirrors MonitorController.java."""

import os

from fastapi import APIRouter

router = APIRouter(prefix="/api/monitor", tags=["Monitor"])


@router.get("")
async def system_metrics():
    """Get system metrics overview."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu": {
                "percent": cpu,
                "cores": psutil.cpu_count(),
            },
            "memory": {
                "totalMb": mem.total // (1024 * 1024),
                "usedMb": mem.used // (1024 * 1024),
                "percent": mem.percent,
            },
            "disk": {
                "totalMb": disk.total // (1024 * 1024),
                "usedMb": disk.used // (1024 * 1024),
                "percent": disk.percent,
            },
        }
    except ImportError:
        return {"cpu": {"percent": 0}, "memory": {"percent": 0}, "disk": {"percent": 0}}
