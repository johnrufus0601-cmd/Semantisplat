"""
Status route — check processing status of a pipeline job.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pathlib import Path

from semantisplat.api.models import StatusResponse
from semantisplat.api.routes.upload import jobs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Get the current processing status of a pipeline job."""
    if job_id not in jobs:
        return StatusResponse(
            job_id=job_id,
            status="not_found",
            stage="unknown",
            progress=0.0,
            message=f"Job '{job_id}' not found.",
        )

    job = jobs[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        stage=job["stage"],
        progress=job["progress"],
        stage_progress=job.get("stage_progress"),
        message=job["message"],
    )


@router.websocket("/ws/status/{job_id}")
async def ws_status(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time processing status updates.

    Sends status updates every second until the job is complete or failed.
    """
    await websocket.accept()

    import asyncio

    try:
        while True:
            if job_id not in jobs:
                await websocket.send_json({
                    "job_id": job_id,
                    "status": "not_found",
                    "stage": "unknown",
                    "progress": 0.0,
                    "stage_progress": 0.0,
                    "message": f"Job '{job_id}' not found.",
                })
                break

            job = jobs[job_id]
            await websocket.send_json({
                "job_id": job_id,
                "status": job["status"],
                "stage": job["stage"],
                "progress": job["progress"],
                "stage_progress": job.get("stage_progress"),
                "message": job["message"],
            })

            if job["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s.", job_id)
    except Exception as e:
        logger.error("WebSocket error for job %s: %s", job_id, e)


@router.get("/heatmap/{scene_id}")
async def get_heatmap(scene_id: str):
    """Serve the coverage heatmap for a given scene."""
    # Using the default data directory from SemantiSplatConfig ("data")
    heatmap_path = Path("data") / "heatmap" / scene_id / "coverage_heatmap.png"
    if heatmap_path.exists():
        return FileResponse(heatmap_path, media_type="image/png")
    return {"status": "not_found", "message": "Heatmap not found."}


@router.get("/scene/{scene_id}")
async def get_scene(scene_id: str):
    """Serve the trained Gaussian splat scene (.ply)."""
    scene_path = Path("data") / "scenes" / scene_id / "scene_final.ply"
    if scene_path.exists():
        return FileResponse(scene_path, media_type="application/octet-stream")
    return {"status": "not_found", "message": "Scene not found."}
