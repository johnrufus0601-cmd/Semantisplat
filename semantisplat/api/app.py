"""
SemantiSplat — FastAPI Application.

Main entry point for the web API. Mounts all route handlers and
serves the frontend static files.

Usage:
    python -m semantisplat.api.app
    # or
    uvicorn semantisplat.api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from semantisplat.api.routes import query, status, upload, reprojection

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="SemantiSplat",
    description=(
        "Open-Vocabulary Semantic 3D Scene Search via Gaussian Splatting. "
        "Upload a video, reconstruct a 3D scene, and search it with natural language."
    ),
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(upload.router)
app.include_router(status.router)
app.include_router(query.router)
app.include_router(reprojection.router)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "SemantiSplat", "version": "1.0.0"}


# Serve frontend static files
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    logger.info("Frontend served from %s.", frontend_dir.resolve())
else:
    logger.warning("Frontend directory '%s' not found. API-only mode.", frontend_dir)


# Entry point
if __name__ == "__main__":
    import os
    import socket
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if s.connect_ex(("127.0.0.1", port)) == 0:
        logger.info("Port %d is occupied, falling back to 8001.", port)
        port = 8001
    s.close()

    logger.info("Starting SemantiSplat server on port %d...", port)
    uvicorn.run(
        "semantisplat.api.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
