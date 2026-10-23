"""
Pydantic models for API request/response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ===== Request Models =====

class QueryRequest(BaseModel):
    """Text query request."""
    text: str = Field(..., description="Natural-language search query", min_length=1)
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=100)
    similarity_threshold: float | None = Field(
        default=None, description="Override similarity threshold (0.0–1.0)"
    )
    scene_id: str = Field(default="default", description="Scene to search")


# ===== Response Models =====

class CentroidResponse(BaseModel):
    x: float
    y: float
    z: float


class BBoxCorner(BaseModel):
    x: float
    y: float
    z: float


class BBox3DResponse(BaseModel):
    min: BBoxCorner
    max: BBoxCorner


class MatchResponse(BaseModel):
    rank: int
    region_id: str
    centroid: CentroidResponse
    similarity: float
    gaussian_count: int
    bbox_3d: BBox3DResponse | None = None
    obb_3d: dict | None = None
    size_str: str | None = None
    highlight_indices: list[int]


class QueryResponse(BaseModel):
    query: str
    match_found: bool
    message: str
    max_similarity: float
    result_count: int
    results: list[MatchResponse]
    spatial_query: dict | None = None


class UploadResponse(BaseModel):
    job_id: str
    message: str
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    stage: str   # Current pipeline stage
    progress: float  # 0.0 - 1.0
    stage_progress: float | None = None # Progress for the current stage
    message: str


class SceneInfo(BaseModel):
    scene_id: str
    name: str
    gaussian_count: int
    semantic_dim: int
    created_at: str


class SceneListResponse(BaseModel):
    scenes: list[SceneInfo]
