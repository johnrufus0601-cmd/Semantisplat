"""
FR-4 (Part B): Result Formatting.

Formats top-K query results for the Web API and UI:
- JSON-serializable output
- 3D bounding boxes for visualization
- Spatial grouping of nearby Gaussians into regions
- Human-readable messages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from semantisplat.retrieval.query_engine import QueryMatch, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class FormattedMatch:
    """A single formatted match result for API/UI consumption."""
    rank: int
    region_id: str
    centroid: dict[str, float]            # {"x": ..., "y": ..., "z": ...}
    similarity_score: float
    gaussian_count: int
    bbox_3d: dict | None                  # {"min": {"x":...}, "max": {"x":...}} or None
    obb_3d: dict | None
    highlight_indices: list[int]           # Gaussian indices to highlight in viewer
    size_str: str | None = None


@dataclass
class FormattedQueryResult:
    """Formatted query result ready for JSON serialization."""
    query_text: str
    match_found: bool
    message: str
    max_similarity: float
    result_count: int
    results: list[FormattedMatch]
    spatial_query: dict | None = None


def format_result(query_result: QueryResult) -> FormattedQueryResult:
    """
    Convert a raw QueryResult into a formatted, JSON-serializable result
    suitable for the Web API response.

    Args:
        query_result: Raw result from SemanticQueryEngine.query().

    Returns:
        FormattedQueryResult with clean, serializable data.
    """
    formatted_matches: list[FormattedMatch] = []

    for rank, match in enumerate(query_result.matches, start=1):
        centroid = {
            "x": float(match.centroid[0]),
            "y": float(match.centroid[1]),
            "z": float(match.centroid[2]),
        }

        bbox = None
        if match.bbox_3d is not None:
            min_c, max_c = match.bbox_3d
            bbox = {
                "min": {"x": float(min_c[0]), "y": float(min_c[1]), "z": float(min_c[2])},
                "max": {"x": float(max_c[0]), "y": float(max_c[1]), "z": float(max_c[2])},
            }

        size_str = None
        if match.obb_3d and "extents" in match.obb_3d:
            ex = match.obb_3d["extents"]
            w = ex[0] * 2.0
            h = ex[1] * 2.0
            d = ex[2] * 2.0
            size_str = f"{w:.2f}m × {h:.2f}m × {d:.2f}m ({w*3.28084:.1f}ft × {h*3.28084:.1f}ft × {d*3.28084:.1f}ft)"

        formatted_matches.append(FormattedMatch(
            rank=rank,
            region_id=f"region_{rank}",
            centroid=centroid,
            similarity_score=round(match.similarity_score, 4),
            gaussian_count=len(match.gaussian_indices),
            bbox_3d=bbox,
            obb_3d=match.obb_3d,
            highlight_indices=match.gaussian_indices,
            size_str=size_str,
        ))

    return FormattedQueryResult(
        query_text=query_result.query_text,
        match_found=query_result.match_found,
        message=query_result.message,
        max_similarity=round(query_result.max_similarity, 4),
        result_count=len(formatted_matches),
        results=formatted_matches,
    )


def result_to_dict(result: FormattedQueryResult) -> dict:
    """
    Convert FormattedQueryResult to a plain dict for JSON serialization.

    This is what gets sent over the API as the response body.
    """
    data = {
        "query": result.query_text,
        "match_found": result.match_found,
        "message": result.message,
        "max_similarity": result.max_similarity,
        "result_count": result.result_count,
        "results": [
            {
                "rank": m.rank,
                "region_id": m.region_id,
                "centroid": m.centroid,
                "similarity": m.similarity_score,
                "gaussian_count": m.gaussian_count,
                "bbox_3d": m.bbox_3d,
                "obb_3d": m.obb_3d,
                "size_str": m.size_str,
                "highlight_indices": m.highlight_indices,
            }
            for m in result.results
        ],
    }
    if result.spatial_query:
        data["spatial_query"] = result.spatial_query
    return data
