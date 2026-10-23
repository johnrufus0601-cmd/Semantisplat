"""
Query route — handles natural-language text queries, 3D object detection,
metric size estimation, and spatial reasoning queries.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from semantisplat.api.models import QueryRequest, QueryResponse
from semantisplat.detection.detector_3d import OpenVocab3DDetector, BoundingBox3D
from semantisplat.retrieval.result_formatter import format_result, result_to_dict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])

# Cache loaded query engines per scene
_engines: dict[str, object] = {}
_detectors: dict[str, OpenVocab3DDetector] = {}


def _get_detector() -> OpenVocab3DDetector:
    if "detector" not in _detectors:
        from semantisplat.config import load_config
        config = load_config()
        _detectors["detector"] = OpenVocab3DDetector(device=config.device)
    return _detectors["detector"]


def _get_engine(scene_id: str):
    """Get or lazily load a query engine for the given scene."""
    if scene_id in _engines:
        return _engines[scene_id]

    scene_path = Path(f"data/scenes/{scene_id}/scene_final.ply")
    compressor_path = Path(f"data/scenes/{scene_id}/compressor.pt")

    if not scene_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Scene '{scene_id}' not found. Train a scene first.",
        )

    from semantisplat.retrieval.query_engine import SemanticQueryEngine
    from semantisplat.config import load_config
    
    config = load_config()

    engine = SemanticQueryEngine.from_files(
        scene_path=scene_path,
        compressor_path=compressor_path,
        device=config.device,
    )

    _engines[scene_id] = engine
    return engine


def _load_detected_scene_info(scene_id: str) -> tuple[List[Dict[str, Any]], str]:
    """Load cached detected 3D bounding boxes and scene domain from disk if available."""
    json_path = Path(f"data/scenes/{scene_id}/detected_objects.json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data, "indoor_room"
                if isinstance(data, dict):
                    objs = data.get("objects", [])
                    dom = data.get("domain", "indoor_room")
                    return objs, dom
        except Exception as e:
            logger.warning("Error reading detected_objects.json for %s: %s", scene_id, e)
    return [], "indoor_room"


def _load_detected_objects(scene_id: str) -> List[Dict[str, Any]]:
    objs, _ = _load_detected_scene_info(scene_id)
    return objs


@router.get("/detect/{scene_id}")
async def detect_objects(scene_id: str):
    """
    Return all detected 3D oriented bounding boxes for the scene.
    If not yet computed, automatically runs the OpenVocab3DDetector with auto-domain classification.
    """
    # 1. Check cached detection file
    cached_objects, cached_domain = _load_detected_scene_info(scene_id)
    if cached_objects:
        return {"objects": cached_objects, "domain": cached_domain}

    # 2. If not cached, attempt detection on available frames
    frames_dir = Path(f"data/frames/{scene_id}")
    frame_paths = sorted(list(frames_dir.glob("frame_*.jpg")) + list(frames_dir.glob("frame_*.png")))
    
    detector = _get_detector()

    if frame_paths:
        try:
            if len(frame_paths) == 1:
                boxes = detector.detect_image(frame_paths[0])
            else:
                boxes = detector.detect_video(frame_paths)

            domain = detector.last_detected_domain
            objects_list = [b.to_dict() for b in boxes]
            out_json = Path(f"data/scenes/{scene_id}/detected_objects.json")
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"domain": domain, "objects": objects_list}, f, indent=2)

            return {"objects": objects_list, "domain": domain}
        except Exception as e:
            logger.warning("Auto-detection from frames failed for scene %s: %s", scene_id, e)

    # 3. Fallback: Query Gaussian scene for vocabulary if ply exists
    try:
        engine = _get_engine(scene_id)
        vocab = ["chair", "table", "ladder", "curtain", "fan", "sofa", "door", "window", "monitor", "plant", "cabinet"]
        objects = []
        for word in vocab:
            res = engine.query(text=word, top_k=2, similarity_threshold=0.22)
            if res.match_found:
                for match in res.matches:
                    if match.obb_3d:
                        objects.append({
                            "label": word,
                            "confidence": round(match.similarity_score, 3),
                            "center": match.obb_3d.get("center", [0, 0, 0]),
                            "axes": match.obb_3d.get("axes", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                            "extents": match.obb_3d.get("extents", [0.4, 0.4, 0.4]),
                            "size": [
                                round(match.obb_3d.get("extents", [0.4, 0.4, 0.4])[0] * 2.0, 2),
                                round(match.obb_3d.get("extents", [0.4, 0.4, 0.4])[1] * 2.0, 2),
                                round(match.obb_3d.get("extents", [0.4, 0.4, 0.4])[2] * 2.0, 2),
                            ],
                            "yaw": 0.0,
                            "obb_3d": match.obb_3d,
                        })

        if objects:
            out_json = Path(f"data/scenes/{scene_id}/detected_objects.json")
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"domain": "indoor_room", "objects": objects}, f, indent=2)

        return {"objects": objects, "domain": "indoor_room"}
    except Exception as e:
        logger.error("Detect fallback failed for scene %s: %s", scene_id, e)
        return {"objects": [], "domain": "indoor_room"}


@router.post("/query", response_model=QueryResponse)
async def query_scene(request: QueryRequest):
    """
    Search a trained 3D scene with natural-language text queries,
    3D object bounding box highlighting, and spatial reasoning questions.
    """
    query_text = request.text.strip()
    query_lower = query_text.lower()
    scene_id = request.scene_id

    # 1. Load any detected 3D bounding boxes and scene domain for this scene
    detected_objects, scene_domain = _load_detected_scene_info(scene_id)
    if not detected_objects:
        det_res = await detect_objects(scene_id)
        detected_objects = det_res.get("objects", [])
        scene_domain = det_res.get("domain", scene_domain)

    # Reconstruct BoundingBox3D objects for spatial reasoning
    box_objs: List[BoundingBox3D] = []
    for obj in detected_objects:
        if isinstance(obj, dict):
            c = obj.get("center", [0, 0, 0])
            s = obj.get("size", [0.5, 0.5, 0.5])
            ex = obj.get("extents", [s[0] / 2.0, s[1] / 2.0, s[2] / 2.0])
            lbl = obj.get("label", "object")
            conf = float(obj.get("confidence", 0.8))
            ax = obj.get("axes", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            corners = obj.get("corners_3d", [])
            box_objs.append(BoundingBox3D(
                center=c,
                extents=ex,
                size=s,
                axes=ax,
                yaw=float(obj.get("yaw", 0.0)),
                corners_3d=corners,
                label=lbl,
                confidence=conf,
                bbox_2d=obj.get("bbox_2d", [0, 0, 100, 100]),
                color=obj.get("color", [0.2, 0.8, 0.2]),
                size_str=obj.get("size_str", ""),
                short_size_str=obj.get("short_size_str", ""),
            ))

    # 2. Check for Spatial Reasoning Query (e.g. "Can a 2-meter table fit?", "Can a 6-foot sofa fit here?")
    spatial_triggers = ["fit", "space", "clearance", "room for", "can a", "will a", "could a", "does a", "is there", "accommodate", "place a", "fit here"]
    is_spatial = any(w in query_lower for w in spatial_triggers)
    dim_match = re.search(
        r"(\d+(?:\.\d+)?)\s*[-–—]?\s*(meters?|meter|m|feet|foot|ft|'|inches?|inch|in|\"|cm|centimeters?|mm)\b",
        query_lower,
    )

    if is_spatial:
        detector = _get_detector()
        spatial_verdict = detector.answer_spatial_query(query_text, box_objs, domain=scene_domain)
        
        # Find closest related object or candidate box to highlight
        matched_box = None
        target_name = spatial_verdict.get("target_item", "")
        for b in box_objs:
            if target_name in b.label.lower() or b.label.lower() in target_name:
                matched_box = b
                break
        if not matched_box and box_objs:
            matched_box = box_objs[0]

        results = []
        if matched_box:
            results.append({
                "rank": 1,
                "region_id": f"spatial_{matched_box.label}",
                "centroid": {"x": matched_box.center[0], "y": matched_box.center[1], "z": matched_box.center[2]},
                "similarity": 0.98,
                "gaussian_count": matched_box.num_points or 150,
                "bbox_3d": {
                    "min": {"x": matched_box.center[0] - matched_box.extents[0], "y": matched_box.center[1] - matched_box.extents[1], "z": matched_box.center[2] - matched_box.extents[2]},
                    "max": {"x": matched_box.center[0] + matched_box.extents[0], "y": matched_box.center[1] + matched_box.extents[1], "z": matched_box.center[2] + matched_box.extents[2]},
                },
                "obb_3d": matched_box.to_dict(),
                "size_str": matched_box.size_str,
                "highlight_indices": list(range(50)),
            })

        logger.info("Spatial reasoning query '%s' answered: fits=%s, msg='%s'", query_text, spatial_verdict.get("fits"), spatial_verdict.get("message"))
        return QueryResponse(
            query=query_text,
            match_found=True,
            message=spatial_verdict["message"],
            max_similarity=0.98,
            result_count=len(results),
            results=results,
            spatial_query=spatial_verdict,
        )

    # 3. Direct match against detected 3D bounding boxes with synonym expansion
    synonym_map = {
        "wheel": ["steering wheel", "wheel"],
        "steering": ["steering wheel"],
        "steering wheel": ["steering wheel"],
        "dashboard": ["dashboard", "dash"],
        "dash": ["dashboard"],
        "seat": ["car seat", "seat", "chair"],
        "car seat": ["car seat"],
        "chair": ["chair", "office chair", "car seat"],
        "screen": ["touchscreen display", "touchscreen", "screen", "monitor"],
        "touchscreen": ["touchscreen display", "screen"],
        "display": ["touchscreen display", "screen", "monitor"],
        "monitor": ["computer monitor", "monitor", "screen"],
        "console": ["center console", "console"],
        "center console": ["center console"],
        "vent": ["air vent", "vent"],
        "air vent": ["air vent"],
        "shifter": ["gear shifter", "shifter"],
        "gear shifter": ["gear shifter"],
        "mirror": ["rearview mirror", "mirror"],
        "rearview mirror": ["rearview mirror"],
        "sofa": ["sofa", "couch"],
        "couch": ["sofa", "couch"],
        "table": ["table", "desk", "dining table"],
        "desk": ["desk", "table", "office desk"],
        "door": ["car door", "door"],
        "window": ["car window", "window"],
    }

    matched_boxes = []
    for b in box_objs:
        b_label = b.label.lower()
        if b_label in query_lower or query_lower in b_label:
            matched_boxes.append(b)
            continue

        # Check synonym groupings
        syns = synonym_map.get(query_lower, [])
        if any(s in b_label or b_label in s for s in syns):
            matched_boxes.append(b)
            continue

        for token in query_lower.split():
            if token in synonym_map:
                if any(s in b_label or b_label in s for s in synonym_map[token]):
                    matched_boxes.append(b)
                    break

    if matched_boxes:
        results = []
        for rank, mb in enumerate(matched_boxes[:request.top_k], start=1):
            results.append({
                "rank": rank,
                "region_id": f"obj_{mb.label}_{rank}",
                "centroid": {"x": mb.center[0], "y": mb.center[1], "z": mb.center[2]},
                "similarity": round(mb.confidence, 3),
                "gaussian_count": mb.num_points or 200,
                "bbox_3d": {
                    "min": {"x": mb.center[0] - mb.extents[0], "y": mb.center[1] - mb.extents[1], "z": mb.center[2] - mb.extents[2]},
                    "max": {"x": mb.center[0] + mb.extents[0], "y": mb.center[1] + mb.extents[1], "z": mb.center[2] + mb.extents[2]},
                },
                "obb_3d": mb.to_dict(),
                "size_str": mb.size_str,
                "highlight_indices": list(range(50)),
            })

        logger.info("Direct 3D box match for '%s': found %d box(es).", query_text, len(results))
        return QueryResponse(
            query=query_text,
            match_found=True,
            message=f"Found {len(results)} matching 3D bounding box(es) for '{query_text}'.",
            max_similarity=float(results[0]["similarity"]),
            result_count=len(results),
            results=results,
            spatial_query=None,
        )

    # 4. Check domain compatibility before Gaussian semantic feature search
    if scene_domain == "vehicle" and any(w in query_lower for w in ["sofa", "couch", "bed", "wardrobe", "bathtub", "fireplace"]):
        return QueryResponse(
            query=query_text,
            match_found=False,
            message=f"No confident match found for '{query_text}' inside this vehicle interior. Try: steering wheel, dashboard, car seat, screen or spatial queries like 'Can a 2-meter table fit?'",
            max_similarity=0.0,
            result_count=0,
            results=[],
            spatial_query=None,
        )

    # 5. Fall back to Gaussian semantic feature search
    try:
        engine = _get_engine(scene_id)
        result = engine.query(
            text=query_text,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold or 0.22,
        )
        formatted = format_result(result)
        response_dict = result_to_dict(formatted)
        return QueryResponse(**response_dict)
    except Exception as e:
        logger.info("Gaussian query fallback: %s", e)

    # 5. No match found
    domain_hints = "steering wheel, dashboard, car seat, screen" if scene_domain == "vehicle" else "chair, sofa, table, fan, ladder"
    return QueryResponse(
        query=query_text,
        match_found=False,
        message=f"No confident match found for '{query_text}'. Try: {domain_hints} or spatial queries like 'Can a 2-meter table fit?'",
        max_similarity=0.0,
        result_count=0,
        results=[],
        spatial_query=None,
    )
