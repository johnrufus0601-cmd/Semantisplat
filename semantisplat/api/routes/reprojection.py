"""
Reprojection Route — Renders 3D metric bounding boxes and geometry
overlaid directly onto the original image or video frames.
Cleaned and optimized for video sequences with top-K filtering,
smart collision avoidance, and crisp rendering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reprojection", tags=["reprojection"])

# Palette for bounding box drawing (BGR format for OpenCV)
BOX_COLORS_BGR = {
    # Vehicle / Automotive
    "steering wheel": (60, 60, 255),       # Crimson red
    "wheel": (60, 60, 255),                # Crimson red
    "dashboard": (255, 180, 50),           # Sky blue
    "car seat": (100, 220, 80),            # Emerald green
    "seat": (100, 220, 80),                # Emerald green
    "center console": (30, 165, 255),      # Orange
    "console": (30, 165, 255),             # Orange
    "touchscreen display": (255, 90, 200), # Violet / Magenta
    "touchscreen": (255, 90, 200),         # Violet / Magenta
    "gear shifter": (50, 215, 255),        # Bright amber
    "shifter": (50, 215, 255),             # Bright amber
    "rearview mirror": (230, 230, 80),     # Cyan
    "windshield": (230, 180, 130),         # Ice blue
    "car door": (180, 150, 150),           # Slate gray
    "car window": (200, 200, 100),         # Aqua
    "air vent": (180, 180, 180),           # Light gray
    "pedal": (50, 130, 205),               # Copper

    # Indoor Furniture & Objects
    "chair": (255, 180, 50),     # Sky blue
    "office chair": (255, 180, 50),
    "sofa": (75, 100, 255),      # Coral red
    "couch": (75, 100, 255),     # Coral red
    "table": (100, 230, 80),     # Emerald green
    "desk": (120, 220, 70),      # Jade green
    "bed": (255, 100, 200),      # Purple
    "fan": (50, 220, 255),       # Bright yellow
    "ladder": (30, 150, 255),    # Orange
    "curtain": (230, 200, 150),  # Soft blue
    "lamp": (130, 240, 255),     # Warm yellow
    "monitor": (230, 130, 50),   # Deep blue
    "computer monitor": (230, 130, 50),
    "laptop": (220, 150, 60),
    "tv": (230, 130, 50),        # Deep blue
    "screen": (230, 130, 50),
    "cabinet": (70, 130, 180),   # Wood brown
    "bookshelf": (70, 130, 180),
    "shelf": (70, 130, 180),
    "plant": (80, 200, 30),      # Leaf green
    "door": (150, 130, 130),     # Slate gray
    "window": (200, 200, 100),   # Aqua
    "refrigerator": (210, 210, 120),
    "default": (50, 200, 230),   # Gold
}


def _get_frame_paths(scene_id: str) -> List[Path]:
    """Find all extracted frames for the scene."""
    frames_dir = Path("data") / "frames" / scene_id
    if not frames_dir.exists():
        return []
    paths = sorted(list(frames_dir.glob("frame_*.jpg")) + list(frames_dir.glob("frame_*.png")))
    return paths


def _load_detected_boxes(scene_id: str) -> List[Dict[str, Any]]:
    """Load detected 3D boxes from JSON."""
    json_path = Path("data") / "scenes" / scene_id / "detected_objects.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "objects" in data:
                    return data["objects"]
        except Exception as e:
            logger.warning("Error reading detected_objects.json: %s", e)
    return []


@router.get("/{scene_id}/info")
async def get_info(scene_id: str):
    """
    Return list of available frames for reprojection.
    Works for both single image uploads and video frame sequences.
    """
    frames = _get_frame_paths(scene_id)
    if not frames:
        raise HTTPException(status_code=404, detail="No frames found for this scene.")

    images = [{"id": i, "name": p.name} for i, p in enumerate(frames)]
    return {"images": images}


@router.get("/{scene_id}/frame/{image_id}")
async def get_frame(
    scene_id: str,
    image_id: int,
    top_k: int = Query(8, ge=1, le=25, description="Number of top confidence boxes to render"),
    label_mode: str = Query("clean", description="Badge style: 'clean', 'compact', or 'none'"),
):
    """
    Serve video/image frame with 3D Oriented Bounding Boxes,
    wireframes, and real-world metric dimensions overlaid.
    Cleaned with top-K confidence filtering and collision-free labels.
    """
    frames = _get_frame_paths(scene_id)
    if not frames or image_id < 0 or image_id >= len(frames):
        raise HTTPException(status_code=404, detail=f"Frame index {image_id} not found.")

    frame_path = frames[image_id]
    img = cv2.imread(str(frame_path))
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to load image file.")

    H, W = img.shape[:2]
    fx = fy = float(max(W, H)) * 0.95
    cx = float(W) / 2.0
    cy = float(H) / 2.0

    boxes = _load_detected_boxes(scene_id)
    if not boxes:
        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return Response(content=encoded.tobytes(), media_type="image/jpeg")

    # 1. Check for COLMAP reconstruction to get exact camera pose for this video frame
    R_cam = np.eye(3, dtype=np.float32)
    t_cam = np.zeros(3, dtype=np.float32)
    has_colmap_pose = False

    sparse_dir = Path("data") / "colmap" / scene_id / "sparse"
    recon_path = sparse_dir / "0" if (sparse_dir / "0").exists() else (sparse_dir if sparse_dir.exists() else None)
    if recon_path and recon_path.exists():
        try:
            import pycolmap
            recon = pycolmap.Reconstruction(str(recon_path))
            for _, img_obj in recon.images.items():
                if img_obj.name == frame_path.name:
                    pose = img_obj.cam_from_world()
                    R_cam = pose.rotation.matrix()
                    t_cam = pose.translation
                    has_colmap_pose = True

                    cam = recon.cameras[img_obj.camera_id]
                    if cam.model == "SIMPLE_PINHOLE":
                        fx = fy = cam.params[0]
                        cx, cy = cam.params[1], cam.params[2]
                    else:
                        fx, fy = cam.params[0], cam.params[1]
                        cx, cy = cam.params[2], cam.params[3]
                    break
        except Exception as e:
            logger.debug("COLMAP pose lookup: %s", e)

    # 2. If no COLMAP pose available for multi-frame video, associate boxes with keyframes
    target_frame_idx = image_id
    if not has_colmap_pose and len(frames) > 1:
        # Collect distinct keyframe indices present in boxes
        available_keyframes = {b.get("frame_index") for b in boxes if b.get("frame_index") is not None}
        if available_keyframes:
            # Find closest keyframe to current frame
            target_frame_idx = min(available_keyframes, key=lambda k: abs(k - image_id))

    # 3. Project and filter candidates
    candidates = []
    for box in boxes:
        # If no COLMAP pose and multiple frames exist, only show boxes from corresponding keyframe
        if not has_colmap_pose and len(frames) > 1:
            box_frame = box.get("frame_index")
            if box_frame is not None and abs(box_frame - target_frame_idx) > 5:
                continue

        lbl = str(box.get("label", "object")).lower()
        color_bgr = BOX_COLORS_BGR.get(lbl, BOX_COLORS_BGR["default"])
        conf = float(box.get("confidence", box.get("score", 0.75)))

        # 8 corners in 3D
        corners_3d = box.get("corners_3d")
        if not corners_3d or len(corners_3d) != 8:
            c = np.array(box.get("center", [0, 0, 2.0]), dtype=np.float32)
            ex = np.array(box.get("extents", [0.4, 0.4, 0.4]), dtype=np.float32)
            ax = np.array(box.get("axes", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=np.float32)
            corners_3d = []
            for sx in [-1, 1]:
                for sy in [-1, 1]:
                    for sz in [-1, 1]:
                        corners_3d.append((c + ax @ (np.array([sx, sy, sz]) * ex)).tolist())

        pts_3d = np.array(corners_3d, dtype=np.float32)

        # Transform to camera space
        if has_colmap_pose:
            pts_cam = (R_cam @ pts_3d.T).T + t_cam
        else:
            pts_cam = pts_3d

        # Depth check: object must be in front of camera
        mean_z = float(np.mean(pts_cam[:, 2]))
        if mean_z <= 0.25 or np.sum(pts_cam[:, 2] > 0.15) < 6:
            continue

        # Project 3D points to 2D screen coordinates
        safe_z = np.maximum(pts_cam[:, 2], 0.05)
        u = (fx * pts_cam[:, 0] / safe_z + cx).astype(np.int32)
        v = (fy * pts_cam[:, 1] / safe_z + cy).astype(np.int32)
        pts_2d = np.stack([u, v], axis=1)

        # Compute 2D bounding rectangle
        min_u, max_u = int(np.min(pts_2d[:, 0])), int(np.max(pts_2d[:, 0]))
        min_v, max_v = int(np.min(pts_2d[:, 1])), int(np.max(pts_2d[:, 1]))
        box_w = max_u - min_u
        box_h = max_v - min_v

        # Discard degenerate projections
        if box_w < 16 or box_h < 16 or box_w > W * 2.5 or box_h > H * 2.5:
            continue

        # Check overlap with visible image area
        ix1 = max(0, min_u)
        iy1 = max(0, min_v)
        ix2 = min(W, max_u)
        iy2 = min(H, max_v)
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter_area < 256 or (inter_area / max(1, box_w * box_h)) < 0.15:
            continue

        # Clean metric size string
        size_str = box.get("short_size_str") or box.get("size_str")
        if not size_str and "size" in box:
            s = box["size"]
            size_str = f"{s[0]:.2f}m × {s[1]:.2f}m × {s[2]:.2f}m"
        elif size_str and "(" in size_str:
            # Strip imperial parentheses for cleaner badge if present
            size_str = size_str.split("(")[0].strip()

        candidates.append({
            "box": box,
            "lbl": lbl,
            "conf": conf,
            "color_bgr": color_bgr,
            "pts_2d": pts_2d,
            "rect_2d": (min_u, min_v, max_u, max_v),
            "mean_z": mean_z,
            "size_str": size_str,
        })

    # 4. 2D NMS / Deduplication across overlapping candidate boxes
    candidates.sort(key=lambda c: c["conf"], reverse=True)
    kept_candidates = []
    for cand in candidates:
        r1 = cand["rect_2d"]
        area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
        suppress = False
        for k in kept_candidates:
            r2 = k["rect_2d"]
            area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
            ix1 = max(r1[0], r2[0])
            iy1 = max(r1[1], r2[1])
            ix2 = min(r1[2], r2[2])
            iy2 = min(r1[3], r2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            iou = inter / max(1, area1 + area2 - inter)

            # Suppress if heavy overlap with same class or near identical footprint
            if (iou > 0.45 and cand["lbl"] == k["lbl"]) or iou > 0.65:
                suppress = True
                break
        if not suppress:
            kept_candidates.append(cand)

    # 5. Top-K Selection & Depth Sorting (back-to-front rendering)
    selected_boxes = kept_candidates[:top_k]
    selected_boxes.sort(key=lambda c: c["mean_z"], reverse=True)

    # 6. Render: Shaded top faces on overlay first, blend, then crisp lines + badges
    overlay = img.copy()
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7),  # Pillars
    ]

    for item in selected_boxes:
        pts_2d = item["pts_2d"]
        color_bgr = item["color_bgr"]
        # Shaded top face
        top_poly = pts_2d[[4, 5, 7, 6]]
        cv2.fillPoly(overlay, [top_poly], color_bgr)

    # Blend top face shading onto img with subtle transparency
    cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)

    # Draw crisp wireframes
    for item in selected_boxes:
        pts_2d = item["pts_2d"]
        color_bgr = item["color_bgr"]

        # 12 wireframe edges
        for p1, p2 in edges:
            pt1 = (int(pts_2d[p1, 0]), int(pts_2d[p1, 1]))
            pt2 = (int(pts_2d[p2, 0]), int(pts_2d[p2, 1]))
            cv2.line(img, pt1, pt2, color_bgr, 2, cv2.LINE_AA)

        # Corner vertex markers for 3D clarity
        for p in range(8):
            cv2.circle(img, (int(pts_2d[p, 0]), int(pts_2d[p, 1])), 2, color_bgr, -1, cv2.LINE_AA)

    # 7. Draw clean, collision-free badges
    if label_mode != "none":
        placed_badges: List[Tuple[int, int, int, int]] = []

        for item in selected_boxes:
            pts_2d = item["pts_2d"]
            color_bgr = item["color_bgr"]
            lbl = item["lbl"]
            conf = item["conf"]
            size_str = item["size_str"]

            if label_mode == "compact":
                label_text = f"{lbl.upper()} ({int(conf * 100)}%)"
            else:
                label_text = f"{lbl.upper()} • {size_str} ({int(conf * 100)}%)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_w = tw + 14
            badge_h = th + 10

            # Target position near top-left of projected box
            min_u = int(np.min(pts_2d[:, 0]))
            min_v = int(np.min(pts_2d[:, 1]))
            bx = int(np.clip(min_u, 10, max(10, W - badge_w - 10)))
            by = int(np.clip(min_v - 8, badge_h + 6, H - 8))

            # Smart collision resolution against previously placed badges
            for _ in range(8):
                collision = False
                for ox1, oy1, ox2, oy2 in placed_badges:
                    if not (bx + badge_w < ox1 or bx > ox2 or by < oy1 or by - badge_h > oy2):
                        collision = True
                        by = oy2 + badge_h + 4  # push down
                        break
                if not collision or by > H - 10:
                    break

            bx = int(np.clip(bx, 10, max(10, W - badge_w - 10)))
            by = int(np.clip(by, badge_h + 6, H - 8))
            placed_badges.append((bx, by - badge_h, bx + badge_w, by))

            # Draw sleek dark pill badge
            bx1, by1 = bx, by - badge_h
            bx2, by2 = bx + badge_w, by

            # Dark pill background
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 24, 33), -1)
            # Accent border matching box class color
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color_bgr, 1, cv2.LINE_AA)
            # Crisp text
            cv2.putText(
                img,
                label_text,
                (bx1 + 7, by - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    # 8. Encode final high-quality JPEG
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
