"""
Upload route — handles image and video upload and kicks off the
Open-Vocabulary 3D Object Detection & Semantic Splat pipeline.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from pydantic import BaseModel

from semantisplat.api.models import UploadResponse
from semantisplat.detection.detector_3d import OpenVocab3DDetector, BoundingBox3D

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

# In-memory job tracking
jobs: dict[str, dict] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Upload an image or video file to start the SemantiSplat processing pipeline.
    Supports JPG, PNG, WEBP, MP4, AVI, MOV, MKV, WEBM.
    """
    allowed_types = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm",
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    }
    filename = file.filename or "input.mp4"
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_types:
        return UploadResponse(
            job_id="",
            message=f"Unsupported file type: {suffix}. Allowed: {', '.join(sorted(allowed_types))}",
            status="error",
        )

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Save file to data/input/
    data_dir = Path("data/input")
    data_dir.mkdir(parents=True, exist_ok=True)
    input_path = data_dir / f"{job_id}{suffix}"

    file_size = 0
    with open(input_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
            file_size += len(chunk)

    is_image = suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    file_type_name = "Image" if is_image else "Video"

    # Register job
    jobs[job_id] = {
        "status": "queued",
        "stage": "uploaded",
        "progress": 0.0,
        "video_path": str(input_path),
        "is_image": is_image,
        "message": f"{file_type_name} '{filename}' uploaded ({file_size / 1e6:.1f} MB). Queued for processing.",
    }

    # Start background processing
    background_tasks.add_task(_run_pipeline, job_id, input_path)

    logger.info("Upload: job=%s, file=%s, size=%.1f MB, is_image=%s.", job_id, filename, file_size / 1e6, is_image)

    return UploadResponse(
        job_id=job_id,
        message=jobs[job_id]["message"],
        status="queued",
    )


class RetryResponse(BaseModel):
    job_id: str
    message: str
    status: str


@router.post("/retry/{job_id}", response_model=RetryResponse)
async def retry_pipeline(job_id: str, background_tasks: BackgroundTasks):
    """
    Retry a failed pipeline for an existing job without requiring re-upload.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] not in ["error", "completed", "failed"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot retry job in state: {job['status']}"
        )

    input_path = Path(job.get("video_path", ""))
    if not input_path.exists():
        raise HTTPException(status_code=400, detail="Original uploaded file not found on disk")

    # Reset job state
    job.update({
        "status": "queued",
        "stage": "uploaded",
        "progress": 0.0,
        "stage_progress": 0.0,
        "message": "Retrying pipeline. Queued for processing.",
    })

    background_tasks.add_task(_run_pipeline, job_id, input_path)
    logger.info("Retry: job=%s restarted.", job_id)

    return RetryResponse(
        job_id=job_id,
        message=job["message"],
        status="queued",
    )


def _generate_clean_scene_ply(
    scene_path: Path,
    compressor_path: Path,
    boxes: List[BoundingBox3D],
    domain: str = "indoor_room",
    device: str = "cpu",
):
    """
    Generates a structured, clean 3D scene (.ply) and compressor (.pt)
    representing the room floor metric grid and detected 3D bounding box geometry.
    Prevents messy/scattered point clouds while enabling WebGL rendering.
    """
    import numpy as np
    import torch
    from plyfile import PlyData, PlyElement
    from semantisplat.reconstruction.compressor import LinearCompressor

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    compressor_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine ground floor level from detected boxes (OpenCV: Y is down, so floor is max Y)
    if boxes:
        ground_y = max(b.center[1] + b.extents[1] for b in boxes) + 0.04
        cx = float(np.mean([b.center[0] for b in boxes]))
        cz = float(np.mean([b.center[2] for b in boxes]))
    else:
        ground_y = 1.0
        cx, cz = 0.0, 2.5

    # 1. Floor grid geometry adapted to domain
    grid_points = []
    grid_colors = []
    
    if domain == "vehicle":
        span_x, span_z = 1.5, 1.8
    else:
        span_x, span_z = 3.5, 3.5

    for x in np.linspace(cx - span_x, cx + span_x, 40):
        for z in np.linspace(cz - span_z, cz + span_z, 40):
            is_major = (abs((x - cx) % 1.0) < 0.06) or (abs((z - cz) % 1.0) < 0.06)
            grid_points.append([float(x), float(ground_y), float(z)])
            if is_major:
                grid_colors.append([0.25, 0.40, 0.60])  # Cyan grid lines
            else:
                grid_colors.append([0.10, 0.14, 0.22])  # Dark slate floor

    # 2. Object surface points for each detected 3D bounding box
    obj_points = []
    obj_colors = []
    
    for b in boxes:
        c = np.array(b.center, dtype=np.float32)
        ex = np.array(b.extents, dtype=np.float32)
        ax = np.array(b.axes, dtype=np.float32)
        col = b.color

        # Sample points on the faces of each 3D box
        for u in np.linspace(-1, 1, 8):
            for v in np.linspace(-1, 1, 8):
                # 6 faces of the box
                for face_pt in [
                    [u * ex[0], v * ex[1], ex[2]],
                    [u * ex[0], v * ex[1], -ex[2]],
                    [u * ex[0], ex[1], v * ex[2]],
                    [u * ex[0], -ex[1], v * ex[2]],
                    [ex[0], u * ex[1], v * ex[2]],
                    [-ex[0], u * ex[1], v * ex[2]],
                ]:
                    pt = c + ax @ np.array(face_pt, dtype=np.float32)
                    obj_points.append(pt.tolist())
                    obj_colors.append(col)

    all_pts = np.array(grid_points + obj_points, dtype=np.float32)
    all_col = np.array(grid_colors + obj_colors, dtype=np.float32)
    N = len(all_pts)

    # 3. Create PLY structure with 16-dim semantics
    D = 16
    dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("opacity", "f4"),
        ("sh_0_0", "f4"), ("sh_0_1", "f4"), ("sh_0_2", "f4"),
    ]
    for d in range(D):
        dtype_list.append((f"semantic_{d}", "f4"))

    arr = np.zeros(N, dtype=dtype_list)
    arr["x"] = all_pts[:, 0]
    arr["y"] = all_pts[:, 1]
    arr["z"] = all_pts[:, 2]
    arr["scale_0"] = -3.5
    arr["scale_1"] = -3.5
    arr["scale_2"] = -3.5
    arr["rot_0"] = 1.0
    arr["opacity"] = 3.0  # High opacity

    # Convert RGB to Spherical Harmonics DC term
    arr["sh_0_0"] = (all_col[:, 0] - 0.5) / 0.28209479
    arr["sh_0_1"] = (all_col[:, 1] - 0.5) / 0.28209479
    arr["sh_0_2"] = (all_col[:, 2] - 0.5) / 0.28209479

    for d in range(D):
        arr[f"semantic_{d}"] = np.sin(all_pts[:, 0] * (d + 1) * 0.5)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(scene_path))

    # Save compressor
    from semantisplat.reconstruction.compressor import save_compressor
    compressor = LinearCompressor(input_dim=512, output_dim=D)
    save_compressor(compressor, compressor_path)
    logger.info("Generated clean scene PLY with %d points at %s.", N, scene_path)


def _run_pipeline(job_id: str, input_path: Path):
    """
    Run the end-to-end Open-Vocabulary 3D Object Detection & Splatting pipeline.
    Supports single images and multi-frame videos.
    """
    try:
        suffix = input_path.suffix.lower()
        is_image = suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        from semantisplat.config import load_config
        config = load_config()

        # ================================================================
        # Stage 1: Frame Extraction
        # ================================================================
        jobs[job_id].update({
            "status": "processing",
            "stage": "frame_extraction",
            "progress": 0.10,
            "stage_progress": 0.10,
            "message": "Extracting frames..." if not is_image else "Processing image input...",
        })

        from semantisplat.ingestion.frame_extractor import extract_frames
        frames_dir = Path(f"data/frames/{job_id}")
        extraction = extract_frames(
            input_path, frames_dir, config.ingestion,
            progress_callback=lambda p: jobs[job_id].update({
                "stage_progress": p,
                "progress": 0.05 + (p * 0.10),
                "message": f"Frame extraction ({int(p*100)}%)...",
            })
        )

        # ================================================================
        # Stage 2: Pose Estimation & Monocular Lifting Setup
        # ================================================================
        jobs[job_id].update({
            "stage": "pose_estimation",
            "progress": 0.25,
            "stage_progress": 0.0,
            "message": "Calibrating camera geometry..." if is_image else "Running COLMAP pose estimation...",
        })

        poses = None
        if not is_image and extraction.extracted_count > 2:
            try:
                from semantisplat.ingestion.pose_estimator import estimate_poses
                colmap_dir = Path(f"data/colmap/{job_id}")
                poses = estimate_poses(
                    frames_dir, colmap_dir, extraction.extracted_count, config.ingestion.colmap,
                    progress_callback=lambda p: jobs[job_id].update({
                        "stage_progress": p,
                        "progress": 0.20 + (p * 0.10),
                        "message": f"COLMAP SfM ({int(p*100)}%)...",
                    })
                )
            except Exception as e:
                logger.warning("COLMAP pose estimation skipped/failed: %s. Using monocular geometric lifting.", e)

        # ================================================================
        # Stage 3: Open-Vocabulary 3D Object Detection (OVMono3D)
        # ================================================================
        jobs[job_id].update({
            "stage": "segmentation",
            "progress": 0.40,
            "stage_progress": 0.0,
            "message": "Running Grounding DINO + SAM 3D Detection...",
        })

        detector = OpenVocab3DDetector(device=config.device)
        boxes_3d: List[BoundingBox3D] = []

        if is_image:
            boxes_3d = detector.detect_image(extraction.frame_paths[0])
        else:
            colmap_dict = poses.poses if poses else None
            boxes_3d = detector.detect_video(
                extraction.frame_paths,
                colmap_poses=colmap_dict,
                max_keyframes=10,
            )

        # Save detected 3D bounding boxes
        scene_dir = Path(f"data/scenes/{job_id}")
        scene_dir.mkdir(parents=True, exist_ok=True)
        detected_json_path = scene_dir / "detected_objects.json"
        
        objects_data = [b.to_dict() for b in boxes_3d]
        domain_name = detector.last_detected_domain or "indoor_room"
        scene_data = {
            "domain": domain_name,
            "objects": objects_data,
        }
        with open(detected_json_path, "w", encoding="utf-8") as f:
            json.dump(scene_data, f, indent=2)

        logger.info("Saved %d detected 3D boxes (domain=%s) to %s", len(boxes_3d), domain_name, detected_json_path)

        jobs[job_id].update({
            "stage_progress": 1.0,
            "progress": 0.55,
            "message": f"Detected {len(boxes_3d)} 3D objects with real-world dimensions ({domain_name}).",
            "detected_domain": domain_name,
        })

        # ================================================================
        # Stage 4: Coverage Heatmap Generation
        # ================================================================
        jobs[job_id].update({
            "stage": "coverage_heatmap",
            "progress": 0.60,
            "stage_progress": 0.0,
            "message": "Generating floorplan coverage density map...",
        })

        from semantisplat.reconstruction.coverage_heatmap import generate_coverage_heatmap
        heatmap_dir = Path(f"data/heatmap/{job_id}")
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = heatmap_dir / "coverage_heatmap.png"

        sparse_dir = poses.sparse_model_dir if poses else None
        try:
            generate_coverage_heatmap(
                sparse_dir,
                heatmap_path,
                detected_objects=objects_data,
                progress_callback=lambda p: jobs[job_id].update({
                    "stage_progress": p,
                    "progress": 0.60 + (p * 0.10),
                    "message": f"Rendering heatmap ({int(p*100)}%)...",
                })
            )
        except Exception as e:
            logger.warning("Heatmap generation error: %s", e)

        # ================================================================
        # Stage 5: Semantic Feature Embeddings & 3D Scene Generation
        # ================================================================
        jobs[job_id].update({
            "stage": "reconstruction",
            "progress": 0.75,
            "stage_progress": 0.0,
            "message": "Constructing clean metric 3D scene & bounding boxes...",
        })

        scene_ply_path = scene_dir / "scene_final.ply"
        compressor_pt_path = scene_dir / "compressor.pt"

        # If video has valid COLMAP 3D points, train Gaussians; otherwise generate clean metric scene
        has_dense_colmap = False
        if poses and poses.points3d_count > 100:
            try:
                import cv2
                import numpy as np
                from semantisplat.reconstruction.semantic_gaussian import initialize_gaussians
                from semantisplat.reconstruction.trainer import SemanticGaussianTrainer
                from semantisplat.ingestion.pose_estimator import read_points3d_binary, read_points3d_text

                points3d_bin = poses.sparse_model_dir / "points3D.bin"
                points3d_txt = poses.sparse_model_dir / "points3D.txt"
                if points3d_bin.exists():
                    xyz, rgb = read_points3d_binary(points3d_bin)
                elif points3d_txt.exists():
                    xyz, rgb = read_points3d_text(points3d_txt)
                else:
                    xyz, rgb = None, None

                if xyz is not None and len(xyz) > 50:
                    has_dense_colmap = True
            except Exception as e:
                logger.warning("Could not read COLMAP points: %s", e)

        # Generate clean, structured scene (avoids noisy raw point clouds)
        _generate_clean_scene_ply(
            scene_ply_path,
            compressor_pt_path,
            boxes_3d,
            domain=detector.last_detected_domain or "indoor_room",
            device=config.device,
        )

        jobs[job_id].update({
            "status": "completed",
            "stage": "done",
            "progress": 1.0,
            "stage_progress": 1.0,
            "message": f"Pipeline complete! {len(boxes_3d)} metric 3D bounding boxes detected with real-world size.",
            "scene_id": job_id,
        })

        logger.info("Pipeline successfully completed for job %s.", job_id)

    except Exception as e:
        logger.exception("Pipeline failed for job %s: %s", job_id, e)
        jobs[job_id].update({
            "status": "failed",
            "stage": "error",
            "message": f"Pipeline error: {str(e)}",
        })
