"""
FR-1 (Part B): Camera Pose Estimation via COLMAP.

Wraps the COLMAP CLI to run structure-from-motion on extracted frames,
producing camera poses (rotation + translation) and intrinsics.

Pipeline position: ... → FRAMES → [Pose Estimator / COLMAP] → CAMERA POSES → ...

COLMAP answers: "Where was the camera when each frame was captured?"
"""

from __future__ import annotations

import logging
import shutil
import struct
import subprocess
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from semantisplat.config import ColmapConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]  # focal_x, focal_y, cx, cy, + distortion params


@dataclass
class CameraPose:
    """Camera extrinsic parameters for a single image."""
    image_id: int
    image_name: str
    rotation: np.ndarray       # 4-element quaternion (qw, qx, qy, qz)
    translation: np.ndarray    # 3-element translation vector
    camera_id: int

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        qw, qx, qy, qz = self.rotation
        return np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])

    @property
    def camera_position(self) -> np.ndarray:
        """Compute camera position in world coordinates: C = -R^T * t."""
        R = self.rotation_matrix
        return -R.T @ self.translation


@dataclass
class PoseEstimationResult:
    """Result of COLMAP pose estimation."""
    workspace_dir: Path
    cameras: dict[int, CameraIntrinsics]
    poses: dict[str, CameraPose]      # keyed by image filename
    total_images: int
    registered_images: int
    registration_rate: float
    registration_warning: bool         # True if rate < threshold
    sparse_model_dir: Path
    points3d_count: int


# ============================================================================
# COLMAP binary wrapper
# ============================================================================

def _run_colmap(cmd: list[str], desc: str) -> subprocess.CompletedProcess:
    """Run a COLMAP CLI command with logging."""
    logger.info("Running COLMAP: %s", desc)
    logger.debug("Command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout
    )

    if result.returncode != 0:
        logger.error("COLMAP %s failed:\nstdout: %s\nstderr: %s",
                      desc, result.stdout[-2000:], result.stderr[-2000:])
        raise RuntimeError(f"COLMAP {desc} failed with code {result.returncode}")

    logger.info(f"COLMAP {desc} output:\n{result.stdout}\n{result.stderr}")

    return result


def _check_colmap_available(binary_path: str) -> bool:
    """Check if COLMAP binary is accessible."""
    colmap_path = shutil.which(binary_path)
    if colmap_path is None:
        # Try common Windows locations
        common_paths = [
            Path("C:/Program Files/COLMAP/COLMAP.bat"),
            Path("C:/COLMAP/COLMAP.bat"),
        ]
        for p in common_paths:
            if p.exists():
                return True
        return False
    return True


# ============================================================================
# COLMAP binary model readers (read cameras.bin, images.bin, points3D.bin)
# ============================================================================

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])

CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
}


def _read_cameras_binary(path: Path) -> dict[int, CameraIntrinsics]:
    """Read COLMAP cameras.bin file."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            model_info = CAMERA_MODELS.get(model_id)
            num_params = model_info.num_params if model_info else 4
            model_name = model_info.model_name if model_info else f"UNKNOWN_{model_id}"

            params = list(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))

            cameras[camera_id] = CameraIntrinsics(
                camera_id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=params,
            )

    return cameras


def _read_images_binary(path: Path) -> list[CameraPose]:
    """Read COLMAP images.bin file."""
    poses = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]

            # Quaternion (qw, qx, qy, qz)
            qvec = np.array(struct.unpack("<4d", f.read(32)))

            # Translation (tx, ty, tz)
            tvec = np.array(struct.unpack("<3d", f.read(24)))

            camera_id = struct.unpack("<I", f.read(4))[0]

            # Image name (null-terminated string)
            name_chars = []
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_chars.append(ch.decode("utf-8"))
            image_name = "".join(name_chars)

            # 2D points (we skip them but must read past them)
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            # Each point2D: x, y (double), point3D_id (int64)
            f.read(num_points2d * 24)

            poses.append(CameraPose(
                image_id=image_id,
                image_name=image_name,
                rotation=qvec,
                translation=tvec,
                camera_id=camera_id,
            ))

    return poses


def _count_points3d_binary(path: Path) -> int:
    """Count 3D points in COLMAP points3D.bin."""
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
    return num_points

def read_points3d_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.bin. Returns (xyz, rgb)."""
    xyz, rgb = [], []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            struct.unpack("<Q", f.read(8))  # point3D_id
            xyz.append(struct.unpack("<3d", f.read(24)))
            rgb.append(struct.unpack("<3B", f.read(3)))
            struct.unpack("<d", f.read(8))  # error
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)  # skip track data (image_id, point2D_idx)
            
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32) / 255.0


# ============================================================================
# Text-based model readers (fallback for cameras.txt, images.txt)
# ============================================================================

def _read_cameras_text(path: Path) -> dict[int, CameraIntrinsics]:
    """Read COLMAP cameras.txt file."""
    cameras = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[camera_id] = CameraIntrinsics(
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def _read_images_text(path: Path) -> list[CameraPose]:
    """Read COLMAP images.txt file."""
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # images.txt has pairs of lines: metadata then 2D points
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(parts[j]) for j in range(1, 5)])
        tvec = np.array([float(parts[j]) for j in range(5, 8)])
        camera_id = int(parts[8])
        image_name = parts[9]

        poses.append(CameraPose(
            image_id=image_id,
            image_name=image_name,
            rotation=qvec,
            translation=tvec,
            camera_id=camera_id,
        ))

    return poses

def read_points3d_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.txt. Returns (xyz, rgb)."""
    xyz, rgb = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32) / 255.0


# ============================================================================
# Main pose estimation function
# ============================================================================

def estimate_poses(
    frames_dir: str | Path,
    workspace_dir: str | Path,
    total_images: int,
    config: ColmapConfig | None = None,
    progress_callback = None,
) -> PoseEstimationResult:
    """
    Run COLMAP structure-from-motion on a directory of extracted frames.

    Runs three COLMAP stages:
      1. Feature extraction (SIFT features per image)
      2. Feature matching (exhaustive / sequential)
      3. Sparse reconstruction (incremental mapper)

    Then reads the output model to extract camera poses and intrinsics.

    Args:
        frames_dir: Directory containing extracted video frames.
        workspace_dir: Working directory for COLMAP files.
        total_images: Total number of images in frames_dir (for registration rate).
        config: COLMAP configuration. Uses defaults if None.

    Returns:
        PoseEstimationResult with camera poses, intrinsics, and registration stats.

    Raises:
        RuntimeError: If COLMAP binary is not found or a stage fails.
    """
    if config is None:
        config = ColmapConfig()

    frames_dir = Path(frames_dir)
    workspace_dir = Path(workspace_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Check COLMAP is available
    if not _check_colmap_available(config.binary_path):
        raise RuntimeError(
            f"COLMAP binary not found at '{config.binary_path}'. "
            "Install COLMAP from https://colmap.github.io/ and ensure it's on PATH."
        )

    # Create workspace directories
    database_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    colmap = config.binary_path

    # ---- Stage 1: Feature Extraction ----
    if progress_callback: progress_callback(0.0)
    _run_colmap([
        colmap, "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(frames_dir),
        "--ImageReader.camera_model", config.camera_model,
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1" if config.use_gpu else "0",
    ], desc="feature_extractor")

    # ---- Stage 2: Feature Matching ----
    if progress_callback: progress_callback(0.33)
    if config.matcher_type == "sequential":
        _run_colmap([
            colmap, "sequential_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "1" if config.use_gpu else "0",
        ], desc="sequential_matcher")
    else:
        _run_colmap([
            colmap, "exhaustive_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "1" if config.use_gpu else "0",
        ], desc="exhaustive_matcher")

    # ---- Stage 3: Sparse Reconstruction (Mapper) ----
    if progress_callback: progress_callback(0.66)
    _run_colmap([
        colmap, "mapper",
        "--database_path", str(database_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
    ], desc="mapper")

    # ---- Read the output model ----
    # COLMAP creates numbered subdirectories (0, 1, 2, ...) for each model
    # We use the largest one (most registered images)
    model_dirs = sorted(sparse_dir.iterdir()) if sparse_dir.exists() else []

    if not model_dirs:
        raise RuntimeError("COLMAP produced no sparse models. Check input images.")

    # Pick the model with the most images
    best_model_dir = model_dirs[0]
    best_count = 0

    for md in model_dirs:
        cameras_bin = md / "cameras.bin"
        cameras_txt = md / "cameras.txt"
        images_bin = md / "images.bin"
        images_txt = md / "images.txt"

        if images_bin.exists():
            poses_list = _read_images_binary(images_bin)
        elif images_txt.exists():
            poses_list = _read_images_text(images_txt)
        else:
            continue

        if len(poses_list) > best_count:
            best_count = len(poses_list)
            best_model_dir = md

    # Read the best model
    cameras_bin = best_model_dir / "cameras.bin"
    cameras_txt = best_model_dir / "cameras.txt"
    images_bin = best_model_dir / "images.bin"
    images_txt = best_model_dir / "images.txt"
    points3d_bin = best_model_dir / "points3D.bin"

    if cameras_bin.exists():
        cameras = _read_cameras_binary(cameras_bin)
    elif cameras_txt.exists():
        cameras = _read_cameras_text(cameras_txt)
    else:
        raise RuntimeError(f"No camera model found in {best_model_dir}")

    if images_bin.exists():
        poses_list = _read_images_binary(images_bin)
    elif images_txt.exists():
        poses_list = _read_images_text(images_txt)
    else:
        raise RuntimeError(f"No image poses found in {best_model_dir}")

    points3d_count = _count_points3d_binary(points3d_bin)

    # Build pose dict keyed by image name
    poses_dict = {pose.image_name: pose for pose in poses_list}

    # Registration rate
    registered = len(poses_dict)
    registration_rate = registered / total_images if total_images > 0 else 0.0
    registration_warning = registration_rate < config.registration_threshold

    if registration_warning:
        logger.warning(
            "⚠️  Low registration rate: %d/%d images registered (%.1f%%). "
            "Threshold is %.0f%%. Results may be unreliable. "
            "Consider recapturing with more overlap between views.",
            registered, total_images, registration_rate * 100,
            config.registration_threshold * 100,
        )
    else:
        logger.info(
            "✅ Registration: %d/%d images (%.1f%%) — above %.0f%% threshold.",
            registered, total_images, registration_rate * 100,
            config.registration_threshold * 100,
        )

    logger.info(
        "COLMAP reconstruction: %d cameras, %d poses, %d 3D points.",
        len(cameras), registered, points3d_count,
    )

    logger.info("Pose estimation complete: %d images registered.", registered)

    if progress_callback:
        progress_callback(1.0)

    return PoseEstimationResult(
        workspace_dir=workspace_dir,
        cameras=cameras,
        poses=poses_dict,
        total_images=total_images,
        registered_images=registered,
        registration_rate=registration_rate,
        registration_warning=registration_warning,
        sparse_model_dir=best_model_dir,
        points3d_count=points3d_count,
    )
