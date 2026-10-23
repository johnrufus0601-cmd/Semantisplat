"""
FR-1 (Part A): Video Frame Extraction.

Extracts frames from an input video at a configurable FPS rate using OpenCV.

Pipeline position: VIDEO → [Frame Extractor] → FRAMES → COLMAP → ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm

from semantisplat.config import IngestionConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of frame extraction from a video."""
    frame_dir: Path
    frame_paths: list[Path]
    total_frames_in_video: int
    extracted_count: int
    video_fps: float
    video_duration_sec: float
    extraction_fps: float


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    config: IngestionConfig | None = None,
    progress_callback = None,
) -> ExtractionResult:
    """
    Extract frames from a video file at a specified FPS rate.

    Args:
        video_path: Path to the input video file (.mp4, .avi, .mov, etc.).
        output_dir: Directory to save extracted frames.
        config: Ingestion configuration. Uses defaults if None.

    Returns:
        ExtractionResult with paths to all extracted frames and metadata.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened or has no frames.
    """
    if config is None:
        config = IngestionConfig()

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Input file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = video_path.suffix.lower()

    # Support single image inputs directly
    if suffix in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        img = cv2.imread(str(video_path))
        if img is None:
            raise RuntimeError(f"Failed to read image file: {video_path}")

        # Downsample if needed
        if getattr(config, "downsample_resolution", 0) > 0:
            max_dim = config.downsample_resolution
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_path = output_dir / f"frame_000000.{config.frame_format}"
        if config.frame_format == "jpg":
            cv2.imwrite(str(frame_path), img, [cv2.IMWRITE_JPEG_QUALITY, config.frame_quality])
        else:
            cv2.imwrite(str(frame_path), img)

        if progress_callback:
            progress_callback(1.0)

        logger.info("Processed single image input: %s -> %s", video_path.name, frame_path.name)
        return ExtractionResult(
            frame_dir=output_dir,
            frame_paths=[frame_path],
            total_frames_in_video=1,
            extracted_count=1,
            video_fps=1.0,
            video_duration_sec=0.0,
            extraction_fps=1.0,
        )

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or video_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video metadata: fps={video_fps}, frames={total_frames}"
        )

    video_duration = total_frames / video_fps
    extraction_fps = config.frame_extraction_fps

    # Calculate frame interval: extract 1 frame every N video frames
    frame_interval = max(1, int(round(video_fps / extraction_fps)))

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Extracting frames from '%s' | Video: %.1f FPS, %d frames, %.1f sec | "
        "Extraction: %.1f FPS (every %d frames)",
        video_path.name, video_fps, total_frames, video_duration,
        extraction_fps, frame_interval,
    )

    frame_paths: list[Path] = []
    frame_idx = 0
    extracted = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Check max frames limit
                if config.max_frames > 0 and extracted >= config.max_frames:
                    logger.info(
                        "Reached max_frames limit (%d). Stopping extraction.",
                        config.max_frames,
                    )
                    break

                # Downsample if needed
                if getattr(config, "downsample_resolution", 0) > 0:
                    max_dim = config.downsample_resolution
                    h, w = frame.shape[:2]
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Save frame
                frame_name = f"frame_{extracted:06d}.{config.frame_format}"
                frame_path = output_dir / frame_name

                if config.frame_format == "jpg":
                    cv2.imwrite(
                        str(frame_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, config.frame_quality],
                    )
                else:
                    cv2.imwrite(str(frame_path), frame)

                frame_paths.append(frame_path)
                extracted += 1

            frame_idx += 1
            pbar.update(1)
            
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(min(1.0, frame_idx / total_frames))

    if progress_callback:
        progress_callback(1.0)

    cap.release()

    if extracted == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    logger.info(
        "Extraction complete: %d frames saved to '%s'", extracted, output_dir
    )

    return ExtractionResult(
        frame_dir=output_dir,
        frame_paths=frame_paths,
        total_frames_in_video=total_frames,
        extracted_count=extracted,
        video_fps=video_fps,
        video_duration_sec=video_duration,
        extraction_fps=extraction_fps,
    )
