"""
FR-2 (Part A): SAM Segmentation — Object/Region Mask Generation.

Uses the Segment Anything Model (SAM) to generate automatic masks
for objects/regions in each extracted frame.

SAM answers: "Which pixels belong to an object/region?"

Pipeline position: ... → FRAMES → [SAM Segmenter] → MASKS → CLIP → ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from semantisplat.config import SAMConfig

logger = logging.getLogger(__name__)


@dataclass
class MaskInfo:
    """A single segmentation mask for an object/region."""
    mask_id: int
    mask: np.ndarray                  # Binary mask (H, W), dtype=bool
    bbox: tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max)
    area: int                         # Number of True pixels
    predicted_iou: float              # SAM's predicted IoU score
    stability_score: float            # Mask stability score
    crop: np.ndarray | None = None    # Cropped RGB region (filled later)


@dataclass
class FrameMasks:
    """All masks extracted from a single frame."""
    frame_path: Path
    frame_index: int
    masks: list[MaskInfo]
    image_height: int
    image_width: int


@dataclass
class SegmentationResult:
    """Result of running SAM across all frames."""
    frame_masks: list[FrameMasks]
    total_masks: int
    frames_processed: int


def _load_sam_model(config: SAMConfig, device: str = "cuda"):
    """
    Load SAM model and create the automatic mask generator.

    Returns:
        (sam_model, mask_generator) tuple.
    """
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    logger.info("Loading SAM model: %s from %s", config.model_type, config.checkpoint)

    checkpoint_path = Path(config.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint_path}. "
            f"Download it from https://github.com/facebookresearch/segment-anything#model-checkpoints"
        )

    sam = sam_model_registry[config.model_type](checkpoint=str(checkpoint_path))
    sam.to(device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=config.points_per_side,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
        min_mask_region_area=config.min_mask_area,
    )

    logger.info("SAM model loaded successfully on %s.", device)
    return sam, mask_generator


def _extract_crop(image: np.ndarray, mask: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Extract the masked RGB crop of an object region.

    The crop is the bounding-box region with non-mask pixels zeroed out.
    This crop is what CLIP will encode to get the semantic embedding.
    """
    x_min, y_min, x_max, y_max = bbox
    crop = image[y_min:y_max, x_min:x_max].copy()
    mask_crop = mask[y_min:y_max, x_min:x_max]

    # Zero out pixels outside the mask
    crop[~mask_crop] = 0

    return crop


def segment_frames(
    frame_paths: list[Path],
    config: SAMConfig | None = None,
    device: str = "cuda",
    extract_crops: bool = True,
    mixed_precision: str = "fp32",
    progress_callback = None,
) -> SegmentationResult:
    """
    Run SAM automatic mask generation on a list of frame images.

    Args:
        frame_paths: List of paths to frame images.
        config: SAM configuration. Uses defaults if None.
        device: Torch device ('cuda' or 'cpu').
        extract_crops: If True, also extract RGB crops for each mask.

    Returns:
        SegmentationResult with masks for all frames.
    """
    if config is None:
        config = SAMConfig()

    _, mask_generator = _load_sam_model(config, device)

    all_frame_masks: list[FrameMasks] = []
    total_masks = 0

    for idx, frame_path in enumerate(tqdm(frame_paths, desc="SAM segmentation")):
        # Read image (BGR -> RGB for SAM)
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            logger.warning("Failed to read frame: %s. Skipping.", frame_path)
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # Generate masks
        with torch.no_grad():
            if mixed_precision != "fp32" and device == "cuda":
                dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    raw_masks = mask_generator.generate(image_rgb)
            else:
                raw_masks = mask_generator.generate(image_rgb)

        # Sort by area (largest first) and cap at max_masks_per_frame
        raw_masks = sorted(raw_masks, key=lambda m: m["area"], reverse=True)
        raw_masks = raw_masks[:config.max_masks_per_frame]

        # Convert to our MaskInfo format
        masks: list[MaskInfo] = []
        for m_idx, raw in enumerate(raw_masks):
            binary_mask = raw["segmentation"].astype(bool)
            bbox_xywh = raw["bbox"]  # SAM gives [x, y, w, h]
            x, y, bw, bh = [int(v) for v in bbox_xywh]
            bbox = (x, y, x + bw, y + bh)

            crop = None
            if extract_crops:
                crop = _extract_crop(image_rgb, binary_mask, bbox)

            masks.append(MaskInfo(
                mask_id=m_idx,
                mask=binary_mask,
                bbox=bbox,
                area=int(raw["area"]),
                predicted_iou=float(raw["predicted_iou"]),
                stability_score=float(raw["stability_score"]),
                crop=crop,
            ))

        all_frame_masks.append(FrameMasks(
            frame_path=frame_path,
            frame_index=idx,
            masks=masks,
            image_height=h,
            image_width=w,
        ))

        total_masks += len(masks)

        if (idx + 1) % 10 == 0:
            logger.info(
                "Processed %d/%d frames, %d masks so far.",
                idx + 1, len(frame_paths), total_masks,
            )
            
        if progress_callback:
            progress_callback((idx + 1) / len(frame_paths))
            
        torch.cuda.empty_cache()

    logger.info(
        "SAM segmentation complete: %d frames, %d total masks (avg %.1f per frame).",
        len(all_frame_masks), total_masks,
        total_masks / max(1, len(all_frame_masks)),
    )

    return SegmentationResult(
        frame_masks=all_frame_masks,
        total_masks=total_masks,
        frames_processed=len(all_frame_masks),
    )
