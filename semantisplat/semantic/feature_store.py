"""
FR-2 (Part C): Semantic Feature Storage.

Provides efficient save/load of per-frame SAM masks + CLIP embeddings,
and lookup utilities (given frame + pixel → which mask → which embedding).

Pipeline position: Storage layer between FR-2 outputs and FR-3 training inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from semantisplat.semantic.embedder import EmbeddingResult, FrameEmbeddings, RegionEmbedding
from semantisplat.semantic.segmenter import FrameMasks, MaskInfo, SegmentationResult

logger = logging.getLogger(__name__)


@dataclass
class PixelQuery:
    """Result of querying what semantic region a pixel belongs to."""
    frame_index: int
    pixel_x: int
    pixel_y: int
    mask_id: int | None
    embedding: np.ndarray | None
    area: int | None


class SemanticFeatureStore:
    """
    Persistent storage for per-frame semantic features (masks + embeddings).

    Stores data as .npz files:
    - {frame_idx:06d}_masks.npz  — binary masks, bboxes, metadata
    - {frame_idx:06d}_embeds.npz — CLIP embeddings per mask

    Also provides in-memory lookup for pixel → mask → embedding mapping
    needed during 3D Gaussian Splatting training.
    """

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self._masks_cache: dict[int, FrameMasks] = {}
        self._embeds_cache: dict[int, FrameEmbeddings] = {}

    def save(
        self,
        segmentation: SegmentationResult,
        embeddings: EmbeddingResult,
    ) -> None:
        """
        Save all segmentation masks and CLIP embeddings to disk.

        Args:
            segmentation: SAM segmentation result from segmenter.py.
            embeddings: CLIP embedding result from embedder.py.
        """
        # Build a lookup from frame_index → FrameEmbeddings
        embed_lookup: dict[int, FrameEmbeddings] = {
            fe.frame_index: fe for fe in embeddings.frame_embeddings
        }

        for fm in segmentation.frame_masks:
            idx = fm.frame_index

            # Save masks
            mask_arrays = []
            bboxes = []
            areas = []
            ious = []
            stabilities = []

            for mask_info in fm.masks:
                mask_arrays.append(mask_info.mask.astype(np.uint8))
                bboxes.append(mask_info.bbox)
                areas.append(mask_info.area)
                ious.append(mask_info.predicted_iou)
                stabilities.append(mask_info.stability_score)

            mask_file = self.storage_dir / f"{idx:06d}_masks.npz"
            np.savez_compressed(
                mask_file,
                masks=np.array(mask_arrays) if mask_arrays else np.array([]),
                bboxes=np.array(bboxes) if bboxes else np.array([]),
                areas=np.array(areas),
                ious=np.array(ious),
                stabilities=np.array(stabilities),
                height=fm.image_height,
                width=fm.image_width,
                frame_path=str(fm.frame_path),
            )

            # Save embeddings for this frame
            fe = embed_lookup.get(idx)
            if fe and fe.embeddings:
                embed_arrays = np.array([re.embedding for re in fe.embeddings])
                mask_ids = np.array([re.mask_id for re in fe.embeddings])
                embed_bboxes = np.array([re.bbox for re in fe.embeddings])

                embed_file = self.storage_dir / f"{idx:06d}_embeds.npz"
                np.savez_compressed(
                    embed_file,
                    embeddings=embed_arrays,
                    mask_ids=mask_ids,
                    bboxes=embed_bboxes,
                )

            # Populate cache
            self._masks_cache[idx] = fm
            if fe:
                self._embeds_cache[idx] = fe

        logger.info(
            "Saved semantic features for %d frames to '%s'.",
            len(segmentation.frame_masks), self.storage_dir,
        )

    def load_frame_masks(self, frame_index: int) -> FrameMasks | None:
        """Load masks for a specific frame from disk or cache."""
        if frame_index in self._masks_cache:
            return self._masks_cache[frame_index]

        mask_file = self.storage_dir / f"{frame_index:06d}_masks.npz"
        if not mask_file.exists():
            return None

        data = np.load(mask_file, allow_pickle=True)
        masks_arr = data["masks"]
        bboxes = data["bboxes"]
        areas = data["areas"]
        ious = data["ious"]
        stabilities = data["stabilities"]
        height = int(data["height"])
        width = int(data["width"])
        frame_path = Path(str(data["frame_path"]))

        masks = []
        for i in range(len(masks_arr)):
            masks.append(MaskInfo(
                mask_id=i,
                mask=masks_arr[i].astype(bool),
                bbox=tuple(bboxes[i]) if len(bboxes) > 0 else (0, 0, 0, 0),
                area=int(areas[i]),
                predicted_iou=float(ious[i]),
                stability_score=float(stabilities[i]),
            ))

        fm = FrameMasks(
            frame_path=frame_path,
            frame_index=frame_index,
            masks=masks,
            image_height=height,
            image_width=width,
        )

        self._masks_cache[frame_index] = fm
        return fm

    def load_frame_embeddings(self, frame_index: int) -> list[RegionEmbedding]:
        """Load CLIP embeddings for a specific frame."""
        if frame_index in self._embeds_cache:
            return self._embeds_cache[frame_index].embeddings

        embed_file = self.storage_dir / f"{frame_index:06d}_embeds.npz"
        if not embed_file.exists():
            return []

        data = np.load(embed_file)
        embeddings = data["embeddings"]
        mask_ids = data["mask_ids"]
        bboxes = data["bboxes"]

        result = []
        for i in range(len(embeddings)):
            result.append(RegionEmbedding(
                frame_index=frame_index,
                mask_id=int(mask_ids[i]),
                bbox=tuple(bboxes[i]),
                embedding=embeddings[i],
                area=0,  # Not stored in embed file; available via mask file
            ))

        return result

    def get_pixel_embedding(
        self, frame_index: int, x: int, y: int
    ) -> np.ndarray | None:
        """
        Given a pixel coordinate in a frame, find which mask contains it
        and return the corresponding CLIP embedding.

        This is used during 3D-GS training to supervise semantic features:
        a rendered pixel should have a semantic feature matching the CLIP
        embedding of the SAM region that pixel belongs to.

        Args:
            frame_index: Index of the frame.
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            CLIP embedding vector if pixel is in a mask, else None.
        """
        fm = self.load_frame_masks(frame_index)
        if fm is None:
            return None

        # Find which mask contains this pixel (prefer smaller/more specific masks)
        best_mask_id = None
        best_area = float("inf")

        for mask_info in fm.masks:
            if (0 <= y < mask_info.mask.shape[0] and
                0 <= x < mask_info.mask.shape[1] and
                mask_info.mask[y, x]):
                if mask_info.area < best_area:
                    best_area = mask_info.area
                    best_mask_id = mask_info.mask_id

        if best_mask_id is None:
            return None

        # Find the embedding for this mask
        embeddings = self.load_frame_embeddings(frame_index)
        for re in embeddings:
            if re.mask_id == best_mask_id:
                return re.embedding

        return None

    def get_indexed_embedding_map(
        self, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Build a memory-efficient indexed embedding map for a frame.

        Returns:
            - mask_map: (H, W) array of np.int16 where each pixel contains an index [0, K-1],
              or -1 if the pixel does not belong to any mask.
            - unique_embeddings: (K, D) array of unique CLIP embeddings.
            Returns (None, None) if the frame is not found.
        """
        fm = self.load_frame_masks(frame_index)
        if fm is None:
            return None, None

        embeddings = self.load_frame_embeddings(frame_index)
        if not embeddings:
            return None, None

        embed_dim = embeddings[0].embedding.shape[0]
        mask_map = np.full((fm.image_height, fm.image_width), -1, dtype=np.int16)

        unique_embeddings = []
        mask_id_to_idx = {}

        for re in embeddings:
            mask_id_to_idx[re.mask_id] = len(unique_embeddings)
            unique_embeddings.append(re.embedding)

        # Process masks from largest to smallest so smaller masks overwrite
        sorted_masks = sorted(fm.masks, key=lambda m: m.area, reverse=True)

        for mask_info in sorted_masks:
            if mask_info.mask_id in mask_id_to_idx:
                idx = mask_id_to_idx[mask_info.mask_id]
                mask_map[mask_info.mask] = idx

        return mask_map, np.array(unique_embeddings, dtype=np.float32)

    def list_frame_indices(self) -> list[int]:
        """List all frame indices that have stored semantic data."""
        indices = set()
        for f in self.storage_dir.glob("*_masks.npz"):
            try:
                idx = int(f.stem.split("_")[0])
                indices.add(idx)
            except ValueError:
                continue
        return sorted(indices)
