"""
FR-2 (Part B): CLIP Embedding Extraction.

Uses OpenCLIP to encode masked image regions into semantic embeddings.

CLIP answers: "What does this region mean?" by mapping image content
into a shared embedding space with text.

Pipeline position: ... → MASKS → [CLIP Embedder] → EMBEDDINGS → 3D-GS → ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from semantisplat.config import CLIPConfig
from semantisplat.semantic.segmenter import FrameMasks, MaskInfo

logger = logging.getLogger(__name__)


@dataclass
class RegionEmbedding:
    """CLIP embedding for a single masked region."""
    frame_index: int
    mask_id: int
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray   # (embedding_dim,) — e.g. (512,) for ViT-B/32
    area: int


@dataclass
class FrameEmbeddings:
    """All CLIP embeddings for a single frame."""
    frame_path: Path
    frame_index: int
    embeddings: list[RegionEmbedding]


@dataclass
class EmbeddingResult:
    """Result of CLIP encoding across all frames."""
    frame_embeddings: list[FrameEmbeddings]
    total_embeddings: int
    embedding_dim: int
    clip_model_name: str


class CLIPEmbedder:
    """
    Wraps OpenCLIP for encoding image crops and text queries.

    The same CLIP model is used for both:
    - Image encoding (FR-2: semantic feature extraction)
    - Text encoding (FR-4: query processing)

    This is why CLIP works for our pipeline: images and text are mapped
    to the SAME embedding space, enabling cosine similarity comparison.
    """

    def __init__(self, config: CLIPConfig | None = None, device: str = "cuda", mixed_precision: str = "fp32"):
        if config is None:
            config = CLIPConfig()

        self.config = config
        self.device = device
        self.mixed_precision = mixed_precision
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Lazy-load the CLIP model on first use."""
        if self._model is not None:
            return

        import open_clip

        logger.info(
            "Loading CLIP model: %s (pretrained=%s) on %s",
            self.config.model_name, self.config.pretrained, self.device,
        )

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=self.device,
        )
        self._tokenizer = open_clip.get_tokenizer(self.config.model_name)
        self._model.eval()

        logger.info("CLIP model loaded successfully.")

    @torch.no_grad()
    def encode_image_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Encode a batch of image crops (numpy RGB arrays) into CLIP embeddings.

        Args:
            crops: List of RGB numpy arrays (H, W, 3), can be different sizes.

        Returns:
            Normalized embeddings array of shape (N, embedding_dim).
        """
        self._ensure_loaded()

        if not crops:
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)

        # Convert numpy crops to PIL images and preprocess
        processed = []
        for crop in crops:
            pil_img = Image.fromarray(crop)
            processed.append(self._preprocess(pil_img))

        # Batch encode
        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(processed), batch_size):
            batch = torch.stack(processed[i:i + batch_size]).to(self.device)
            if self.mixed_precision != "fp32" and self.device == "cuda":
                dtype = torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    features = self._model.encode_image(batch)
            else:
                features = self._model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize
            all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode text queries into CLIP embeddings.

        Args:
            texts: List of text strings to encode.

        Returns:
            Normalized embeddings array of shape (N, embedding_dim).
        """
        self._ensure_loaded()

        tokens = self._tokenizer(texts).to(self.device)
        features = self._model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

        return features.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text query. Returns shape (embedding_dim,)."""
        return self.encode_text([text])[0]


def extract_embeddings(
    frame_masks_list: list[FrameMasks],
    config: CLIPConfig | None = None,
    device: str = "cuda",
    mixed_precision: str = "fp32",
    progress_callback = None,
) -> EmbeddingResult:
    """
    Run CLIP on all masked regions from SAM to produce semantic embeddings.

    For each frame, for each mask, the masked RGB crop is encoded by CLIP's
    image encoder to produce a semantic feature vector.

    Args:
        frame_masks_list: List of FrameMasks from SAM segmentation.
        config: CLIP configuration. Uses defaults if None.
        device: Torch device.

    Returns:
        EmbeddingResult with embeddings for every region across all frames.
    """
    if config is None:
        config = CLIPConfig()

    embedder = CLIPEmbedder(config, device, mixed_precision)

    all_frame_embeddings: list[FrameEmbeddings] = []
    total_embeddings = 0

    total_frames = len(frame_masks_list)
    for i, fm in enumerate(tqdm(frame_masks_list, desc="CLIP embedding")):
        if progress_callback:
            progress_callback(min(1.0, i / total_frames))
        
        # Collect all crops for this frame
        crops: list[np.ndarray] = []
        valid_masks: list[MaskInfo] = []

        for mask_info in fm.masks:
            if mask_info.crop is not None and mask_info.crop.size > 0:
                # Ensure minimum crop size for CLIP
                h, w = mask_info.crop.shape[:2]
                if h >= 4 and w >= 4:
                    crops.append(mask_info.crop)
                    valid_masks.append(mask_info)

        if not crops:
            all_frame_embeddings.append(FrameEmbeddings(
                frame_path=fm.frame_path,
                frame_index=fm.frame_index,
                embeddings=[],
            ))
            continue

        # Batch encode all crops for this frame
        embeddings_array = embedder.encode_image_crops(crops)

        # Create RegionEmbedding objects
        region_embeddings: list[RegionEmbedding] = []
        for i, mask_info in enumerate(valid_masks):
            region_embeddings.append(RegionEmbedding(
                frame_index=fm.frame_index,
                mask_id=mask_info.mask_id,
                bbox=mask_info.bbox,
                embedding=embeddings_array[i],
                area=mask_info.area,
            ))

        all_frame_embeddings.append(FrameEmbeddings(
            frame_path=fm.frame_path,
            frame_index=fm.frame_index,
            embeddings=region_embeddings,
        ))

        total_embeddings += len(region_embeddings)
        torch.cuda.empty_cache()

    logger.info(
        "CLIP embedding complete: %d frames, %d total embeddings (dim=%d).",
        len(all_frame_embeddings), total_embeddings, config.embedding_dim,
    )

    if progress_callback:
        progress_callback(1.0)

    return EmbeddingResult(
        frame_embeddings=all_frame_embeddings,
        total_embeddings=total_embeddings,
        embedding_dim=embedder.config.embedding_dim,
        clip_model_name=embedder.config.model_name,
    )
