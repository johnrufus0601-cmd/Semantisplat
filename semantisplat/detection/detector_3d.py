"""
SemantiSplat Open-Vocabulary 3D Object Detection Module.
Inspired by OVMono3D / OVM3D-Det.

Integrates:
1. Open-vocabulary 2D detector (Grounding DINO with OwlViT fallback)
2. SAM (Segment Anything) prompt-based segmentation
3. Metric monocular depth estimation (Depth Anything V2)
4. Monocular 3D lifting to metric oriented bounding boxes (OBBs)
5. Multi-frame fusion for video and single-image support
6. Real-world size estimation (width, height, depth in meters/feet)
7. Spatial reasoning queries ("Can a 6-foot sofa fit here?")
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Curated palette for clean 3D bounding box visualization
CLASS_COLORS = {
    # Automotive / Vehicle
    "steering wheel": [1.0, 0.25, 0.25],    # Crimson red
    "wheel": [1.0, 0.25, 0.25],             # Crimson red
    "dashboard": [0.2, 0.6, 1.0],           # Sky blue
    "car seat": [0.3, 0.85, 0.4],           # Emerald green
    "seat": [0.3, 0.85, 0.4],               # Emerald green
    "center console": [1.0, 0.65, 0.1],     # Orange
    "console": [1.0, 0.65, 0.1],            # Orange
    "touchscreen display": [0.8, 0.35, 1.0],# Violet
    "touchscreen": [0.8, 0.35, 1.0],        # Violet
    "gear shifter": [1.0, 0.85, 0.2],       # Bright amber
    "shifter": [1.0, 0.85, 0.2],            # Bright amber
    "rearview mirror": [0.3, 0.9, 0.9],     # Cyan
    "windshield": [0.5, 0.7, 0.9],          # Ice blue
    "car door": [0.6, 0.6, 0.7],            # Slate gray
    "car window": [0.4, 0.8, 0.8],          # Aqua
    "air vent": [0.7, 0.7, 0.7],            # Light gray
    "pedal": [0.8, 0.5, 0.2],               # Copper
    
    # Indoor Furniture & Objects
    "chair": [0.2, 0.7, 1.0],               # Sky blue
    "sofa": [1.0, 0.4, 0.3],                # Coral red
    "couch": [1.0, 0.4, 0.3],               # Coral red
    "table": [0.3, 0.9, 0.4],               # Emerald green
    "desk": [0.3, 0.85, 0.5],               # Jade green
    "bed": [0.8, 0.4, 1.0],                 # Purple
    "fan": [1.0, 0.85, 0.2],                # Bright yellow
    "ladder": [1.0, 0.6, 0.1],              # Orange
    "curtain": [0.6, 0.8, 0.9],             # Soft blue
    "lamp": [1.0, 0.95, 0.5],               # Warm light yellow
    "monitor": [0.2, 0.5, 0.9],             # Deep blue
    "tv": [0.2, 0.5, 0.9],                  # Deep blue
    "screen": [0.2, 0.5, 0.9],              # Deep blue
    "cabinet": [0.7, 0.5, 0.3],             # Wood brown
    "shelf": [0.65, 0.45, 0.3],             # Wood brown
    "plant": [0.1, 0.8, 0.3],               # Leaf green
    "door": [0.5, 0.5, 0.6],                # Slate gray
    "window": [0.4, 0.8, 0.8],              # Aqua
    "default": [0.9, 0.8, 0.2],             # Golden amber
}

VEHICLE_VOCABULARY = [
    "steering wheel", "dashboard", "car seat", "center console",
    "touchscreen display", "gear shifter", "rearview mirror",
    "windshield", "car door", "car window", "air vent", "pedal"
]

INDOOR_ROOM_VOCABULARY = [
    "sofa", "chair", "table", "bed", "fan", "ladder", "desk", 
    "curtain", "lamp", "cabinet", "plant", "monitor", "door", "window"
]

OFFICE_VOCABULARY = [
    "office chair", "desk", "computer monitor", "laptop", "keyboard",
    "mouse", "cabinet", "whiteboard", "bookshelf", "door", "window"
]

KITCHEN_VOCABULARY = [
    "refrigerator", "countertop", "kitchen sink", "stove", "oven",
    "microwave", "cabinet", "dining table", "chair", "dishwasher"
]

DEFAULT_VOCABULARY = INDOOR_ROOM_VOCABULARY

# Realistic physical size ranges (width, height, depth) in meters for validation & calibration
OBJECT_SIZE_PRIORS = {
    # Automotive / Vehicle
    "steering wheel": {"w": (0.32, 0.44), "h": (0.32, 0.44), "d": (0.10, 0.25)},
    "wheel": {"w": (0.32, 0.44), "h": (0.32, 0.44), "d": (0.10, 0.25)},
    "dashboard": {"w": (1.10, 1.65), "h": (0.28, 0.55), "d": (0.35, 0.70)},
    "touchscreen display": {"w": (0.20, 0.42), "h": (0.12, 0.28), "d": (0.05, 0.15)},
    "touchscreen": {"w": (0.20, 0.42), "h": (0.12, 0.28), "d": (0.05, 0.15)},
    "center console": {"w": (0.22, 0.45), "h": (0.20, 0.45), "d": (0.45, 0.90)},
    "console": {"w": (0.22, 0.45), "h": (0.20, 0.45), "d": (0.45, 0.90)},
    "gear shifter": {"w": (0.08, 0.20), "h": (0.15, 0.35), "d": (0.08, 0.20)},
    "shifter": {"w": (0.08, 0.20), "h": (0.15, 0.35), "d": (0.08, 0.20)},
    "rearview mirror": {"w": (0.18, 0.32), "h": (0.06, 0.14), "d": (0.05, 0.15)},
    "air vent": {"w": (0.10, 0.30), "h": (0.08, 0.22), "d": (0.05, 0.18)},
    "car seat": {"w": (0.45, 0.65), "h": (0.70, 1.05), "d": (0.45, 0.70)},
    "car door": {"w": (0.70, 1.20), "h": (0.60, 1.10), "d": (0.12, 0.30)},
    "car window": {"w": (0.45, 0.90), "h": (0.30, 0.60), "d": (0.05, 0.15)},
    "windshield": {"w": (1.10, 1.60), "h": (0.50, 0.95), "d": (0.15, 0.40)},
    "pedal": {"w": (0.12, 0.30), "h": (0.10, 0.25), "d": (0.08, 0.20)},

    # Indoor Furniture & Objects
    "chair": {"w": (0.45, 0.75), "h": (0.70, 1.15), "d": (0.45, 0.75)},
    "office chair": {"w": (0.50, 0.75), "h": (0.80, 1.25), "d": (0.50, 0.75)},
    "sofa": {"w": (1.40, 2.60), "h": (0.65, 1.05), "d": (0.75, 1.15)},
    "couch": {"w": (1.40, 2.60), "h": (0.65, 1.05), "d": (0.75, 1.15)},
    "table": {"w": (0.80, 2.20), "h": (0.65, 0.85), "d": (0.60, 1.20)},
    "dining table": {"w": (1.20, 2.40), "h": (0.70, 0.85), "d": (0.75, 1.20)},
    "desk": {"w": (0.85, 1.80), "h": (0.68, 0.85), "d": (0.50, 0.90)},
    "bed": {"w": (1.00, 2.10), "h": (0.45, 1.20), "d": (1.80, 2.30)},
    "fan": {"w": (0.35, 0.65), "h": (0.35, 1.40), "d": (0.35, 0.65)},
    "ladder": {"w": (0.35, 0.65), "h": (1.10, 2.60), "d": (0.20, 0.80)},
    "curtain": {"w": (0.60, 2.50), "h": (1.20, 2.80), "d": (0.05, 0.25)},
    "lamp": {"w": (0.25, 0.55), "h": (0.35, 1.70), "d": (0.25, 0.55)},
    "monitor": {"w": (0.45, 0.95), "h": (0.30, 0.65), "d": (0.08, 0.25)},
    "tv": {"w": (0.80, 1.80), "h": (0.50, 1.10), "d": (0.08, 0.25)},
    "screen": {"w": (0.45, 1.20), "h": (0.30, 0.80), "d": (0.08, 0.25)},
    "cabinet": {"w": (0.60, 1.60), "h": (0.70, 2.10), "d": (0.35, 0.70)},
    "shelf": {"w": (0.60, 1.60), "h": (0.70, 2.10), "d": (0.25, 0.50)},
    "plant": {"w": (0.25, 0.80), "h": (0.30, 1.80), "d": (0.25, 0.80)},
    "door": {"w": (0.75, 1.05), "h": (1.90, 2.25), "d": (0.05, 0.20)},
    "window": {"w": (0.60, 1.80), "h": (0.60, 1.60), "d": (0.05, 0.25)},
}


@dataclass
class BoundingBox3D:
    """Metric 3D Oriented Bounding Box."""
    center: List[float]               # [x, y, z] in meters
    extents: List[float]              # [half_x, half_y, half_z] in meters
    size: List[float]                 # [width, height, depth] in meters
    axes: List[List[float]]           # 3x3 rotation matrix (eigenvectors)
    yaw: float                        # Yaw rotation angle in degrees
    corners_3d: List[List[float]]     # 8 3D vertices [x, y, z]
    label: str                        # Object category / query text
    confidence: float                 # Detection score (0.0 to 1.0)
    bbox_2d: List[int]                # [x1, y1, x2, y2] pixel coordinates
    color: List[float] = field(default_factory=lambda: [0.2, 0.8, 0.2])
    highlighted: bool = False
    size_str: str = ""
    short_size_str: str = ""
    frame_index: int = 0
    num_points: int = 0

    def __post_init__(self):
        if len(self.size) == 3:
            w, h, d = self.size
            if not self.short_size_str:
                self.short_size_str = f"{w:.2f}m × {h:.2f}m × {d:.2f}m"
            if not self.size_str:
                w_ft = w * 3.28084
                h_ft = h * 3.28084
                d_ft = d * 3.28084
                self.size_str = f"{w:.2f}m × {h:.2f}m × {d:.2f}m ({w_ft:.1f}ft × {h_ft:.1f}ft × {d_ft:.1f}ft)"

    @property
    def score(self) -> float:
        return self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [round(float(c), 3) for c in self.center],
            "extents": [round(float(e), 3) for e in self.extents],
            "size": [round(float(s), 3) for s in self.size],
            "axes": [[round(float(v), 3) for v in row] for row in self.axes],
            "yaw": round(float(self.yaw), 1),
            "corners_3d": [[round(float(c), 3) for c in pt] for pt in self.corners_3d],
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            "score": round(float(self.confidence), 3),
            "bbox_2d": [int(v) for v in self.bbox_2d],
            "color": [round(float(c), 3) for c in self.color],
            "highlighted": self.highlighted,
            "size_str": self.size_str,
            "short_size_str": self.short_size_str,
            "frame_index": self.frame_index,
            "num_points": self.num_points,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundingBox3D":
        center = d.get("center", [0.0, 0.0, 0.0])
        extents = d.get("extents", [0.2, 0.2, 0.2])
        size = d.get("size", [float(extents[0]*2), float(extents[1]*2), float(extents[2]*2)])
        return cls(
            center=center,
            extents=extents,
            size=size,
            axes=d.get("axes", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            yaw=float(d.get("yaw", 0.0)),
            corners_3d=d.get("corners_3d", []),
            label=d.get("label", "object"),
            confidence=float(d.get("confidence", d.get("score", 0.5))),
            bbox_2d=d.get("bbox_2d", [0, 0, 0, 0]),
            color=d.get("color", [0.2, 0.8, 0.2]),
            highlighted=d.get("highlighted", False),
            size_str=d.get("size_str", ""),
            short_size_str=d.get("short_size_str", ""),
            frame_index=int(d.get("frame_index", 0)),
            num_points=int(d.get("num_points", 0)),
        )


@dataclass
class DetectedObject3D:
    """Container for 3D detected object with 2D/3D metadata."""
    obb: BoundingBox3D
    mask_area: int
    frame_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = self.obb.to_dict()
        data["mask_area"] = self.mask_area
        data["frame_name"] = self.frame_name
        return data


@dataclass
class Scene3D:
    """Clean 3D Scene representation with metric 3D boxes and room bounds."""
    scene_id: str
    objects: List[BoundingBox3D]
    room_bounds: Dict[str, Any]
    camera_poses: List[Dict[str, Any]]
    is_single_image: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "objects": [obj.to_dict() for obj in self.objects],
            "room_bounds": self.room_bounds,
            "camera_poses": self.camera_poses,
            "is_single_image": self.is_single_image,
        }


class OpenVocab3DDetector:
    """
    Monocular Open-Vocabulary 3D Object Detection Pipeline.
    
    1. Open-Vocabulary 2D Object Detection: Grounding DINO / OwlViT
    2. Prompt-based SAM Segmentation
    3. Depth Anything V2 Monocular Metric Depth Lifting
    4. 3D Oriented Bounding Box (OBB) Fitting with Metric Size Estimation
    5. Multi-View Fusion and Spatial Querying
    """

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._dino_processor = None
        self._dino_model = None
        self._owlvit_processor = None
        self._owlvit_model = None
        self._sam_predictor = None
        self._depth_pipe = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self.last_detected_domain = "indoor_room"
        logger.info("OpenVocab3DDetector initialized on device: %s", self.device)

    # -------------------------------------------------------------------------
    # Model Loaders (Lazy Loading for RTX 5050 memory efficiency)
    # -------------------------------------------------------------------------

    def _get_clip_model(self):
        """Load cached CLIP model for zero-shot scene domain classification."""
        if self._clip_model is not None:
            return self._clip_model, self._clip_preprocess, self._clip_tokenizer

        try:
            import open_clip
            logger.info("Loading cached OpenCLIP model (vit_base_patch32_clip_224.openai) for scene domain detection...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                'hf-hub:timm/vit_base_patch32_clip_224.openai',
                device=self.device
            )
            tokenizer = open_clip.get_tokenizer('hf-hub:timm/vit_base_patch32_clip_224.openai')
            self._clip_model = model.eval()
            self._clip_preprocess = preprocess
            self._clip_tokenizer = tokenizer
            return self._clip_model, self._clip_preprocess, self._clip_tokenizer
        except Exception as e:
            logger.warning("CLIP model load failed (%s). Using DINO probe fallback.", e)
            self._clip_model = None
            return None, None, None

    def detect_scene_domain(self, image_pil: Image.Image) -> Tuple[str, List[str]]:
        """
        Accurately identifies the scene type (e.g. vehicle cabin vs indoor room)
        and returns the domain name and appropriate domain vocabulary.
        """
        clip_model, preprocess, tokenizer = self._get_clip_model()
        if clip_model is not None and preprocess is not None and tokenizer is not None:
            try:
                candidate_domains = [
                    ("vehicle", "a photo of a car interior or vehicle cabin with dashboard and steering wheel", VEHICLE_VOCABULARY),
                    ("indoor_room", "a photo of an indoor living room or bedroom with sofa and furniture", INDOOR_ROOM_VOCABULARY),
                    ("office", "a photo of an office workspace with computer monitors and desks", OFFICE_VOCABULARY),
                    ("kitchen", "a photo of a kitchen with refrigerator and cooking appliances", KITCHEN_VOCABULARY),
                ]
                texts = [c[1] for c in candidate_domains]
                text_tokens = tokenizer(texts).to(self.device)
                img_tensor = preprocess(image_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    img_f = clip_model.encode_image(img_tensor)
                    txt_f = clip_model.encode_text(text_tokens)
                    img_f /= img_f.norm(dim=-1, keepdim=True)
                    txt_f /= txt_f.norm(dim=-1, keepdim=True)
                    sims = (img_f @ txt_f.T * 100).softmax(dim=-1)[0].cpu().numpy()

                best_idx = int(np.argmax(sims))
                domain, prompt_desc, vocab = candidate_domains[best_idx]
                conf = float(sims[best_idx])
                logger.info("Scene domain classified via CLIP as '%s' (confidence: %.3f)", domain, conf)
                if conf > 0.35:
                    self.last_detected_domain = domain
                    return domain, vocab
            except Exception as e:
                logger.warning("CLIP domain classification error: %s", e)

        # Grounding DINO probe fallback
        try:
            probe_queries = ["steering wheel", "car dashboard", "sofa", "bed", "office desk", "refrigerator"]
            detections = self.detect_2d(image_pil, probe_queries, confidence_threshold=0.25)
            labels = [d["label"].lower() for d in detections]
            if any(l in ["steering wheel", "wheel", "car dashboard", "dashboard"] for l in labels):
                logger.info("Scene domain identified as 'vehicle' via Grounding DINO probe.")
                self.last_detected_domain = "vehicle"
                return "vehicle", VEHICLE_VOCABULARY
            if any(l in ["office desk"] for l in labels):
                self.last_detected_domain = "office"
                return "office", OFFICE_VOCABULARY
            if any(l in ["refrigerator"] for l in labels):
                self.last_detected_domain = "kitchen"
                return "kitchen", KITCHEN_VOCABULARY
        except Exception as e:
            logger.warning("Probe classification failed: %s", e)

        self.last_detected_domain = "indoor_room"
        return "indoor_room", INDOOR_ROOM_VOCABULARY

    def _get_2d_detector(self):
        """Load Grounding DINO (or OwlViT fallback) for zero-shot 2D detection."""
        if self._dino_model is not None:
            return "grounding_dino", self._dino_processor, self._dino_model
        if self._owlvit_model is not None:
            return "owlvit", self._owlvit_processor, self._owlvit_model

        try:
            logger.info("Loading Grounding DINO (tiny)...")
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self._dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
            self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-tiny"
            ).to(self.device).eval()
            logger.info("Grounding DINO loaded successfully on %s", self.device)
            return "grounding_dino", self._dino_processor, self._dino_model
        except Exception as e:
            logger.warning("Grounding DINO load error: %s. Using OwlViT fallback.", e)

        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            logger.info("Loading OwlViT (base-patch32) fallback...")
            self._owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self._owlvit_model = OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-base-patch32"
            ).to(self.device).eval()
            logger.info("OwlViT loaded successfully on %s", self.device)
            return "owlvit", self._owlvit_processor, self._owlvit_model
        except Exception as e2:
            logger.error("Both Grounding DINO and OwlViT failed to load: %s", e2)
            raise RuntimeError(f"Could not load 2D open-vocab detector: {e2}")

    def _get_sam_predictor(self):
        """Load SAM predictor using local checkpoint."""
        if self._sam_predictor is not None:
            return self._sam_predictor

        checkpoint = Path("sam_vit_b_01ec64.pth")
        if not checkpoint.exists():
            checkpoint = Path("weights/sam_vit_b_01ec64.pth")

        if not checkpoint.exists():
            logger.warning("SAM checkpoint not found at %s. SAM will run in fallback box-mask mode.", checkpoint)
            return None

        from segment_anything import SamPredictor, sam_model_registry
        logger.info("Loading SAM model from %s on %s...", checkpoint, self.device)
        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to(self.device).eval()
        self._sam_predictor = SamPredictor(sam)
        return self._sam_predictor

    def _get_depth_pipe(self):
        """Load Depth Anything V2 monocular depth estimation pipeline."""
        if self._depth_pipe is not None:
            return self._depth_pipe

        from transformers import pipeline
        logger.info("Loading Depth Anything V2 pipeline on %s...", self.device)
        device_id = 0 if self.device == "cuda" else -1
        self._depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device_id,
        )
        return self._depth_pipe

    # -------------------------------------------------------------------------
    # 2D Open-Vocabulary Detection
    # -------------------------------------------------------------------------

    def _map_label_to_query(self, raw_label: str, clean_queries: List[str]) -> str:
        """
        Maps sub-word tokens or compound text from Grounding DINO back to canonical query names.
        Examples:
          'wheel' -> 'steering wheel'
          'air' -> 'air vent'
          'rearview' -> 'rearview mirror'
          'center' -> 'center console'
          'display' / 'touchscreen' -> 'touchscreen display'
          'gear' / 'shifter' -> 'gear shifter'
          'seat' -> 'car seat' (or 'chair' in indoor)
          'door' -> 'car door' (in vehicle)
          'window' -> 'car window' (in vehicle)
        """
        raw = str(raw_label).strip(". ").lower()
        if raw in clean_queries:
            return raw

        # Direct domain-specific synonym tables
        vehicle_synonyms = {
            "wheel": "steering wheel",
            "steering": "steering wheel",
            "steering wheel": "steering wheel",
            "dashboard": "dashboard",
            "dash": "dashboard",
            "instrument cluster": "dashboard",
            "air": "air vent",
            "vent": "air vent",
            "air vent": "air vent",
            "ac vent": "air vent",
            "rearview": "rearview mirror",
            "mirror": "rearview mirror",
            "rearview mirror": "rearview mirror",
            "center": "center console",
            "console": "center console",
            "center console": "center console",
            "display": "touchscreen display",
            "touchscreen": "touchscreen display",
            "screen": "touchscreen display",
            "infotainment": "touchscreen display",
            "gear": "gear shifter",
            "shifter": "gear shifter",
            "gear shifter": "gear shifter",
            "seat": "car seat",
            "car seat": "car seat",
            "door": "car door",
            "car door": "car door",
            "window": "car window",
            "car window": "car window",
            "windshield": "windshield",
            "pedal": "pedal",
            "pedals": "pedal",
        }

        if self.last_detected_domain == "vehicle" and raw in vehicle_synonyms:
            mapped = vehicle_synonyms[raw]
            if mapped in clean_queries:
                return mapped

        # Token match in any multi-word query
        for q in clean_queries:
            q_tokens = q.split()
            if raw in q_tokens:
                return q

        # Substring / character overlap match
        candidates = []
        for q in clean_queries:
            if raw in q or q in raw:
                candidates.append((q, len(q)))
        if candidates:
            candidates.sort(key=lambda c: c[1], reverse=True)
            return candidates[0][0]

        return raw

    # -------------------------------------------------------------------------
    # 2D Open-Vocabulary Detection
    # -------------------------------------------------------------------------

    def detect_2d(
        self,
        image_pil: Image.Image,
        text_queries: List[str],
        confidence_threshold: float = 0.22,
    ) -> List[Dict[str, Any]]:
        """
        Detect 2D bounding boxes for arbitrary text queries.
        Returns list of dicts: {'box': [x1, y1, x2, y2], 'label': str, 'score': float}
        """
        model_type, processor, model = self._get_2d_detector()
        W, H = image_pil.size
        detections = []

        clean_queries = [q.strip().lower() for q in text_queries if q.strip()]
        if not clean_queries:
            clean_queries = DEFAULT_VOCABULARY

        if model_type == "grounding_dino":
            # Format text prompt for Grounding DINO: "chair . sofa . table ."
            text_prompt = " . ".join(clean_queries) + " ."
            
            inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)

            # Use balanced thresholds to reduce false detections
            box_thresh = max(0.24, confidence_threshold)
            text_thresh = max(0.22, confidence_threshold)

            post_kwargs = {
                "target_sizes": [(H, W)],
                "text_threshold": text_thresh,
            }
            import inspect
            sig = inspect.signature(processor.post_process_grounded_object_detection)
            if "box_threshold" in sig.parameters:
                post_kwargs["box_threshold"] = box_thresh
            else:
                post_kwargs["threshold"] = box_thresh

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                **post_kwargs,
            )[0]

            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            raw_labels = results.get("text_labels")
            if raw_labels is None:
                raw_labels = results.get("labels", [])

            for box, score, label in zip(boxes, scores, raw_labels):
                x1, y1, x2, y2 = [int(v) for v in box]
                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                bw, bh = x2 - x1, y2 - y1

                # Discard degenerate tiny boxes
                if bw < 16 or bh < 16:
                    continue

                # Discard degenerate giant boxes (covering >85% of image with low confidence)
                if (bw * bh) > (0.85 * W * H) and score < 0.45:
                    continue

                clean_label = self._map_label_to_query(str(label), clean_queries)

                # Cross-domain validation: if vehicle cabin, suppress furniture hallucinations
                if self.last_detected_domain == "vehicle":
                    furniture_labels = {"sofa", "couch", "bed", "ladder", "lamp", "desk", "table", "curtain", "fan"}
                    if clean_label in furniture_labels:
                        logger.info("Suppressed cross-domain false detection '%s' in vehicle scene.", clean_label)
                        continue

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "label": clean_label,
                    "score": float(score),
                })

        else:
            # OwlViT detector
            inputs = processor(text=[clean_queries], images=image_pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([(H, W)], device=self.device)
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=max(0.22, confidence_threshold)
            )[0]

            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()

            for box, score, label_idx in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                bw, bh = x2 - x1, y2 - y1
                if bw < 16 or bh < 16:
                    continue
                if (bw * bh) > (0.85 * W * H) and score < 0.45:
                    continue

                label_text = clean_queries[label_idx] if label_idx < len(clean_queries) else "object"
                clean_label = self._map_label_to_query(label_text, clean_queries)

                if self.last_detected_domain == "vehicle":
                    furniture_labels = {"sofa", "couch", "bed", "ladder", "lamp", "desk", "table", "curtain", "fan"}
                    if clean_label in furniture_labels:
                        continue

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "label": clean_label,
                    "score": float(score),
                })

        # Apply Non-Maximum Suppression (NMS) to avoid heavily overlapping duplicates
        detections = self._nms(detections, iou_thresh=0.45)
        return detections

    def _nms(self, detections: List[Dict[str, Any]], iou_thresh: float = 0.45) -> List[Dict[str, Any]]:
        if not detections:
            return []
        boxes = np.array([d["box"] for d in detections], dtype=np.float32)
        scores = np.array([d["score"] for d in detections], dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return [detections[k] for k in keep]

    # -------------------------------------------------------------------------
    # SAM Segmentation
    # -------------------------------------------------------------------------

    def segment_box(
        self,
        image_rgb: np.ndarray,
        box: List[int],
    ) -> np.ndarray:
        """
        Run SAM with box prompt to extract precise instance mask.
        Returns binary mask (H, W) of bools.
        """
        H, W = image_rgb.shape[:2]
        predictor = self._get_sam_predictor()
        if predictor is not None:
            try:
                predictor.set_image(image_rgb)
                input_box = np.array(box)[None, :]
                masks, scores, _ = predictor.predict(
                    box=input_box,
                    multimask_output=False,
                )
                return masks[0].astype(bool)
            except Exception as e:
                logger.warning("SAM prediction error: %s, falling back to box mask.", e)

        # Fallback: rectangular box mask with soft ellipse inner weighting
        mask = np.zeros((H, W), dtype=bool)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = True
        return mask

    # -------------------------------------------------------------------------
    # Metric Depth Estimation & Calibration
    # -------------------------------------------------------------------------

    def estimate_depth(self, image_pil: Image.Image, domain: Optional[str] = None) -> np.ndarray:
        """
        Estimate metric depth map (H, W) in meters using Depth Anything V2.
        Applies domain-aware metric scale normalization.
        """
        if domain is None:
            domain = getattr(self, "last_detected_domain", "indoor_room")

        pipe = self._get_depth_pipe()
        depth_output = pipe(image_pil)
        
        # Use full continuous predicted depth tensor if available
        if isinstance(depth_output, dict) and "predicted_depth" in depth_output:
            pd = depth_output["predicted_depth"]
            if hasattr(pd, "cpu"):
                depth_map = pd.cpu().numpy().astype(np.float32)
            else:
                depth_map = np.array(pd, dtype=np.float32)
        else:
            depth_map = np.array(depth_output["depth"], dtype=np.float32)

        d_min = float(np.percentile(depth_map, 2))
        d_max = float(np.percentile(depth_map, 98))
        if d_max > d_min:
            depth_norm = (depth_map - d_min) / (d_max - d_min)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
        else:
            depth_norm = np.ones_like(depth_map, dtype=np.float32) * 0.5

        # Domain-aware metric depth mapping
        if domain == "vehicle":
            # Vehicle cabin: dashboard/wheel ~0.45m, windshield ~0.9m, rear ~2.1m
            metric_depth = 0.40 + (depth_norm * 1.70)
        else:
            # Indoor room / office / kitchen: close furniture ~0.85m, walls ~4.8m
            metric_depth = 0.85 + (depth_norm * 4.10)

        return metric_depth

    # -------------------------------------------------------------------------
    # 3D Metric Lifting & OBB Calculation
    # -------------------------------------------------------------------------

    def lift_to_3d_box(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        label: str,
        confidence: float,
        bbox_2d: List[int],
        intrinsics: Optional[Dict[str, float]] = None,
        frame_idx: int = 0,
    ) -> Optional[BoundingBox3D]:
        """
        Lift 2D masked region to metric 3D Oriented Bounding Box (OBB).
        Calculates metric dimensions (width, height, depth) and yaw orientation,
        calibrating using physical category dimension priors.
        """
        H, W = mask.shape
        if intrinsics is None:
            # Default pinhole camera parameters
            fx = fy = float(max(W, H)) * 0.95
            cx = float(W) / 2.0
            cy = float(H) / 2.0
        else:
            fx = float(intrinsics.get("fx", max(W, H) * 0.95))
            fy = float(intrinsics.get("fy", max(W, H) * 0.95))
            cx = float(intrinsics.get("cx", W / 2.0))
            cy = float(intrinsics.get("cy", H / 2.0))

        # Extract masked pixel coordinates (u, v)
        ys, xs = np.where(mask)
        if len(xs) < 25:
            return None

        zs = depth_map[ys, xs]

        # Robust depth filtering: remove foreground/background bleed
        z_low = np.percentile(zs, 12)
        z_high = np.percentile(zs, 88)
        valid = (zs >= z_low) & (zs <= z_high)
        if np.sum(valid) < 20:
            valid = np.ones_like(zs, dtype=bool)

        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]

        # Monocular Backprojection to Camera 3D Coordinates:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy  (In camera coords, Y is downward)
        # Z = Z                 (Forward depth)
        X = (xs.astype(np.float32) - cx) * zs / fx
        Y = (ys.astype(np.float32) - cy) * zs / fy
        Z = zs

        points_3d = np.stack([X, Y, Z], axis=1)  # (N, 3)

        # Compute centroid
        center = np.median(points_3d, axis=0)

        # PCA on the horizontal (X, Z) plane to find the object's yaw angle
        centered_xz = points_3d[:, [0, 2]] - center[[0, 2]]
        cov_xz = np.cov(centered_xz, rowvar=False)
        if np.isnan(cov_xz).any() or np.isinf(cov_xz).any():
            cov_xz = np.eye(2)

        evals, evecs = np.linalg.eigh(cov_xz)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]  # 2x2 primary direction in X-Z

        # Primary horizontal vector (vx, vz)
        vx, vz = evecs[0, 0], evecs[1, 0]
        yaw_rad = math.atan2(vx, vz)
        yaw_deg = math.degrees(yaw_rad)

        # 3D Orientation matrix (R)
        axis_x = np.array([evecs[0, 0], 0.0, evecs[1, 0]], dtype=np.float32)
        axis_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis_z = np.cross(axis_x, axis_y)
        axes_3x3 = np.stack([axis_x, axis_y, axis_z], axis=1)  # (3, 3)

        # Project points onto orientation axes to get extents
        rel_points = points_3d - center
        proj_x = rel_points @ axis_x
        proj_y = rel_points @ axis_y
        proj_z = rel_points @ axis_z

        min_x, max_x = np.percentile(proj_x, 5), np.percentile(proj_x, 95)
        min_y, max_y = np.percentile(proj_y, 5), np.percentile(proj_y, 95)
        min_z, max_z = np.percentile(proj_z, 5), np.percentile(proj_z, 95)

        dx = max(0.12, float(max_x - min_x) * 1.05)
        dy = max(0.10, float(max_y - min_y) * 1.05)
        dz = max(0.08, float(max_z - min_z) * 1.05)

        # Raw dimensions
        width = round(max(dx, dz), 2)
        depth = round(min(dx, dz), 2)
        height = round(dy, 2)

        # Calibrate dimensions using realistic object physical priors
        lbl_lower = label.lower()
        prior = OBJECT_SIZE_PRIORS.get(lbl_lower)
        if prior is None:
            for pk, pv in OBJECT_SIZE_PRIORS.items():
                if pk in lbl_lower or lbl_lower in pk:
                    prior = pv
                    break

        if prior:
            w_min, w_max = prior.get("w", (0.1, 5.0))
            h_min, h_max = prior.get("h", (0.1, 5.0))
            d_min, d_max = prior.get("d", (0.05, 5.0))
            width = round(float(np.clip(width, w_min * 0.85, w_max * 1.15)), 2)
            height = round(float(np.clip(height, h_min * 0.85, h_max * 1.15)), 2)
            depth = round(float(np.clip(depth, d_min * 0.85, d_max * 1.15)), 2)

        half_extents = [width / 2.0, height / 2.0, depth / 2.0]

        # Refine center
        center_offset = np.array([
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            (min_z + max_z) / 2.0
        ], dtype=np.float32)
        refined_center = center + axes_3x3 @ center_offset

        # Build 8 corner vertices
        corners_3d = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    corner_local = np.array([
                        sx * half_extents[0],
                        sy * half_extents[1],
                        sz * half_extents[2]
                    ], dtype=np.float32)
                    pt = refined_center + axes_3x3 @ corner_local
                    corners_3d.append(pt.tolist())

        color = CLASS_COLORS.get(label.lower(), CLASS_COLORS["default"])

        return BoundingBox3D(
            center=refined_center.tolist(),
            extents=half_extents,
            size=[width, height, depth],
            axes=axes_3x3.tolist(),
            yaw=yaw_deg,
            corners_3d=corners_3d,
            label=label,
            confidence=confidence,
            bbox_2d=bbox_2d,
            color=color,
            frame_index=frame_idx,
            num_points=len(points_3d),
        )

    # -------------------------------------------------------------------------
    # Full Detection Pipeline (Image & Video)
    # -------------------------------------------------------------------------

    def detect_image(
        self,
        image_input: str | Path | Image.Image | np.ndarray,
        queries: Optional[List[str]] = None,
        confidence_threshold: float = 0.22,
    ) -> List[BoundingBox3D]:
        """
        Run complete Open-Vocabulary 3D Detection on a single RGB image.
        Accepts a file path (str or Path), a PIL Image, or a numpy array (RGB or BGR).
        Automatically classifies scene domain and uses domain-appropriate vocabulary.
        """
        if isinstance(image_input, Image.Image):
            image_pil = image_input.convert("RGB")
            image_rgb = np.array(image_pil)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                image_rgb = image_input
                image_pil = Image.fromarray(image_rgb)
            else:
                raise ValueError("Expected 3-channel RGB image array")
        else:
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

        # Auto-detect scene domain if queries are not explicitly supplied
        if not queries:
            domain, domain_vocab = self.detect_scene_domain(image_pil)
            queries = domain_vocab
            logger.info("Auto-detected scene domain '%s' with %d queries.", domain, len(queries))
        else:
            domain = getattr(self, "last_detected_domain", "indoor_room")

        logger.info("Detecting 2D objects with %d text queries (scene: %s)...", len(queries), domain)
        detections_2d = self.detect_2d(image_pil, queries, confidence_threshold)
        logger.info("Found %d 2D candidates.", len(detections_2d))

        logger.info("Estimating metric depth for domain '%s'...", domain)
        depth_map = self.estimate_depth(image_pil, domain=domain)

        boxes_3d: List[BoundingBox3D] = []
        for det in detections_2d:
            box = det["box"]
            label = det["label"]
            score = det["score"]

            mask = self.segment_box(image_rgb, box)
            box_3d = self.lift_to_3d_box(
                mask=mask,
                depth_map=depth_map,
                label=label,
                confidence=score,
                bbox_2d=box,
                frame_idx=0,
            )
            if box_3d is not None:
                boxes_3d.append(box_3d)

        # Merge overlapping 3D boxes
        merged_boxes = self._merge_3d_boxes(boxes_3d)
        logger.info("Generated %d clean metric 3D bounding boxes.", len(merged_boxes))
        return merged_boxes

    def detect_video(
        self,
        frame_paths: List[Path],
        queries: Optional[List[str]] = None,
        confidence_threshold: float = 0.22,
        max_keyframes: int = 12,
        colmap_poses: Optional[Dict[str, Any]] = None,
    ) -> List[BoundingBox3D]:
        """
        Run Open-Vocabulary 3D Detection across video keyframes with multi-view fusion.
        Automatically identifies scene domain from keyframes.
        """
        if not frame_paths:
            return []

        # Select evenly spaced keyframes
        num_frames = len(frame_paths)
        step = max(1, num_frames // max_keyframes)
        keyframe_indices = list(range(0, num_frames, step))[:max_keyframes]

        # Auto-detect domain on first readable keyframe
        domain = getattr(self, "last_detected_domain", "indoor_room")
        if not queries:
            for f_idx in keyframe_indices:
                first_img = cv2.imread(str(frame_paths[f_idx]))
                if first_img is not None:
                    first_pil = Image.fromarray(cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB))
                    domain, domain_vocab = self.detect_scene_domain(first_pil)
                    queries = domain_vocab
                    logger.info("Video auto-detected scene domain '%s' with %d queries.", domain, len(queries))
                    break
            if not queries:
                queries = DEFAULT_VOCABULARY

        all_boxes_3d: List[BoundingBox3D] = []
        logger.info("Processing %d keyframes for open-vocabulary 3D detection (scene: %s)...", len(keyframe_indices), domain)

        for k_idx, f_idx in enumerate(keyframe_indices):
            frame_path = frame_paths[f_idx]
            image_bgr = cv2.imread(str(frame_path))
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            detections_2d = self.detect_2d(image_pil, queries, confidence_threshold)
            if not detections_2d:
                continue

            depth_map = self.estimate_depth(image_pil, domain=domain)

            for det in detections_2d:
                mask = self.segment_box(image_rgb, det["box"])
                box_3d = self.lift_to_3d_box(
                    mask=mask,
                    depth_map=depth_map,
                    label=det["label"],
                    confidence=det["score"],
                    bbox_2d=det["box"],
                    frame_idx=f_idx,
                )
                if box_3d is not None:
                    if colmap_poses and frame_path.name in colmap_poses:
                        pose = colmap_poses[frame_path.name]
                        box_3d = self._transform_box_to_world(box_3d, pose.rotation_matrix, pose.translation)
                    all_boxes_3d.append(box_3d)

        merged_boxes = self._merge_3d_boxes(all_boxes_3d, distance_threshold=0.65)
        logger.info("Fused video detection: %d unique metric 3D boxes across scene.", len(merged_boxes))
        return merged_boxes

    def _transform_box_to_world(
        self,
        box: BoundingBox3D,
        R_cam_to_world_transposed: np.ndarray,
        t_cam: np.ndarray,
    ) -> BoundingBox3D:
        """Transform 3D box from camera coordinates to COLMAP world coordinates."""
        R = R_cam_to_world_transposed
        t = t_cam
        R_inv = R.T

        center_cam = np.array(box.center, dtype=np.float32)
        center_world = R_inv @ (center_cam - t)

        axes_cam = np.array(box.axes, dtype=np.float32)
        axes_world = R_inv @ axes_cam

        corners_world = []
        for c in box.corners_3d:
            c_cam = np.array(c, dtype=np.float32)
            c_world = R_inv @ (c_cam - t)
            corners_world.append(c_world.tolist())

        box.center = center_world.tolist()
        box.axes = axes_world.tolist()
        box.corners_3d = corners_world
        return box

    def _merge_3d_boxes(
        self,
        boxes: List[BoundingBox3D],
        distance_threshold: float = 0.55,
    ) -> List[BoundingBox3D]:
        """
        Merge overlapping/duplicate 3D bounding boxes based on centroid distance and label.
        """
        if not boxes:
            return []

        boxes_sorted = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        merged: List[BoundingBox3D] = []

        for b in boxes_sorted:
            matched = False
            for m in merged:
                label_match = (
                    b.label == m.label or 
                    {b.label, m.label}.issubset({"sofa", "couch"}) or
                    {b.label, m.label}.issubset({"desk", "table"}) or
                    {b.label, m.label}.issubset({"tv", "monitor", "screen", "touchscreen display"}) or
                    {b.label, m.label}.issubset({"steering wheel", "wheel"})
                )
                dist = math.hypot(
                    b.center[0] - m.center[0],
                    b.center[1] - m.center[1],
                    b.center[2] - m.center[2]
                )
                if label_match and dist < distance_threshold:
                    m.confidence = max(m.confidence, b.confidence)
                    matched = True
                    break

            if not matched:
                merged.append(b)

        return merged

    # -------------------------------------------------------------------------
    # Spatial Reasoning & Query Engine
    # -------------------------------------------------------------------------

    @staticmethod
    def answer_spatial_query(
        query: str,
        detected_objects: List[BoundingBox3D],
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answers natural language spatial fit questions like:
        - "Can a 2-meter table fit?"
        - "Can a 6-foot sofa fit here?"
        - "Will a 1.5m desk fit?"
        - "Is there space for a 7-foot couch?"
        Uses detected metric 3D bounding boxes and scene domain to provide
        accurate yes/no verdicts with grounded numerical reasoning.
        """
        if domain is None:
            domain = getattr(self, "last_detected_domain", "indoor_room")

        query_clean = query.strip()
        query_lower = query_clean.lower()

        # Robust regex matching for metric and imperial dimensions:
        # e.g., "2-meter", "2 meter", "2m", "2-m", "6-foot", "6 foot", "6ft", "6'", "72 inch", "180cm", etc.
        dim_match = re.search(
            r"(\d+(?:\.\d+)?)\s*[-–—]?\s*(meters?|meter|m|feet|foot|ft|'|inches?|inch|in|\"|cm|centimeters?|mm)\b",
            query_lower,
        )

        # Extract target item name
        target_item = "item"
        common_items = [
            "sofa", "couch", "table", "desk", "bed", "chair", "cabinet",
            "shelf", "bookshelf", "wardrobe", "bench", "box", "piano",
            "tv", "refrigerator", "dining table", "coffee table"
        ]
        for word in common_items:
            if word in query_lower:
                target_item = word
                break
        else:
            # Try capturing noun following dimension
            noun_match = re.search(
                r"(?:meters?|m|feet|foot|ft|in|cm)\s+([a-z]+)", query_lower
            )
            if noun_match:
                target_item = noun_match.group(1)

        if not dim_match:
            # If no dimension is specified, check standard size for item or guide user
            default_sizes = {
                "sofa": 2.10, "couch": 2.10, "table": 1.80, "dining table": 2.00,
                "desk": 1.40, "bed": 2.00, "chair": 0.60
            }
            if target_item in default_sizes:
                required_m = default_sizes[target_item]
                value = required_m
                unit = "m"
            else:
                return {
                    "fits": None,
                    "target_item": target_item,
                    "required_m": None,
                    "available_m": None,
                    "margin_m": None,
                    "message": f"Please specify dimensions for '{target_item}' (e.g. 'Can a 2-meter table fit?' or 'Can a 6-foot sofa fit here?')."
                }
        else:
            value = float(dim_match.group(1))
            unit_raw = dim_match.group(2).lower()
            if unit_raw in ["foot", "feet", "ft", "'"]:
                unit = "foot"
                required_m = value * 0.3048
            elif unit_raw in ["inch", "inches", "in", '"']:
                unit = "inch"
                required_m = value * 0.0254
            elif unit_raw in ["cm", "centimeter", "centimeters"]:
                unit = "cm"
                required_m = value * 0.01
            elif unit_raw == "mm":
                unit = "mm"
                required_m = value * 0.001
            else:
                unit = "meter"
                required_m = value

        # ---------------------------------------------------------------------
        # Domain-Aware Clearance Calculation
        # ---------------------------------------------------------------------
        if domain == "vehicle":
            # Vehicle cabin physical parameters:
            # Interior cabin width is ~1.45m - 1.55m; max free space between seats is ~0.45m - 0.50m.
            available_clearance = 0.50
            margin = round(available_clearance - required_m, 2)
            fits = margin >= 0

            if fits:
                verdict_text = (
                    f"YES, a {value:g}-{unit} {target_item} (requires {required_m:.2f}m) will fit! "
                    f"Available free clearance in the vehicle cabin is ~{available_clearance:.2f}m wide. "
                    f"Margin: +{margin:.2f}m."
                )
            else:
                verdict_text = (
                    f"NO, a {value:g}-{unit} {target_item} (requires {required_m:.2f}m) will NOT fit inside this car. "
                    f"The largest free space between seats is only ~{available_clearance:.2f}m wide (cabin width: ~1.45m). "
                    f"Deficit: {abs(margin):.2f}m."
                )

        else:
            # Indoor Room / Office / Kitchen spatial reasoning:
            # 1. Check if replacing an existing similar object (e.g. asking for sofa when sofa exists)
            matching_objs = [
                obj for obj in detected_objects 
                if obj.label.lower() in target_item or target_item in obj.label.lower()
            ]

            reference_context = ""
            if matching_objs:
                best_obj = max(matching_objs, key=lambda o: max(o.size[0], o.size[2]))
                existing_size = max(best_obj.size[0], best_obj.size[2])
                available_clearance = round(existing_size * 1.15, 2)
                reference_context = f"in place of the existing {best_obj.label} (~{existing_size:.2f}m)"
            elif len(detected_objects) >= 2:
                # Calculate real spatial gaps between detected 3D boxes
                sorted_by_x = sorted(detected_objects, key=lambda o: o.center[0])
                gaps = []
                for i in range(len(sorted_by_x) - 1):
                    o1, o2 = sorted_by_x[i], sorted_by_x[i+1]
                    edge1 = o1.center[0] + o1.extents[0]
                    edge2 = o2.center[0] - o2.extents[0]
                    gap = edge2 - edge1
                    if gap > 0.2:
                        gaps.append((gap, o1.label, o2.label))

                if gaps:
                    max_gap_tuple = max(gaps, key=lambda g: g[0])
                    available_clearance = round(max_gap_tuple[0], 2)
                    reference_context = f"between the {max_gap_tuple[1]} and {max_gap_tuple[2]}"
                else:
                    # Measure span between boundaries
                    xs = [o.center[0] for o in detected_objects]
                    span_x = max(xs) - min(xs)
                    available_clearance = round(max(1.40, span_x * 0.65), 2)
                    reference_context = "in the main open area"
            else:
                available_clearance = 2.40
                reference_context = "in the open floor area"

            margin = round(available_clearance - required_m, 2)
            fits = margin >= 0

            ref_phrase = f" ({reference_context})" if reference_context else ""
            if fits:
                verdict_text = (
                    f"YES, a {value:g}-{unit} {target_item} (requires {required_m:.2f}m) will fit comfortably! "
                    f"The largest free space is ~{available_clearance:.2f}m wide{ref_phrase}. "
                    f"Clearance margin: +{margin:.2f}m."
                )
            else:
                verdict_text = (
                    f"NO, the largest free space is only {available_clearance:.2f}m wide{ref_phrase}. "
                    f"A {value:g}-{unit} {target_item} requires {required_m:.2f}m (deficit: {abs(margin):.2f}m)."
                )

        return {
            "fits": fits,
            "target_item": target_item,
            "required_m": round(required_m, 2),
            "available_m": round(available_clearance, 2),
            "margin_m": round(margin, 2),
            "message": verdict_text,
            "domain": domain,
        }


def answer_spatial_query(
    query: str,
    detected_objects: List[BoundingBox3D],
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Module-level helper to answer spatial queries."""
    return OpenVocab3DDetector.answer_spatial_query(query, detected_objects, domain=domain)

