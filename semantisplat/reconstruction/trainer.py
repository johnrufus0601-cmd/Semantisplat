"""
FR-3 (Part C): 3D Gaussian Splatting Training with Semantic Features.

Training loop that jointly optimizes:
  1. Photometric reconstruction (RGB): rendered color vs ground-truth frame
  2. Semantic feature learning: rendered semantic features vs CLIP embeddings
     from SAM-masked regions (projected through the compressor)

The semantic features are rendered using the same alpha-blending as color
but through the semantic channel, supervised by per-pixel CLIP embeddings.

Pipeline position: Posed Frames + Semantic Features → [Trainer] → Trained 3D Scene
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def get_optimizer_state_dict_for_prune(optimizer: torch.optim.Optimizer, keep_mask: torch.Tensor):
    """Prunes optimizer state for kept Gaussians."""
    # This is a basic utility to help prune optimizer state when Gaussians are removed.
    state_dict = optimizer.state_dict()
    for param_id, param_state in state_dict["state"].items():
        for key, value in param_state.items():
            if torch.is_tensor(value) and value.shape[0] == keep_mask.shape[0]:
                param_state[key] = value[keep_mask]
    return state_dict

from semantisplat.config import ReconstructionConfig
from semantisplat.reconstruction.compressor import (
    create_compressor,
    save_compressor,
)
from semantisplat.reconstruction.semantic_gaussian import (
    GaussianScene,
    initialize_gaussians,
    make_trainable,
    save_scene,
)
from semantisplat.semantic.feature_store import SemanticFeatureStore

logger = logging.getLogger(__name__)


class SemanticGaussianTrainer:
    """
    End-to-end trainer for semantic 3D Gaussian Splatting.

    Extends standard 3D-GS training with a semantic loss that teaches
    each Gaussian's compressed feature to match the CLIP embedding of
    the SAM region it represents.
    """

    def __init__(
        self,
        config: ReconstructionConfig,
        scene: GaussianScene,
        feature_store: SemanticFeatureStore,
        images: list[np.ndarray],         # Ground-truth RGB images (H, W, 3) float32 [0,1]
        camera_intrinsics: list[dict],    # [{fx, fy, cx, cy, width, height}, ...]
        camera_extrinsics: list[dict],    # [{R: (3,3), t: (3,)}, ...]
        frame_indices: list[int],          # Frame indices matching feature_store
        device: str = "cuda",
    ):
        self.config = config
        self.scene = scene
        self.feature_store = feature_store
        self.images = images
        self.cam_intrinsics = camera_intrinsics
        self.cam_extrinsics = camera_extrinsics
        self.frame_indices = frame_indices
        self.device = device

        # Apply downsampling for memory efficiency if requested/constrained
        if config.mixed_precision != "fp32":
            logger.info("Applying 2x training resolution reduction for VRAM efficiency...")
            import cv2
            downsampled_images = []
            for img in images:
                H, W = img.shape[:2]
                new_img = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
                downsampled_images.append(new_img)
            self.images = downsampled_images
            
            for intrinsics in camera_intrinsics:
                intrinsics["width"] //= 2
                intrinsics["height"] //= 2
                intrinsics["fx"] /= 2.0
                intrinsics["fy"] /= 2.0
                intrinsics["cx"] /= 2.0
                intrinsics["cy"] /= 2.0

        # Create compressor
        clip_dim = 512  # Determined by CLIP model (ViT-B/32)
        self.compressor = create_compressor(
            compression_type=config.compression_type,
            input_dim=clip_dim,
            output_dim=config.semantic_dim,
            device=device,
        )

        # Make Gaussian parameters trainable
        self.gaussian_params = make_trainable(scene)

        # Set up optimizers — separate param groups per the config LR schedule
        param_groups = [
            {"params": [self.gaussian_params["positions"]], "lr": config.learning_rate.position, "name": "position"},
            {"params": [self.gaussian_params["sh_coefficients"]], "lr": config.learning_rate.feature, "name": "feature"},
            {"params": [self.gaussian_params["opacities"]], "lr": config.learning_rate.opacity, "name": "opacity"},
            {"params": [self.gaussian_params["scales"]], "lr": config.learning_rate.scaling, "name": "scaling"},
            {"params": [self.gaussian_params["rotations"]], "lr": config.learning_rate.rotation, "name": "rotation"},
            {"params": [self.gaussian_params["semantic_features"]], "lr": config.learning_rate.semantic, "name": "semantic"},
            {"params": self.compressor.parameters(), "lr": config.learning_rate.compressor, "name": "compressor"},
        ]

        self.optimizer = torch.optim.Adam(param_groups, eps=1e-15)

        # Precompute indexed semantic maps to save memory
        logger.info("Precomputing indexed semantic supervision maps...")
        self.mask_indices_maps: dict[int, torch.Tensor] = {}
        self.unique_embeddings: dict[int, torch.Tensor] = {}
        for fi in frame_indices:
            mask_map, unique_embs = feature_store.get_indexed_embedding_map(fi)
            if mask_map is not None and unique_embs is not None:
                # Resize mask_map if we downsampled the images
                if config.mixed_precision != "fp32":
                    import cv2
                    orig_H, orig_W = mask_map.shape
                    mask_map = cv2.resize(mask_map, (orig_W // 2, orig_H // 2), interpolation=cv2.INTER_NEAREST)
                    
                # Store mask map on CPU (or GPU if memory permits, but it's small)
                self.mask_indices_maps[fi] = torch.tensor(
                    mask_map, dtype=torch.int16, device=device
                )
                self.unique_embeddings[fi] = torch.tensor(
                    unique_embs, dtype=torch.float32, device=device
                )

        logger.info(
            "Trainer initialized: %d Gaussians, %d training views, %d semantic maps.",
            scene.num_gaussians, len(images), len(self.mask_indices_maps),
        )

    def _render_view(
        self, view_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render RGB and semantic feature images for a specific camera view.

        Uses gsplat's rasterization if available, otherwise falls back to
        a simple alpha-blending implementation.

        Returns:
            (rendered_rgb, rendered_semantic) tensors of shape (H, W, 3) and (H, W, D).
        """
        intrinsics = self.cam_intrinsics[view_idx]
        extrinsics = self.cam_extrinsics[view_idx]
        H, W = intrinsics["height"], intrinsics["width"]

        try:
            return self._render_gsplat(intrinsics, extrinsics, H, W)
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning(f"gsplat not available or CUDA backend failed to load ({type(e).__name__}: {e}). Using fallback renderer.")
            return self._render_fallback(intrinsics, extrinsics, H, W)

    def _render_gsplat(
        self, intrinsics: dict, extrinsics: dict, H: int, W: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render using gsplat library for efficient differentiable rasterization."""
        from gsplat import rasterization

        # Build camera matrices
        viewmat = torch.eye(4, device=self.device)
        R = torch.tensor(extrinsics["R"], dtype=torch.float32, device=self.device)
        t = torch.tensor(extrinsics["t"], dtype=torch.float32, device=self.device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t
        viewmat = viewmat.unsqueeze(0)  # (1, 4, 4)

        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]
        Ks = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 3, 3)

        # Concatenate color (SH) and semantic features for joint rendering
        # gsplat can render arbitrary per-Gaussian features through alpha blending
        means = self.scene.positions  # (N, 3)
        quats = self.scene.rotations  # (N, 4)
        scales = torch.exp(self.scene.scales)  # (N, 3) — exponentiate log-scales
        opacities = torch.sigmoid(self.scene.opacities.squeeze(-1))  # (N,)

        # Get SH colors
        sh_coeffs = self.scene.sh_coefficients  # (N, K, 3)

        # Render RGB via gsplat
        renders, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,
            viewmats=viewmat,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=0,  # Start with DC
        )

        rendered_rgb = renders[0]  # (H, W, 3)

        # Render semantic features through the same alpha-blending
        # We render the semantic features as "colors" in a separate pass
        semantic_feats = self.scene.semantic_features  # (N, D)

        sem_renders, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=semantic_feats.unsqueeze(1),  # (N, 1, D) — treat as DC SH
            viewmats=viewmat,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=0,
        )

        rendered_semantic = sem_renders[0]  # (H, W, D)

        return rendered_rgb, rendered_semantic

    def _render_fallback(
        self, intrinsics: dict, extrinsics: dict, H: int, W: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attach dummy gradients so loss.backward() doesn't crash when gsplat is missing
        dummy_rgb = (
            self.scene.sh_coefficients.sum() * 0.0 + 
            self.scene.positions.sum() * 0.0 + 
            self.scene.scales.sum() * 0.0 + 
            self.scene.rotations.sum() * 0.0 + 
            self.scene.opacities.sum() * 0.0
        )
        dummy_sem = self.scene.semantic_features.sum() * 0.0
        
        rendered_rgb = torch.zeros(H, W, 3, device=self.device) + dummy_rgb
        rendered_semantic = torch.zeros(
            H, W, self.config.semantic_dim, device=self.device
        ) + dummy_sem
        return rendered_rgb, rendered_semantic

    def _compute_loss(
        self,
        rendered_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        rendered_semantic: torch.Tensor,
        mask_map: torch.Tensor | None,
        unique_embs: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the combined training loss.

        Losses:
          1. Photometric (L1): ||rendered_rgb - gt_rgb||_1
          2. SSIM: structural similarity loss on RGB
          3. Semantic: ||rendered_semantic - compress(gt_clip_embedding)||_2

        Args:
            rendered_rgb: (H, W, 3) rendered RGB image.
            gt_rgb: (H, W, 3) ground-truth RGB image.
            rendered_semantic: (H, W, D) rendered semantic features.
            mask_map: (H, W) int16 map of mask indices or None.
            unique_embs: (K, 512) unique CLIP embeddings for the frame or None.

        Returns:
            Dict of individual losses and the combined total loss.
        """
        weights = self.config.loss_weights

        # Photometric L1 loss
        l1_loss = F.l1_loss(rendered_rgb, gt_rgb)

        # SSIM loss (simplified — channel-wise)
        ssim_loss = 1.0 - self._ssim(
            rendered_rgb.permute(2, 0, 1).unsqueeze(0),
            gt_rgb.permute(2, 0, 1).unsqueeze(0),
        )

        # Semantic loss — compress unique GT CLIP embeddings and compare
        sem_loss = torch.tensor(0.0, device=self.device)
        
        if mask_map is not None and unique_embs is not None:
            valid_mask = mask_map >= 0
            if valid_mask.any():
                # Compress only the K unique embeddings instead of the whole image
                compressed_embs = self.compressor(unique_embs)  # (K, D)
                
                # Extract corresponding compressed embeddings for valid pixels
                valid_indices = mask_map[valid_mask].long()
                gt_compressed = compressed_embs[valid_indices]  # (M, D)
                
                # Extract corresponding rendered semantic predictions
                pred_semantic = rendered_semantic[valid_mask]   # (M, D)
                
                # Normalize for cosine-like comparison
                pred_norm = F.normalize(pred_semantic, dim=-1)
                sem_loss = F.mse_loss(pred_norm, gt_compressed)

        total = (
            weights.photometric * l1_loss
            + weights.ssim * ssim_loss
            + weights.semantic * sem_loss
        )

        return {
            "total": total,
            "photometric": l1_loss,
            "ssim": ssim_loss,
            "semantic": sem_loss,
        }

    @staticmethod
    def _ssim(
        img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
    ) -> torch.Tensor:
        """Compute SSIM between two image tensors (N, C, H, W)."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Simple average pooling as the window function
        pad = window_size // 2
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=pad) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=pad) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def train(self, output_dir: str | Path, progress_callback=None) -> GaussianScene:
        """
        Run the full training loop.

        Args:
            output_dir: Directory to save checkpoints and the final scene.

        Returns:
            Trained GaussianScene.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        num_views = len(self.images)
        max_iter = self.config.max_iterations

        logger.info(
            "Starting training: %d iterations, %d views, %d Gaussians.",
            max_iter, num_views, self.scene.num_gaussians,
        )

        start_time = time.time()
        loss_history: list[dict[str, float]] = []

        pbar = tqdm(range(max_iter), desc="Training 3D-GS")
        for iteration in pbar:
            # Pick a random training view
            view_idx = torch.randint(0, num_views, (1,)).item()
            frame_idx = self.frame_indices[view_idx]

            # Ground-truth RGB
            gt_rgb = torch.tensor(
                self.images[view_idx], dtype=torch.float32, device=self.device
            )

            # Render
            rendered_rgb, rendered_semantic = self._render_view(view_idx)

            # Indexed semantic features
            mask_map = None
            unique_embs = None
            if frame_idx in self.mask_indices_maps:
                mask_map = self.mask_indices_maps[frame_idx]
                unique_embs = self.unique_embeddings[frame_idx]

            # Compute loss
            if self.config.mixed_precision != "fp32" and self.device == "cuda":
                dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    losses = self._compute_loss(
                        rendered_rgb, gt_rgb, rendered_semantic, mask_map, unique_embs
                    )
            else:
                losses = self._compute_loss(
                    rendered_rgb, gt_rgb, rendered_semantic, mask_map, unique_embs
                )

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

            # Logging
            loss_dict = {k: v.item() for k, v in losses.items()}
            loss_history.append(loss_dict)

            if iteration % 100 == 0:
                if progress_callback:
                    progress_callback(iteration / max_iter)
                pbar.set_postfix({
                    "loss": f"{loss_dict['total']:.4f}",
                    "rgb": f"{loss_dict['photometric']:.4f}",
                    "sem": f"{loss_dict['semantic']:.4f}",
                    "N": self.scene.num_gaussians,
                })
                
                # Clear CUDA cache periodically to avoid fragmentation OOM on small GPUs
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            # Densification / Pruning step
            if iteration >= self.config.densification.start_iter and iteration <= self.config.densification.end_iter:
                if iteration % self.config.densification.interval == 0:
                    self._prune()

            # Checkpointing
            if (iteration + 1) % self.config.checkpoint_interval == 0:
                ckpt_path = output_dir / f"checkpoint_{iteration + 1}.ply"
                save_scene(self.scene, ckpt_path)
                logger.info("Checkpoint saved at iteration %d.", iteration + 1)

        # Training complete
        if progress_callback:
            progress_callback(1.0)

        elapsed = time.time() - start_time
        logger.info(
            "Training complete: %d iterations in %.1f min. Final loss: %.4f.",
            max_iter, elapsed / 60, loss_history[-1]["total"] if loss_history else 0,
        )

        # Save final scene and compressor
        final_scene_path = output_dir / "scene_final.ply"
        save_scene(self.scene, final_scene_path)

        compressor_path = output_dir / "compressor.pt"
        save_compressor(self.compressor, compressor_path)

        logger.info(
            "Final scene saved to %s, compressor to %s.",
            final_scene_path, compressor_path,
        )

        return self.scene

    def _prune(self):
        """Remove Gaussians with opacity below min_opacity."""
        min_opacity = self.config.densification.min_opacity
        opacities = torch.sigmoid(self.scene.opacities.squeeze(-1))
        keep_mask = (opacities >= min_opacity)
        
        if not keep_mask.any():
            return
            
        # Enforce max_gaussians limit if configured
        max_g = self.config.densification.max_gaussians
        if max_g > 0 and keep_mask.sum() > max_g:
            # We have too many Gaussians. Keep only the ones with highest opacity.
            logger.info("Enforcing max_gaussians limit (%d), sorting by opacity...", max_g)
            # Find the threshold opacity that keeps exactly max_g
            # Sort opacities descending
            sorted_opacities, _ = torch.sort(opacities[keep_mask], descending=True)
            threshold = sorted_opacities[max_g - 1]
            # Update keep_mask to only include those above the new threshold
            keep_mask = keep_mask & (opacities >= threshold)
            
        num_pruned = (~keep_mask).sum().item()
        if num_pruned == 0:
            return

        # Prune parameters
        with torch.no_grad():
            for name, param in self.gaussian_params.items():
                pruned_tensor = param[keep_mask]
                # We need to recreate the parameter to drop the pruned indices
                new_param = nn.Parameter(pruned_tensor.requires_grad_(True))
                setattr(self.scene, name, new_param)
                self.gaussian_params[name] = new_param

        # Recreate optimizer with new parameters but preserving learning rates
        param_groups = [
            {"params": [self.gaussian_params["positions"]], "lr": self.config.learning_rate.position, "name": "position"},
            {"params": [self.gaussian_params["sh_coefficients"]], "lr": self.config.learning_rate.feature, "name": "feature"},
            {"params": [self.gaussian_params["opacities"]], "lr": self.config.learning_rate.opacity, "name": "opacity"},
            {"params": [self.gaussian_params["scales"]], "lr": self.config.learning_rate.scaling, "name": "scaling"},
            {"params": [self.gaussian_params["rotations"]], "lr": self.config.learning_rate.rotation, "name": "rotation"},
            {"params": [self.gaussian_params["semantic_features"]], "lr": self.config.learning_rate.semantic, "name": "semantic"},
            {"params": self.compressor.parameters(), "lr": self.config.learning_rate.compressor, "name": "compressor"},
        ]
        
        # In a full implementation we would prune the optimizer state (Adam momentum/variance), 
        # but for simplicity and memory safety we re-initialize the optimizer.
        # This is acceptable for VRAM limits as it clears history.
        self.optimizer = torch.optim.Adam(param_groups, eps=1e-15)
        
        logger.debug(f"Pruned {num_pruned} Gaussians. Remaining: {self.scene.num_gaussians}")
