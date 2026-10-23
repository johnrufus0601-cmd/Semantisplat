"""
FR-3 (Part B): Semantic Gaussian — Extended 3D Gaussian with Semantic Features.

A standard 3D Gaussian stores: position, covariance (scale+rotation), color, opacity.
SemantiSplat extends each Gaussian with a compressed semantic feature vector.

Each Gaussian = {
    position:          (3,)   — xyz in world coordinates
    scale:             (3,)   — log-scale along each axis
    rotation:          (4,)   — quaternion (wxyz)
    sh_coefficients:   (K, 3) — spherical harmonics for view-dependent color
    opacity:           (1,)   — sigmoid-activated opacity
    semantic_feature:  (D,)   — compressed CLIP feature (D = semantic_dim, e.g. 16)
}

3D Gaussian Splatting represents a 3D scene using a collection of 3D Gaussian
primitives and renders the scene through differentiable rasterization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class GaussianScene:
    """
    A collection of 3D Gaussians representing a scene.

    All tensors are on the same device and require gradients for training.
    """
    # Core Gaussian parameters (standard 3D-GS)
    positions: torch.Tensor          # (N, 3) — xyz
    scales: torch.Tensor             # (N, 3) — log-scale
    rotations: torch.Tensor          # (N, 4) — quaternion (wxyz)
    sh_coefficients: torch.Tensor    # (N, K, 3) — spherical harmonics
    opacities: torch.Tensor          # (N, 1) — raw opacity (pre-sigmoid)

    # Semantic extension (SemantiSplat-specific)
    semantic_features: torch.Tensor  # (N, semantic_dim) — compressed CLIP features

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def semantic_dim(self) -> int:
        return self.semantic_features.shape[1]

    @property
    def device(self) -> torch.device:
        return self.positions.device


def initialize_gaussians(
    points3d: np.ndarray,
    colors: np.ndarray | None = None,
    semantic_dim: int = 16,
    sh_degree: int = 0,
    device: str = "cuda",
) -> GaussianScene:
    """
    Initialize Gaussians from a sparse 3D point cloud (e.g. from COLMAP).

    Args:
        points3d: (N, 3) array of 3D point positions.
        colors: (N, 3) optional RGB colors in [0, 1]. Random if None.
        semantic_dim: Dimensionality of compressed semantic features.
        sh_degree: Spherical harmonics degree (0 = DC only).
        device: Torch device.

    Returns:
        GaussianScene ready for training.
    """
    # Target a denser initial point cloud since we don't have training-time densification
    TARGET_POINTS = 100000
    
    if len(points3d) < TARGET_POINTS:
        from scipy.spatial import KDTree
        tree = KDTree(points3d)
        distances, indices = tree.query(points3d, k=4)
        
        needed = TARGET_POINTS - len(points3d)
        new_points = []
        new_colors = [] if colors is not None else None
        
        for _ in range(needed):
            idx = np.random.randint(len(points3d))
            neighbor_idx = indices[idx, np.random.randint(1, 4)]
            
            alpha = np.random.rand()
            dist = distances[idx, np.random.randint(1, 4)]
            noise = np.random.randn(3) * dist * 0.2
            
            pt = points3d[idx] * alpha + points3d[neighbor_idx] * (1 - alpha) + noise
            new_points.append(pt)
            
            if colors is not None:
                col = colors[idx] * alpha + colors[neighbor_idx] * (1 - alpha)
                new_colors.append(col)
                
        points3d = np.vstack([points3d, np.array(new_points)])
        if colors is not None:
            colors = np.vstack([colors, np.array(new_colors)])

    N = points3d.shape[0]
    logger.info("Initializing %d Gaussians (semantic_dim=%d).", N, semantic_dim)

    # Positions
    positions = torch.tensor(points3d, dtype=torch.float32, device=device)

    # Scales — initialize based on average nearest-neighbor distance
    scales = torch.zeros(N, 3, dtype=torch.float32, device=device)
    nn_dist = _compute_nn_distances(points3d)
    init_scale = np.log(np.clip(nn_dist, 1e-7, None))
    scales[:] = torch.tensor(init_scale, dtype=torch.float32, device=device).unsqueeze(1).expand(-1, 3)

    # Rotations — identity quaternion
    rotations = torch.zeros(N, 4, dtype=torch.float32, device=device)
    rotations[:, 0] = 1.0  # w=1, x=y=z=0

    # Spherical harmonics — DC component only at init
    K = (sh_degree + 1) ** 2
    sh_coefficients = torch.zeros(N, K, 3, dtype=torch.float32, device=device)
    if colors is not None:
        # Set DC coefficient to match the input colors
        # SH DC component formula: color = 0.5 + C0 * sh[0]
        # Therefore, sh[0] = (color - 0.5) / C0
        C0 = 0.28209479177387814
        sh_coefficients[:, 0, :] = (torch.tensor(
            colors, dtype=torch.float32, device=device
        ) - 0.5) / C0
    else:
        sh_coefficients[:, 0, :] = torch.rand(N, 3, device=device) - 0.5

    # Opacity — initialize to ~0.1 (inverse sigmoid)
    opacities = torch.full((N, 1), fill_value=-2.197, dtype=torch.float32, device=device)

    # Semantic features — initialize small random
    semantic_features = torch.randn(N, semantic_dim, dtype=torch.float32, device=device) * 0.01

    scene = GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        sh_coefficients=sh_coefficients,
        opacities=opacities,
        semantic_features=semantic_features,
    )

    logger.info(
        "Gaussians initialized: %d points, %d SH coeffs, %d semantic dim.",
        N, K, semantic_dim,
    )

    return scene


def _compute_nn_distances(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Compute average k-nearest-neighbor distance for each point."""
    from scipy.spatial import KDTree

    if len(points) < k + 1:
        return np.full(len(points), 0.01)

    tree = KDTree(points)
    distances, _ = tree.query(points, k=k + 1)  # +1 because closest is self
    avg_dist = np.mean(distances[:, 1:], axis=1)  # skip self-distance

    return avg_dist


def make_trainable(scene: GaussianScene) -> dict[str, nn.Parameter]:
    """
    Convert scene tensors to nn.Parameters for gradient-based optimization.

    Returns a dict of name → Parameter suitable for torch optimizers.
    """
    params = {}
    for name in [
        "positions", "scales", "rotations", "sh_coefficients",
        "opacities", "semantic_features",
    ]:
        tensor = getattr(scene, name)
        param = nn.Parameter(tensor.detach().clone().requires_grad_(True))
        setattr(scene, name, param)
        params[name] = param

    return params


def save_scene(scene: GaussianScene, path: str | Path) -> None:
    """
    Save a Gaussian scene to a .ply file with extended semantic attributes.

    The PLY format is extended with 'semantic_N' properties for each
    dimension of the compressed semantic feature.
    """
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    N = scene.num_gaussians
    D = scene.semantic_dim

    # Collect all attributes
    positions = scene.positions.detach().cpu().numpy()
    scales = scene.scales.detach().cpu().numpy()
    rotations = scene.rotations.detach().cpu().numpy()
    opacities = scene.opacities.detach().cpu().numpy().squeeze(-1)
    sh = scene.sh_coefficients.detach().cpu().numpy()
    semantics = scene.semantic_features.detach().cpu().numpy()

    # Build structured array
    dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("opacity", "f4"),
    ]

    # SH coefficients
    K = sh.shape[1]
    for k in range(K):
        for c in range(3):
            dtype_list.append((f"sh_{k}_{c}", "f4"))

    # Semantic features
    for d in range(D):
        dtype_list.append((f"semantic_{d}", "f4"))

    arr = np.zeros(N, dtype=dtype_list)
    arr["x"] = positions[:, 0]
    arr["y"] = positions[:, 1]
    arr["z"] = positions[:, 2]
    arr["scale_0"] = scales[:, 0]
    arr["scale_1"] = scales[:, 1]
    arr["scale_2"] = scales[:, 2]
    arr["rot_0"] = rotations[:, 0]
    arr["rot_1"] = rotations[:, 1]
    arr["rot_2"] = rotations[:, 2]
    arr["rot_3"] = rotations[:, 3]
    arr["opacity"] = opacities

    for k in range(K):
        for c in range(3):
            arr[f"sh_{k}_{c}"] = sh[:, k, c]

    for d in range(D):
        arr[f"semantic_{d}"] = semantics[:, d]

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(path))

    logger.info("Scene saved: %d Gaussians to %s.", N, path)


def load_scene(path: str | Path, device: str = "cuda") -> GaussianScene:
    """Load a Gaussian scene from a .ply file with semantic attributes."""
    from plyfile import PlyData

    path = Path(path)
    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]

    N = len(vertex)

    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1)
    rotations = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1)
    opacities = vertex["opacity"].reshape(-1, 1)

    # Detect SH degree from properties
    sh_props = [p.name for p in vertex.properties if p.name.startswith("sh_")]
    K = len(sh_props) // 3
    sh = np.zeros((N, max(K, 1), 3))
    for k in range(K):
        for c in range(3):
            sh[:, k, c] = vertex[f"sh_{k}_{c}"]

    # Detect semantic dim from properties
    sem_props = [p.name for p in vertex.properties if p.name.startswith("semantic_")]
    D = len(sem_props)
    semantics = np.zeros((N, max(D, 1)))
    for d in range(D):
        semantics[:, d] = vertex[f"semantic_{d}"]

    scene = GaussianScene(
        positions=torch.tensor(positions, dtype=torch.float32, device=device),
        scales=torch.tensor(scales, dtype=torch.float32, device=device),
        rotations=torch.tensor(rotations, dtype=torch.float32, device=device),
        sh_coefficients=torch.tensor(sh, dtype=torch.float32, device=device),
        opacities=torch.tensor(opacities, dtype=torch.float32, device=device),
        semantic_features=torch.tensor(semantics, dtype=torch.float32, device=device),
    )

    logger.info("Scene loaded: %d Gaussians, %d semantic dim from %s.", N, D, path)
    return scene
