"""
FR-3 (Part A): Semantic Feature Compression.

Compresses full CLIP embeddings (512-dim) to a compact representation (e.g. 16-dim)
that can be stored per-Gaussian without exceeding GPU memory.

Why compress?
  Storing full 512-dim CLIP vectors for millions of Gaussians would require
  too much VRAM. For example, 1M Gaussians × 512 floats × 4 bytes = 2 GB.
  Compressing to 16-dim: 1M × 16 × 4 = 64 MB — a 32× reduction.

The compressor is trained jointly with the Gaussians so the compressed
representation retains maximum semantic discriminability.

Pipeline position: CLIP 512-d → [Compressor] → 16-d (stored per Gaussian)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LinearCompressor(nn.Module):
    """
    Simple learned linear projection: CLIP dim → compressed dim.

    This is the approach used in LangSplat / LEGaussians:
    a single linear layer trained end-to-end with the Gaussian scene.
    """

    def __init__(self, input_dim: int = 512, output_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim, bias=True)

        # Initialize with a truncated SVD-style init for stable training
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        logger.info(
            "LinearCompressor initialized: %d → %d (%.1f× compression).",
            input_dim, output_dim, input_dim / output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress CLIP embeddings.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Compressed tensor of shape (..., output_dim), L2-normalized.
        """
        compressed = self.projection(x)
        # L2-normalize so cosine similarity works correctly in compressed space
        compressed = nn.functional.normalize(compressed, dim=-1)
        return compressed


class MLPCompressor(nn.Module):
    """
    Two-layer MLP projection: CLIP dim → hidden → compressed dim.

    Slightly more expressive than linear, useful if semantic_dim is very small.
    """

    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 128, output_dim: int = 16
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        # Init
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        logger.info(
            "MLPCompressor initialized: %d → %d → %d.",
            input_dim, hidden_dim, output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress and L2-normalize."""
        compressed = self.network(x)
        compressed = nn.functional.normalize(compressed, dim=-1)
        return compressed


def create_compressor(
    compression_type: str = "linear",
    input_dim: int = 512,
    output_dim: int = 16,
    device: str = "cuda",
) -> nn.Module:
    """
    Factory function to create a semantic feature compressor.

    Args:
        compression_type: "linear" or "mlp".
        input_dim: CLIP embedding dimensionality (512 for ViT-B/32).
        output_dim: Compressed feature dimensionality.
        device: Torch device.

    Returns:
        Compressor module on the specified device.
    """
    if compression_type == "mlp":
        compressor = MLPCompressor(input_dim, input_dim // 4, output_dim)
    else:
        compressor = LinearCompressor(input_dim, output_dim)

    return compressor.to(device)


def save_compressor(compressor: nn.Module, path: str | Path) -> None:
    """Save compressor weights to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": compressor.state_dict(),
        "input_dim": compressor.input_dim,
        "output_dim": compressor.output_dim,
        "type": "mlp" if isinstance(compressor, MLPCompressor) else "linear",
    }, path)
    logger.info("Compressor saved to %s.", path)


def load_compressor(path: str | Path, device: str = "cuda") -> nn.Module:
    """Load compressor weights from disk."""
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, nn.Module):
        compressor = checkpoint.to(device)
        compressor.eval()
        logger.info("Compressor module loaded directly from %s.", path)
        return compressor

    compressor = create_compressor(
        compression_type=checkpoint["type"],
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        device=device,
    )
    compressor.load_state_dict(checkpoint["state_dict"])
    compressor.eval()
    logger.info("Compressor loaded from %s.", path)
    return compressor
