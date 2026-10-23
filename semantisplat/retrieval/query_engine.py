"""
FR-4 (Part A): Natural-Language Query Engine.

Accepts a text query, encodes it with CLIP's text encoder, projects through
the compressor to compressed space, then computes cosine similarity against
every Gaussian's semantic feature to find the top-K matching regions.

If no Gaussian exceeds the similarity threshold, the system reports
"no confident match found" instead of returning a false-positive location.

Pipeline position: Text → [CLIP text encoder] → [Compressor] → Cosine Similarity
                   → Top-K Gaussians → Result
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from semantisplat.config import RetrievalConfig
from semantisplat.reconstruction.compressor import load_compressor
from semantisplat.reconstruction.semantic_gaussian import GaussianScene, load_scene
from semantisplat.semantic.embedder import CLIPEmbedder

logger = logging.getLogger(__name__)


@dataclass
class QueryMatch:
    """A single matching Gaussian region."""
    gaussian_indices: list[int]           # Indices of matched Gaussians
    centroid: np.ndarray                  # (3,) mean position of the cluster
    similarity_score: float               # Cosine similarity score
    bbox_3d: tuple[np.ndarray, np.ndarray] | None  # (min_corner, max_corner) or None
    obb_3d: dict | None = None            # Oriented Bounding Box: {"center": [...], "axes": [...], "extents": [...]}


@dataclass
class QueryResult:
    """Result of a text query against the trained scene."""
    query_text: str
    match_found: bool
    matches: list[QueryMatch]
    max_similarity: float
    message: str
    query_embedding_dim: int


class SemanticQueryEngine:
    """
    Searches a trained semantic 3D Gaussian scene using natural language.

    The query pipeline:
      1. Encode text query with CLIP text encoder → (512,) vector
      2. Project through compressor → (semantic_dim,) compressed vector
      3. Compute cosine similarity with all Gaussians' semantic features
      4. Return top-K matches above the similarity threshold
      5. If nothing exceeds threshold → "no confident match found"
    """

    def __init__(
        self,
        scene: GaussianScene,
        compressor: torch.nn.Module,
        clip_embedder: CLIPEmbedder,
        config: RetrievalConfig | None = None,
        device: str = "cuda",
    ):
        self.scene = scene
        self.compressor = compressor
        self.clip_embedder = clip_embedder
        self.config = config or RetrievalConfig()
        self.device = device

        # Pre-normalize Gaussian semantic features for fast cosine similarity
        self._semantic_features = F.normalize(
            scene.semantic_features.detach(), dim=-1
        )

        logger.info(
            "QueryEngine ready: %d Gaussians, semantic_dim=%d, top_k=%d, threshold=%.2f.",
            scene.num_gaussians, scene.semantic_dim,
            self.config.top_k, self.config.similarity_threshold,
        )

    @classmethod
    def from_files(
        cls,
        scene_path: str | Path,
        compressor_path: str | Path,
        config: RetrievalConfig | None = None,
        clip_model_name: str = "ViT-B-32",
        device: str = "cuda",
    ) -> "SemanticQueryEngine":
        """
        Load a query engine from saved scene and compressor files.

        Args:
            scene_path: Path to the .ply scene file.
            compressor_path: Path to the compressor .pt file.
            config: Retrieval configuration.
            clip_model_name: CLIP model for text encoding.
            device: Torch device.
        """
        from semantisplat.config import CLIPConfig

        scene = load_scene(scene_path, device)
        compressor = load_compressor(compressor_path, device)
        clip_embedder = CLIPEmbedder(
            CLIPConfig(model_name=clip_model_name), device
        )

        return cls(scene, compressor, clip_embedder, config, device)

    @torch.no_grad()
    def query(
        self,
        text: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> QueryResult:
        """
        Search the 3D scene with a natural-language text query.

        Args:
            text: Free-text query (e.g. "blue coffee mug", "wooden chair").
            top_k: Number of results to return. Uses config default if None.
            similarity_threshold: Min similarity for a match. Uses config default if None.

        Returns:
            QueryResult with matched regions or a "no match" message.
        """
        if top_k is None:
            top_k = self.config.top_k
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold

        logger.info("Query: '%s' (top_k=%d, threshold=%.2f)", text, top_k, similarity_threshold)

        # Step 1: Encode text with CLIP
        text_embedding = self.clip_embedder.encode_single_text(text)  # (512,)
        text_tensor = torch.tensor(
            text_embedding, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, 512)

        # Step 2: Compress to same space as Gaussian features
        query_compressed = self.compressor(text_tensor)  # (1, D)
        query_compressed = F.normalize(query_compressed, dim=-1)

        # Step 3: Cosine similarity with all Gaussians
        # semantic_features: (N, D), query_compressed: (1, D)
        similarities = torch.mm(
            self._semantic_features, query_compressed.T
        ).squeeze(-1)  # (N,)

        max_sim = similarities.max().item()

        # Step 4: Check threshold
        if max_sim < similarity_threshold:
            logger.info(
                "No confident match found. Max similarity: %.3f < threshold: %.3f.",
                max_sim, similarity_threshold,
            )
            return QueryResult(
                query_text=text,
                match_found=False,
                matches=[],
                max_similarity=max_sim,
                message=(
                    f"No confident match found for '{text}'. "
                    f"Maximum similarity ({max_sim:.3f}) is below the "
                    f"threshold ({similarity_threshold:.3f})."
                ),
                query_embedding_dim=self.scene.semantic_dim,
            )

        # Step 5: Get top-K Gaussians
        k = min(top_k * 50, len(similarities))  # Get more candidates for clustering
        top_values, top_indices = torch.topk(similarities, k)

        # Filter by threshold
        mask = top_values >= similarity_threshold
        top_values = top_values[mask]
        top_indices = top_indices[mask]

        if len(top_indices) == 0:
            return QueryResult(
                query_text=text,
                match_found=False,
                matches=[],
                max_similarity=max_sim,
                message=f"No Gaussians exceeded similarity threshold for '{text}'.",
                query_embedding_dim=self.scene.semantic_dim,
            )

        # Step 6: Spatial clustering (group nearby Gaussians into regions)
        positions = self.scene.positions[top_indices].detach().cpu().numpy()
        scores = top_values.cpu().numpy()
        indices = top_indices.cpu().numpy()

        if self.config.spatial_clustering.enabled:
            matches = self._cluster_matches(positions, scores, indices, top_k)
        else:
            # No clustering — return top-K individual Gaussians
            matches = []
            for i in range(min(top_k, len(indices))):
                matches.append(QueryMatch(
                    gaussian_indices=[int(indices[i])],
                    centroid=positions[i],
                    similarity_score=float(scores[i]),
                    bbox_3d=None,
                ))

        logger.info(
            "Query '%s': %d matches found, best similarity: %.3f.",
            text, len(matches), max_sim,
        )

        return QueryResult(
            query_text=text,
            match_found=True,
            matches=matches,
            max_similarity=max_sim,
            message=f"Found {len(matches)} matching region(s) for '{text}'.",
            query_embedding_dim=self.scene.semantic_dim,
        )

    def _cluster_matches(
        self,
        positions: np.ndarray,
        scores: np.ndarray,
        indices: np.ndarray,
        top_k: int,
    ) -> list[QueryMatch]:
        """
        Cluster spatially close Gaussians into coherent regions using DBSCAN.
        Returns up to top_k clusters, sorted by best similarity score.
        """
        from sklearn.cluster import DBSCAN

        clustering = DBSCAN(
            eps=self.config.spatial_clustering.eps,
            min_samples=self.config.spatial_clustering.min_samples,
        ).fit(positions)

        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        clusters: list[QueryMatch] = []

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_positions = positions[cluster_mask]
            cluster_scores = scores[cluster_mask]
            cluster_indices = indices[cluster_mask]

            centroid = cluster_positions.mean(axis=0)
            best_score = float(cluster_scores.max())
            min_corner = cluster_positions.min(axis=0)
            max_corner = cluster_positions.max(axis=0)

            obb_3d = None
            if len(cluster_positions) >= 4:
                # OBB computation using PCA
                centered = cluster_positions - centroid
                cov = np.cov(centered, rowvar=False)
                # Fallback to Identity if covariance is ill-conditioned
                if np.isnan(cov).any() or np.isinf(cov).any():
                    cov = np.eye(3)
                evals, evecs = np.linalg.eigh(cov)
                # Sort descending
                idx = np.argsort(evals)[::-1]
                evecs = evecs[:, idx]
                
                # Project points onto eigenvectors
                proj = centered @ evecs
                
                min_extents = proj.min(axis=0)
                max_extents = proj.max(axis=0)
                
                # Refine centroid
                center_offset = (min_extents + max_extents) / 2.0
                obb_center = centroid + evecs @ center_offset
                half_extents = (max_extents - min_extents) / 2.0
                
                obb_3d = {
                    "center": obb_center.tolist(),
                    "axes": evecs.tolist(),
                    "extents": half_extents.tolist()
                }

            clusters.append(QueryMatch(
                gaussian_indices=cluster_indices.tolist(),
                centroid=centroid,
                similarity_score=best_score,
                bbox_3d=(min_corner, max_corner),
                obb_3d=obb_3d,
            ))

        # Sort by best similarity score (highest first)
        clusters.sort(key=lambda c: c.similarity_score, reverse=True)

        # If no clusters formed, return top individual matches
        if not clusters:
            for i in range(min(top_k, len(indices))):
                clusters.append(QueryMatch(
                    gaussian_indices=[int(indices[i])],
                    centroid=positions[i],
                    similarity_score=float(scores[i]),
                    bbox_3d=None,
                ))

        return clusters[:top_k]
