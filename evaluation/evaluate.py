"""
SemantiSplat Evaluation Script.

Evaluates retrieval accuracy and localization quality on a scene
with ground-truth annotations (e.g. Replica or ScanNet).

Usage:
    python evaluation/evaluate.py --scene data/scenes/replica_room0 \\
                                  --queries evaluation/queries_replica.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_queries(queries_path: str | Path) -> list[dict]:
    """
    Load evaluation queries from a JSON file.

    Expected format:
    [
        {
            "text": "blue chair",
            "ground_truth_indices": [100, 101, 102, ...],
            "ground_truth_bbox": {"min": [x,y,z], "max": [x,y,z]}
        },
        ...
    ]
    """
    path = Path(queries_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    scene_dir: str | Path,
    queries_path: str | Path,
    top_k: int = 5,
    device: str = "cuda",
) -> dict:
    """
    Run the full evaluation pipeline.

    1. Load trained scene and compressor
    2. Load evaluation queries with ground truth
    3. Run each query through the retrieval engine
    4. Compute metrics (top-K accuracy, MRR, IoU)

    Args:
        scene_dir: Path to trained scene directory (containing scene_final.ply, compressor.pt).
        queries_path: Path to JSON file with evaluation queries.
        top_k: Max K for retrieval.
        device: Torch device.

    Returns:
        Dict with all evaluation metrics.
    """
    scene_dir = Path(scene_dir)
    scene_path = scene_dir / "scene_final.ply"
    compressor_path = scene_dir / "compressor.pt"

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    # Load query engine
    from semantisplat.retrieval.query_engine import SemanticQueryEngine

    engine = SemanticQueryEngine.from_files(
        scene_path=scene_path,
        compressor_path=compressor_path,
        device=device,
    )

    # Load queries
    queries = load_queries(queries_path)
    logger.info("Loaded %d evaluation queries from %s.", len(queries), queries_path)

    # Run queries
    all_predicted_indices = []
    all_gt_indices = []
    all_predicted_bboxes = []
    all_gt_bboxes = []

    for q in queries:
        result = engine.query(q["text"], top_k=top_k)

        # Collect predicted indices
        pred_indices = []
        pred_bbox = None
        for match in result.matches:
            pred_indices.extend(match.gaussian_indices)
            if match.bbox_3d is not None and pred_bbox is None:
                pred_bbox = match.bbox_3d

        all_predicted_indices.append(pred_indices)
        all_gt_indices.append(set(q.get("ground_truth_indices", [])))

        # Bboxes
        all_predicted_bboxes.append(pred_bbox)
        gt_bb = q.get("ground_truth_bbox")
        if gt_bb:
            all_gt_bboxes.append((
                np.array(gt_bb["min"]),
                np.array(gt_bb["max"]),
            ))
        else:
            all_gt_bboxes.append(None)

    # Compute metrics
    metrics = compute_all_metrics(
        predicted_indices=all_predicted_indices,
        ground_truth_indices=all_gt_indices,
        predicted_bboxes=all_predicted_bboxes,
        ground_truth_bboxes=all_gt_bboxes,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("Scene: %s", scene_dir)
    logger.info("Queries: %d", len(queries))
    for key, value in metrics.items():
        logger.info("  %-20s: %.4f", key, value)
    logger.info("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SemantiSplat Evaluation")
    parser.add_argument("--scene", required=True, help="Path to trained scene directory")
    parser.add_argument("--queries", required=True, help="Path to queries JSON file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    metrics = run_evaluation(args.scene, args.queries, args.top_k, args.device)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Results saved to %s.", output_path)


if __name__ == "__main__":
    main()
