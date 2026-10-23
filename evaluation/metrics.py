"""
Evaluation metrics for SemantiSplat retrieval quality.

Computes:
  - Top-K Retrieval Accuracy: Is the correct object in the top-K results?
  - Mean Reciprocal Rank (MRR): At what rank does the correct result appear?
  - Localization IoU: Overlap between retrieved Gaussian cluster and GT region.
"""

from __future__ import annotations

import numpy as np


def top_k_accuracy(
    predicted_indices: list[list[int]],
    ground_truth_indices: list[set[int]],
    k: int = 5,
) -> float:
    """
    Compute top-K retrieval accuracy across a set of queries.

    For each query, check if any ground-truth Gaussian index appears
    in the top-K predicted indices.

    Args:
        predicted_indices: List of ranked predicted index lists per query.
        ground_truth_indices: List of ground-truth index sets per query.
        k: Number of top predictions to consider.

    Returns:
        Accuracy in [0, 1].
    """
    if not predicted_indices:
        return 0.0

    hits = 0
    for pred, gt in zip(predicted_indices, ground_truth_indices):
        top_k_pred = set(pred[:k])
        if top_k_pred & gt:  # intersection
            hits += 1

    return hits / len(predicted_indices)


def mean_reciprocal_rank(
    predicted_indices: list[list[int]],
    ground_truth_indices: list[set[int]],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) across queries.

    MRR = (1/N) * Σ (1 / rank_of_first_correct_result)

    Args:
        predicted_indices: List of ranked predicted index lists per query.
        ground_truth_indices: List of ground-truth index sets per query.

    Returns:
        MRR in [0, 1].
    """
    if not predicted_indices:
        return 0.0

    reciprocal_ranks = []
    for pred, gt in zip(predicted_indices, ground_truth_indices):
        rr = 0.0
        for rank, idx in enumerate(pred, start=1):
            if idx in gt:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def localization_iou_3d(
    predicted_bbox: tuple[np.ndarray, np.ndarray],
    ground_truth_bbox: tuple[np.ndarray, np.ndarray],
) -> float:
    """
    Compute 3D IoU between two axis-aligned bounding boxes.

    Args:
        predicted_bbox: (min_corner, max_corner) each (3,).
        ground_truth_bbox: (min_corner, max_corner) each (3,).

    Returns:
        IoU in [0, 1].
    """
    pred_min, pred_max = predicted_bbox
    gt_min, gt_max = ground_truth_bbox

    # Intersection
    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)

    # Union
    pred_dims = np.maximum(0, pred_max - pred_min)
    gt_dims = np.maximum(0, gt_max - gt_min)
    pred_vol = np.prod(pred_dims)
    gt_vol = np.prod(gt_dims)
    union_vol = pred_vol + gt_vol - inter_vol

    if union_vol <= 0:
        return 0.0

    return float(inter_vol / union_vol)


def compute_all_metrics(
    predicted_indices: list[list[int]],
    ground_truth_indices: list[set[int]],
    predicted_bboxes: list[tuple[np.ndarray, np.ndarray] | None] | None = None,
    ground_truth_bboxes: list[tuple[np.ndarray, np.ndarray] | None] | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        predicted_indices: Ranked predicted Gaussian indices per query.
        ground_truth_indices: Ground-truth Gaussian index sets per query.
        predicted_bboxes: Optional predicted 3D bounding boxes per query.
        ground_truth_bboxes: Optional ground-truth 3D bounding boxes per query.
        k_values: List of K values for top-K accuracy (default: [1, 3, 5, 10]).

    Returns:
        Dict of all computed metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    results = {}

    # Top-K accuracy for each K
    for k in k_values:
        results[f"top_{k}_accuracy"] = top_k_accuracy(
            predicted_indices, ground_truth_indices, k
        )

    # MRR
    results["mrr"] = mean_reciprocal_rank(predicted_indices, ground_truth_indices)

    # Localization IoU
    if predicted_bboxes and ground_truth_bboxes:
        ious = []
        for pred_bb, gt_bb in zip(predicted_bboxes, ground_truth_bboxes):
            if pred_bb is not None and gt_bb is not None:
                ious.append(localization_iou_3d(pred_bb, gt_bb))
        if ious:
            results["mean_iou"] = float(np.mean(ious))
            results["median_iou"] = float(np.median(ious))

    return results
