"""
SemantiSplat — End-to-End Pipeline Orchestrator.

Runs the complete pipeline:
  Video → Frames → COLMAP → SAM → CLIP → 3D-GS → Trained Semantic Scene

Usage:
    python -m semantisplat.pipeline --config config/default.yaml --input video.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from semantisplat.config import SemantiSplatConfig, load_config, get_data_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()


def run_pipeline(
    video_path: str | Path,
    config: SemantiSplatConfig,
    scene_id: str | None = None,
) -> dict:
    """
    Run the complete SemantiSplat pipeline end-to-end.

    Args:
        video_path: Path to the input video file.
        config: Pipeline configuration.
        scene_id: Optional scene identifier. Derived from video name if None.

    Returns:
        Dict with pipeline results and timing info.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if scene_id is None:
        scene_id = video_path.stem

    console.print(f"\n[bold cyan]◎ SemantiSplat Pipeline[/bold cyan]")
    console.print(f"  Video:    {video_path}")
    console.print(f"  Scene ID: {scene_id}")
    console.print(f"  Device:   {config.device}")
    console.print()

    timings = {}
    results = {"scene_id": scene_id}

    # ================================================================
    # Stage 1: Frame Extraction
    # ================================================================
    console.print("[bold]Stage 1/6:[/bold] Frame Extraction 🎬")
    t0 = time.time()

    from semantisplat.ingestion.frame_extractor import extract_frames

    frames_dir = get_data_path(config, "frames", scene_id)
    extraction = extract_frames(video_path, frames_dir, config.ingestion)

    timings["frame_extraction"] = time.time() - t0
    results["frames"] = extraction.extracted_count
    console.print(f"  → {extraction.extracted_count} frames extracted "
                  f"({timings['frame_extraction']:.1f}s)\n")

    # ================================================================
    # Stage 2: COLMAP Pose Estimation
    # ================================================================
    console.print("[bold]Stage 2/6:[/bold] COLMAP Pose Estimation 📐")
    t0 = time.time()

    from semantisplat.ingestion.pose_estimator import estimate_poses

    colmap_dir = get_data_path(config, "colmap", scene_id)
    poses = estimate_poses(
        frames_dir, colmap_dir,
        extraction.extracted_count,
        config.ingestion.colmap,
    )

    timings["pose_estimation"] = time.time() - t0
    results["registered_images"] = poses.registered_images
    results["registration_rate"] = poses.registration_rate
    results["registration_warning"] = poses.registration_warning
    results["points_3d"] = poses.points3d_count

    status = "[green]✓[/green]" if not poses.registration_warning else "[yellow]⚠[/yellow]"
    console.print(f"  → {poses.registered_images}/{poses.total_images} images registered "
                  f"({poses.registration_rate:.0%}) {status}")
    console.print(f"  → {poses.points3d_count} 3D points "
                  f"({timings['pose_estimation']:.1f}s)\n")

    # ================================================================
    # Stage 3: Coverage Heatmap
    # ================================================================
    console.print("[bold]Stage 3/6:[/bold] Coverage Heatmap 🗺️")
    t0 = time.time()

    from semantisplat.reconstruction.coverage_heatmap import generate_coverage_heatmap
    
    heatmap_dir = get_data_path(config, "heatmap", scene_id)
    heatmap_path = heatmap_dir / "coverage_heatmap.png"
    
    try:
        generate_coverage_heatmap(poses.sparse_model_dir, heatmap_path)
        console.print(f"  → Heatmap generated: {heatmap_path.name}")
        results["coverage_heatmap_path"] = str(heatmap_path)
    except Exception as e:
        console.print(f"  → [yellow]Heatmap generation failed: {e}[/yellow]")
        results["coverage_heatmap_path"] = None

    timings["coverage_heatmap"] = time.time() - t0
    console.print(f"  → ({timings['coverage_heatmap']:.1f}s)\n")

    # ================================================================
    # Stage 4: SAM Segmentation
    # ================================================================
    console.print("[bold]Stage 4/6:[/bold] SAM Segmentation 🔬")
    t0 = time.time()

    from semantisplat.semantic.segmenter import segment_frames

    segmentation = segment_frames(
        extraction.frame_paths,
        config.semantic.sam,
        config.device,
        mixed_precision=config.semantic.mixed_precision,
    )

    timings["segmentation"] = time.time() - t0
    results["total_masks"] = segmentation.total_masks
    console.print(f"  → {segmentation.total_masks} masks across "
                  f"{segmentation.frames_processed} frames "
                  f"({timings['segmentation']:.1f}s)\n")

    # ================================================================
    # Stage 5: CLIP Embedding Extraction
    # ================================================================
    console.print("[bold]Stage 5/6:[/bold] CLIP Embedding Extraction 🧬")
    t0 = time.time()

    from semantisplat.semantic.embedder import extract_embeddings
    from semantisplat.semantic.feature_store import SemanticFeatureStore

    embeddings = extract_embeddings(
        segmentation.frame_masks,
        config.semantic.clip,
        config.device,
        mixed_precision=config.semantic.mixed_precision,
    )

    # Save semantic features
    semantic_dir = get_data_path(config, "semantic", scene_id)
    store = SemanticFeatureStore(semantic_dir)
    store.save(segmentation, embeddings)

    timings["embedding"] = time.time() - t0
    results["total_embeddings"] = embeddings.total_embeddings
    console.print(f"  → {embeddings.total_embeddings} embeddings "
                  f"(dim={embeddings.embedding_dim}) "
                  f"({timings['embedding']:.1f}s)\n")

    # ================================================================
    # Stage 6: 3D Gaussian Splatting Training
    # ================================================================
    console.print("[bold]Stage 6/6:[/bold] 3D Gaussian Splatting Training 🌐")
    t0 = time.time()

    from semantisplat.reconstruction.semantic_gaussian import initialize_gaussians
    from semantisplat.reconstruction.trainer import SemanticGaussianTrainer
    from semantisplat.ingestion.pose_estimator import read_points3d_binary, read_points3d_text
    
    # 1. Load initial points from COLMAP
    points3d_bin = poses.sparse_model_dir / "points3D.bin"
    points3d_txt = poses.sparse_model_dir / "points3D.txt"
    if points3d_bin.exists():
        xyz, rgb = read_points3d_binary(points3d_bin)
    elif points3d_txt.exists():
        xyz, rgb = read_points3d_text(points3d_txt)
    else:
        # Fallback if no points3d
        xyz = np.random.rand(10000, 3).astype(np.float32)
        rgb = np.random.rand(10000, 3).astype(np.float32)

    # 2. Build camera lists and load images
    images = []
    intrinsics_list = []
    extrinsics_list = []
    frame_indices = []

    import cv2
    import numpy as np

    for idx, frame_path in enumerate(extraction.frame_paths):
        img_name = frame_path.name
        if img_name in poses.poses:
            pose = poses.poses[img_name]
            cam = poses.cameras[pose.camera_id]
            
            # Load and normalize image
            img_bgr = cv2.imread(str(frame_path))
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Downsample again if needed (to match training limits if extractor didn't save them correctly)
            # Extractor already downsampled them, so we just use the loaded resolution
            H, W = img_rgb.shape[:2]

            images.append(img_rgb)
            frame_indices.append(idx)
            
            # Extract focal lengths and principal points (assuming SIMPLE_RADIAL or PINHOLE)
            # SIMPLE_PINHOLE: f, cx, cy
            # PINHOLE: fx, fy, cx, cy
            if cam.model == "SIMPLE_PINHOLE":
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
            else:
                # Default to fx, fy, cx, cy layout
                fx, fy = cam.params[0], cam.params[1]
                cx, cy = cam.params[2], cam.params[3]
                
            intrinsics_list.append({
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "width": W, "height": H
            })
            
            extrinsics_list.append({
                "R": pose.rotation_matrix,
                "t": pose.translation
            })

    if len(images) == 0:
        raise RuntimeError("No registered images found for training.")

    # 3. Initialize Gaussian Scene
    scene = initialize_gaussians(
        points3d=xyz,
        colors=rgb,
        semantic_dim=config.reconstruction.semantic_dim,
        device=config.device
    )

    # 4. Train
    trainer = SemanticGaussianTrainer(
        config=config.reconstruction,
        scene=scene,
        feature_store=store,
        images=images,
        camera_intrinsics=intrinsics_list,
        camera_extrinsics=extrinsics_list,
        frame_indices=frame_indices,
        device=config.device
    )
    
    reconstruction_dir = get_data_path(config, "reconstruction", scene_id)
    trained_scene = trainer.train(reconstruction_dir)
    results["trained_gaussians"] = trained_scene.num_gaussians
    console.print(f"  → Training finished. Final Gaussians: {trained_scene.num_gaussians}")

    timings["reconstruction"] = time.time() - t0

    # ================================================================
    # Summary
    # ================================================================
    total_time = sum(timings.values())

    console.print()
    table = Table(title="Pipeline Summary", show_header=True)
    table.add_column("Stage", style="bold")
    table.add_column("Result", style="cyan")
    table.add_column("Time", justify="right", style="green")

    table.add_row("Frame Extraction", f"{results['frames']} frames",
                  f"{timings['frame_extraction']:.1f}s")
    table.add_row("COLMAP Poses", f"{results['registered_images']} registered "
                  f"({results['registration_rate']:.0%})",
                  f"{timings['pose_estimation']:.1f}s")
    heatmap_res = "Generated" if results.get("coverage_heatmap_path") else "Failed"
    table.add_row("Coverage Heatmap", heatmap_res, f"{timings.get('coverage_heatmap', 0):.1f}s")
    table.add_row("SAM Masks", f"{results['total_masks']} masks",
                  f"{timings['segmentation']:.1f}s")
    table.add_row("CLIP Embeddings", f"{results['total_embeddings']} embeddings",
                  f"{timings['embedding']:.1f}s")
    table.add_row("3D-GS Training", f"{results.get('trained_gaussians', 0)} Gaussians",
                  f"{timings['reconstruction']:.1f}s")
    table.add_row("", "", "")
    table.add_row("[bold]Total[/bold]", "", f"[bold]{total_time:.1f}s[/bold]")

    console.print(table)
    console.print()

    results["timings"] = timings
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SemantiSplat — End-to-End Semantic 3D Scene Search Pipeline"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input video file")
    parser.add_argument("--config", "-c", default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--scene-id", default=None,
                        help="Scene identifier (default: derived from video name)")
    parser.add_argument("--device", default=None,
                        help="Override device (cuda/cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config.device = args.device

    try:
        results = run_pipeline(args.input, config, args.scene_id)
    except Exception as e:
        console.print(f"\n[bold red]Pipeline failed:[/bold red] {e}")
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
