"""
Coverage Heatmap Generator for SemantiSplat.
Transforms camera trajectories and detected 3D objects into a rich,
metric 2D floorplan coverage density map.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)


def generate_coverage_heatmap(
    sparse_model_dir: str | Path | None,
    output_path: str | Path,
    detected_objects: Optional[List[Dict[str, Any]]] = None,
    progress_callback = None,
) -> Path:
    """
    Generates a camera-coverage & spatial density heatmap.
    Overlays camera poses, viewing frustums, and detected 3D object footprints on the floor plane.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cam_positions = []
    cam_directions = []
    points3d = []

    # 1. Try reading COLMAP reconstruction if available
    if sparse_model_dir:
        sparse_dir = Path(sparse_model_dir)
        if sparse_dir.exists():
            try:
                import pycolmap
                recon = pycolmap.Reconstruction(str(sparse_dir))
                for _, image in recon.images.items():
                    pose = image.cam_from_world()
                    R = pose.rotation.matrix()
                    t = pose.translation
                    pos = -R.T @ t
                    cam_positions.append(pos)
                    # Forward direction is 3rd column of R^T (viewing ray into scene)
                    forward = R.T[:, 2]
                    cam_directions.append(forward)

                for _, pt in recon.points3D.items():
                    points3d.append(pt.xyz)
            except Exception as e:
                logger.warning("Could not read COLMAP sparse model: %s", e)

    if progress_callback:
        progress_callback(0.4)

    # 2. Extract object footprints if detected objects are available
    obj_boxes = []
    if detected_objects:
        for obj in detected_objects:
            if isinstance(obj, dict):
                center = obj.get("center", [0, 0, 0])
                size = obj.get("size", [0.5, 0.5, 0.5])
                label = obj.get("label", "Object")
                yaw = obj.get("yaw", 0.0)
                color = obj.get("color", [0.2, 0.8, 0.4])
                obj_boxes.append({
                    "x": center[0],
                    "z": center[2],
                    "width": size[0],
                    "depth": size[2] if len(size) > 2 else size[1],
                    "label": label,
                    "yaw": yaw,
                    "color": color,
                })

    # 3. Fallback coordinate system if no COLMAP points (e.g. single image mode)
    if len(cam_positions) == 0:
        cam_positions = np.array([[0.0, 0.0, 0.0]])
        cam_directions = np.array([[0.0, 0.0, 1.0]])
    else:
        cam_positions = np.array(cam_positions)
        cam_directions = np.array(cam_directions)

    # Determine ground plane: In standard vision coordinates, X is horizontal, Z is depth.
    # We plot (X, Z) as the floorplan.
    cam_x = cam_positions[:, 0]
    cam_z = cam_positions[:, 2]

    if len(points3d) > 0:
        pts = np.array(points3d)
        pts_x = pts[:, 0]
        pts_z = pts[:, 2]
        # Filter extreme outliers
        x_min, x_max = np.percentile(pts_x, 2), np.percentile(pts_x, 98)
        z_min, z_max = np.percentile(pts_z, 2), np.percentile(pts_z, 98)
    else:
        # Use object footprints and camera positions to frame the floorplan
        all_x = list(cam_x) + [b["x"] for b in obj_boxes]
        all_z = list(cam_z) + [b["z"] for b in obj_boxes]
        x_min, x_max = min(all_x) - 1.2, max(all_x) + 1.2
        z_min, z_max = min(all_z) - 0.5, max(all_z) + 1.5

    pad_x = max(0.8, (x_max - x_min) * 0.2)
    pad_z = max(0.8, (z_max - z_min) * 0.2)
    xmin, xmax = float(x_min - pad_x), float(x_max + pad_x)
    zmin, zmax = float(z_min - pad_z), float(z_max + pad_z)

    # 4. Compute 2D Density Grid (Gaussian smoothed)
    bins = 150
    # Collect all density points (cameras + points + object centers)
    density_x = list(cam_x)
    density_z = list(cam_z)
    if len(points3d) > 0:
        density_x.extend(pts[:, 0])
        density_z.extend(pts[:, 2])
    for b in obj_boxes:
        density_x.extend([b["x"]] * 15)
        density_z.extend([b["z"]] * 15)

    heatmap, xedges, yedges = np.histogram2d(
        density_x, density_z,
        bins=bins,
        range=[[xmin, xmax], [zmin, zmax]]
    )

    # Smooth density with Gaussian filter
    heatmap_smoothed = ndimage.gaussian_filter(heatmap, sigma=3.5).T

    # 5. Plotting Figure with High Aesthetic Standards
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160, facecolor="#0b0f19")
    ax.set_facecolor("#0b0f19")

    # Display smooth density map
    im = ax.imshow(
        heatmap_smoothed,
        extent=[xmin, xmax, zmin, zmax],
        origin="lower",
        cmap="turbo",
        alpha=0.68,
        aspect="equal",
        interpolation="bicubic",
    )

    # Add subtle grid lines every 0.5m
    ax.grid(True, which="both", color="#334155", linestyle="--", linewidth=0.6, alpha=0.5)

    # 6. Plot Camera Trajectory & Directional Cones
    if len(cam_x) > 1:
        ax.plot(cam_x, cam_z, color="#38bdf8", linestyle="-", linewidth=1.5, alpha=0.7, label="Camera Path")
    
    # Plot camera dots
    ax.scatter(cam_x, cam_z, s=36, color="#0284c7", edgecolor="#ffffff", linewidth=1.0, zorder=5, label=f"Cameras ({len(cam_x)})")

    # Draw viewing direction cones/arrows for key cameras
    step = max(1, len(cam_x) // 10)
    for idx in range(0, len(cam_x), step):
        cx, cz = cam_x[idx], cam_z[idx]
        if len(cam_directions) > idx:
            dx, dz = cam_directions[idx][0] * 0.45, cam_directions[idx][2] * 0.45
            ax.arrow(cx, cz, dx, dz, head_width=0.15, head_length=0.15, fc="#38bdf8", ec="#ffffff", alpha=0.85, zorder=6)

    # 7. Draw Detected Object Footprints (2D OBBs)
    for b in obj_boxes:
        bx, bz = b["x"], b["z"]
        bw, bd = b["width"], b["depth"]
        lbl = b["label"]
        col = b["color"]

        # Draw rotated rectangle footprint
        rect = patches.Rectangle(
            (bx - bw / 2, bz - bd / 2),
            bw, bd,
            angle=b["yaw"],
            rotation_point=(bx, bz),
            linewidth=2.0,
            edgecolor=col,
            facecolor=(*col, 0.22),
            zorder=7,
        )
        ax.add_patch(rect)

        # Label above object
        ax.text(
            bx, bz,
            f"{lbl.upper()}\n{bw:.1f}×{bd:.1f}m",
            color="#ffffff",
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#1e293b", ec=col, alpha=0.85, lw=1.2),
            zorder=8,
        )

    # 8. Titles and Labels
    ax.set_title("Room Coverage & 3D Object Density Map", fontsize=14, fontweight="bold", pad=12, color="#f8fafc")
    ax.set_xlabel("Lateral Width X (meters)", fontsize=10, labelpad=8, color="#cbd5e1")
    ax.set_ylabel("Depth Z (meters)", fontsize=10, labelpad=8, color="#cbd5e1")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spatial Coverage Density", color="#cbd5e1", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#cbd5e1", labelcolor="#cbd5e1")

    # Summary box
    room_w = max(0.5, xmax - xmin)
    room_d = max(0.5, zmax - zmin)
    summary_text = f"Span: {room_w:.1f}m × {room_d:.1f}m\nObjects: {len(obj_boxes)}\nViews: {len(cam_x)}"
    ax.text(
        0.02, 0.98, summary_text,
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#0f172a", ec="#475569", alpha=0.88, lw=1.0),
        color="#38bdf8",
        zorder=9,
    )

    ax.legend(loc="upper right", framealpha=0.7, facecolor="#0f172a", edgecolor="#475569", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=160, facecolor=fig.get_facecolor())
    plt.close()

    if progress_callback:
        progress_callback(1.0)

    logger.info("Coverage heatmap saved to %s", output_path)
    return output_path
