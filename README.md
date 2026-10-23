# SemantiSplat

Open-Vocabulary Semantic 3D Scene Search via Gaussian Splatting.

> **Video → COLMAP → SAM + CLIP → 3D Gaussian Splatting → Natural-Language Query → Highlighted 3D Region**

## Status

🚧 Under active development.

## Setup for RTX 5050 (8GB VRAM limit)

For memory constrained environments like the NVIDIA RTX 5050 (8GB), SemantiSplat uses an optimized configuration with mixed-precision (fp16/bf16), aggressive densification limits (pruning transparent Gaussians without cloning), lower SAM masks limits, and feature compression.

To run the pipeline within 7.2GB VRAM:
```bash
python -m semantisplat.pipeline --config config/rtx5050.yaml --input data/input/your_video.mp4
```

## Feature Fusion Design (Semantic 3DGS)

Our core research contribution is how we lift 2D CLIP features into 3D Gaussians:
1. **Compressor**: Instead of storing the full 512-dim CLIP vectors per Gaussian, we project them to a low-dimensional space (`semantic_dim = 16`).
2. **Joint Optimization**: During 3DGS training, we render the semantic features alongside RGB using the same alpha-blending rasterization process.
3. **Supervision**: The rendered semantic image is supervised using an L2 loss against the compressed ground-truth CLIP features from the SAM masks of the training views.
This approach significantly reduces memory usage while preserving rich semantic representation in the 3D scene.

## User Interface & Evaluation

Once trained, you can interact with the scene in two ways:

1. **Gradio Offline Query UI**:
```bash
python scripts/run_gradio_ui.py
```
Provides a simple web UI to type natural language queries, showing a 3D heatmap of matching Gaussians and cosine similarity metrics.

2. **Interactive 3D WebGL Viewer**:
```bash
python -m semantisplat.api.app
```
Starts a FastAPI server on port 8000. Open `http://localhost:8000` for real-time 3D navigation and highlighting of queried objects.

3. **Evaluation Script**:
```bash
python scripts/evaluate.py --scene your_video --queries "find the blue coffee mug" "place to sit"
```
