import argparse
import gradio as gr
import requests
import json
import plotly.graph_objects as go
import numpy as np

def query_scene(scene_id, query_text, top_k, threshold):
    url = "http://localhost:8000/api/query"
    payload = {
        "scene_id": scene_id,
        "text": query_text,
        "top_k": int(top_k),
        "similarity_threshold": float(threshold)
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return f"Error: {str(e)}", None

    if not data["match_found"]:
        return data["message"], None
        
    # Format text output
    lines = [f"Matches found: {len(data['matches'])}", f"Max Similarity: {data['max_similarity']:.3f}\n"]
    for i, match in enumerate(data['matches']):
        lines.append(f"Region {i+1}: Score={match['similarity_score']:.3f}, Gaussians={len(match['gaussian_indices'])}")
        
    # Create 3D scatter plot of the matched centroids
    fig = go.Figure()
    
    x, y, z = [], [], []
    scores = []
    
    for match in data['matches']:
        centroid = match['centroid']
        x.append(centroid[0])
        y.append(centroid[1])
        z.append(centroid[2])
        scores.append(match['similarity_score'])
        
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=10,
            color=scores,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Similarity')
        ),
        text=[f"Score: {s:.3f}" for s in scores]
    ))
    
    fig.update_layout(
        title="Matched Gaussian Regions (Spatial Centroids)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    
    return "\n".join(lines), fig

def create_ui():
    with gr.Blocks(title="SemanticSplat Open-Vocabulary Search") as app:
        gr.Markdown("# SemanticSplat Query Interface")
        gr.Markdown("Search your trained 3D Gaussian Splatting scenes using natural language.")
        
        with gr.Row():
            with gr.Column():
                scene_id = gr.Textbox(label="Scene ID", placeholder="e.g. my_room")
                query_text = gr.Textbox(label="Search Query", placeholder="e.g. find the blue coffee mug")
                
                with gr.Row():
                    top_k = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Top K Matches")
                    threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, step=0.01, label="Similarity Threshold")
                    
                search_btn = gr.Button("Search", variant="primary")
                
                output_text = gr.Textbox(label="Results Summary", lines=5)
                
            with gr.Column():
                output_plot = gr.Plot(label="3D Match Heatmap")
                
        search_btn.click(
            query_scene,
            inputs=[scene_id, query_text, top_k, threshold],
            outputs=[output_text, output_plot]
        )
        
        gr.Markdown("Note: For full interactive 3D navigation, open the WebGL viewer served at `http://localhost:8000/`")
        
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port)
