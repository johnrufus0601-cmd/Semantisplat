import argparse
import sys
import json
from pathlib import Path
from rich.console import Console

from semantisplat.config import load_config
from semantisplat.retrieval.query_engine import SemanticQueryEngine
from semantisplat.api.routes.query import _get_engine

console = Console()

def evaluate(scene_id: str, queries: list[str]):
    console.print(f"\n[bold cyan]Evaluating SemanticSplat on Scene: {scene_id}[/bold cyan]")
    
    try:
        engine = _get_engine(scene_id)
    except Exception as e:
        console.print(f"[red]Failed to load scene {scene_id}: {e}[/red]")
        sys.exit(1)

    for query in queries:
        console.print(f"\n[bold yellow]Query:[/bold yellow] '{query}'")
        result = engine.query(query, top_k=5)
        
        if not result.match_found:
            console.print(f"  [red]No match found.[/red] Max similarity: {result.max_similarity:.3f}")
        else:
            console.print(f"  [green]Match found![/green] Max similarity: {result.max_similarity:.3f}")
            for i, match in enumerate(result.matches):
                console.print(f"    Region {i+1}: Score = {match.similarity_score:.3f}, Gaussians = {len(match.gaussian_indices)}")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SemanticSplat Queries")
    parser.add_argument("--scene", "-s", required=True, help="Scene ID to evaluate")
    parser.add_argument("--queries", "-q", nargs="+", default=["find the blue coffee mug", "place to sit", "wooden table"], help="List of text queries")
    args = parser.parse_args()
    
    evaluate(args.scene, args.queries)
