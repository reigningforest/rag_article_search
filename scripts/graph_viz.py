"""
Graph visualization utilities.
"""

import os
import yaml
import torch
from dotenv import load_dotenv
from langchain_core.runnables.graph import MermaidDrawMethod


def visualize_graph(graph, output_path: str, overwrite: bool = False) -> bool:
    """
    Visualize the graph and save it as a PNG file.

    Args:
        graph: The compiled LangGraph workflow
        output_path (str): Path to save the visualization
        overwrite (bool): Whether to overwrite existing files

    Returns:
        bool: True if successful, False otherwise
    """
    # Check if the output path already exists and handle accordingly
    if os.path.exists(output_path):
        if not overwrite:
            # Create a new filename with timestamp
            base, ext = os.path.splitext(output_path)
            import time
            timestamp = int(time.time())
            output_path = f"{base}_{timestamp}{ext}"
            print(f"File already exists. Creating new file: {output_path}")
        else:
            print(f"Overwriting existing file: {output_path}")

    try:
        # Generate PNG data from the graph
        png_data = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API, background_color="white", padding=10
        )

        # Save the PNG data to a file
        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"Graph visualization saved as '{output_path}'")
        return True
    except Exception as e:
        print(f"Error generating graph visualization: {e}")
        return False


def main():
    """Main function to run graph visualization as a standalone script."""
    # Add project root to path for absolute imports
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Now use absolute imports
    from src.models import load_all_components
    from src.core import build_rag_graph
    
    # Load the config file
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load the environment variables
    load_dotenv(dotenv_path=config["env_file"])

    # Load directories
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]

    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("GEMINI_API_KEY not found in environment variables.")
        print("Please make sure you have a .env file with your GEMINI_API_KEY.")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the device for embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU for embeddings.")
    else:
        print(f"Using device: {device}")

    print("Loading components...")
    # Load all components
    splits, index, gemini_llm, embedder, gemini_model = load_all_components(
        config, data_dir, device, gemini_api_key
    )

    print("Building RAG graph...")
    # Build the RAG graph
    rag_graph = build_rag_graph(
        splits, index, gemini_llm, embedder, config, gemini_model
    )

    # Visualize the graph
    graph_output_path = os.path.join(output_dir, config["fine_tuned_graph_file_name"])
    print("Generating graph visualization...")
    
    success = visualize_graph(rag_graph, graph_output_path, overwrite=True)
    if success:
        print("Graph visualization completed successfully!")
        print(f"Output saved to: {graph_output_path}")
    else:
        print("Failed to generate graph visualization.")


if __name__ == "__main__":
    main()
