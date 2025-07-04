"""
Graph visualization utilities.
"""

import os
from langchain_core.runnables.graph import MermaidDrawMethod


def visualize_graph(graph, output_path: str) -> bool:
    """
    Visualize the graph and save it as a PNG file.

    Args:
        graph: The compiled LangGraph workflow
        output_path (str): Path to save the visualization

    Returns:
        bool: True if successful, False otherwise
    """
    # Check if the output path already exists
    if os.path.exists(output_path):
        print(
            f"File '{output_path}' already exists. Please delete it or choose a different path."
        )
        return False

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
