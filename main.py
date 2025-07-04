"""
Main application for the Agentic RAG system.
This is the refactored version using modular components with Gemini.
"""

import os
import yaml
import torch
from dotenv import load_dotenv

from src.models import load_all_components
from src.core import build_rag_graph
from src.visualization import visualize_graph


def main():
    """Main application entry point."""
    # Load the config file
    with open("config.yaml", "r") as file:
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

    # Load all components
    splits, index, gemini_llm, embedder, gemini_model = load_all_components(
        config, data_dir, device, gemini_api_key
    )

    # Build the RAG graph
    rag_graph = build_rag_graph(
        splits, index, gemini_llm, embedder, config, gemini_model
    )

    # Visualize the graph
    graph_output_path = os.path.join(output_dir, config["fine_tuned_graph_file_name"])
    visualize_graph(rag_graph, graph_output_path)

    print("LangGraph RAG System initialized")
    print("Enter 'exit' to quit the program.")

    # Interactive query loop
    while True:
        query_str = input("Enter a question or type 'exit': ")
        if query_str == "exit":
            break

        # Initialize state
        initial_state = {
            "query": query_str,
            "needs_arxiv": False,
            "rewrites": [],
            "documents": [],
            "response": "",
            "current_step": "initialized",
        }

        # Execute the graph with the query
        try:
            # Get the final result
            result = rag_graph.invoke(initial_state)

            # Print the response
            print("\nResponse:")
            print(result["response"])

        except Exception as e:
            print(f"Error during execution: {e}")


if __name__ == "__main__":
    main()
