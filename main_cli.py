"""
CLI application for the Agentic RAG system.
"""

import os
import yaml
import torch
from dotenv import load_dotenv

from src.models import load_all_components
from src.rag import build_rag_graph
from src.connections.logger import get_shared_logger

logger = get_shared_logger(__name__)


def main():
    """Main application entry point."""
    # Load the config file
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the environment variables
    load_dotenv(dotenv_path=config["env_file"])

    # Load directories
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the device for embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU for embeddings.")
    else:
        logger.info(f"IMPORTANT! Using device: {device}")

    # Load all components
    splits, index, gemini_llm, embedder, gemini_model = load_all_components(
        config, data_dir, device, gemini_api_key
    )

    # Build the RAG graph
    rag_graph = build_rag_graph(
        splits, index, gemini_llm, embedder, config, gemini_model
    )

    logger.info("LangGraph RAG System initialized")
    logger.info("Enter 'exit' to quit the program.")

    # Interactive query loop
    while True:
        query_str = input("Enter a question or type 'exit': ")

        logger.info(f"User query: {query_str}")
        
        if query_str.lower() == "exit":
            break

        # Initialize state
        initial_state = {
            "query": query_str,
            "needs_arxiv": False,
            "rewrite": "",
            "documents": [],
            "simplified_documents": [],
            "response": "",
            "current_step": "initialized",
        }

        # Execute the graph with the query
        try:
            # Get the final result
            result = rag_graph.invoke(initial_state)

            # Log the response
            logger.info(f"\nResponse: {result['response']}")

            # Log the simplified documents
            logger.info("SIMPLIFIED SOURCE DOCUMENTS:")
            for i, doc in enumerate(result["simplified_documents"], 0):
                logger.info(f"\n---- Document {i+1} ----")
                logger.info(f"Title: {doc['title']}")
                logger.info(f"Date: {doc['date']}")
                logger.info(f"Simplified Abstract: {doc['text']}")

        except Exception as e:
            logger.error(f"Error during execution: {e}")


if __name__ == "__main__":
    main()
