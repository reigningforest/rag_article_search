"""
Abstract simplification functions for the RAG workflow.
"""

from typing import Dict, Any
from tqdm import tqdm

from ..state import RAGState
from ...connections.gemini_query import query_client
from ...connections.logger import get_shared_logger

logger = get_shared_logger(__name__)


def create_simplify_node(client: Any, simplifier_prompt: str, model: str):
    """Create an interactive abstract simplification node function using Gemini."""

    def simplify_node(state: RAGState) -> Dict[str, Any]:
        documents = state["documents"]
        simplified_documents = state.get("simplified_documents", [])

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("simplify_abstracts")

        # Display available documents for selection
        logger.info("\nAvailable documents for simplification:")
        for i, doc in enumerate(documents):
            # Show title if available, otherwise show first 100 chars of text
            title = doc.get("title", doc["text"][:100] + "...")
            simplified_status = " (ALREADY SIMPLIFIED)" if any(
                sd.get("original_index") == i for sd in simplified_documents
            ) else ""
            logger.info(f"{i}: {title}{simplified_status}")

        # Get user selection
        while True:
            selection = input("\nEnter document numbers to simplify (e.g., 0,1,3 or 'all'): ").strip()
            
            if selection.lower() == "all":
                selected_indices = list(range(len(documents)))
                break
            else:
                try:
                    # Parse comma-separated indices
                    selected_indices = [int(x.strip()) for x in selection.split(",") if x.strip()]
                    # Validate indices
                    if all(0 <= idx < len(documents) for idx in selected_indices):
                        break
                    else:
                        logger.warning(f"Please enter valid document numbers (0-{len(documents)-1})")
                except ValueError:
                    logger.warning("Please enter valid numbers separated by commas, or 'all'")

        # Filter out already simplified documents
        new_indices = [idx for idx in selected_indices 
                      if not any(sd.get("original_index") == idx for sd in simplified_documents)]
        
        if not new_indices:
            logger.info("All selected documents are already simplified.")
        else:
            logger.info(f"Simplifying {len(new_indices)} new abstracts")

            # Simplify selected documents
            for idx in tqdm(new_indices, desc="Simplifying selected abstracts"):
                doc = documents[idx]
                
                # Format the prompt with the abstract
                formatted_prompt = simplifier_prompt.format(abstract=doc["text"])

                # Get simplified version of the abstract using Gemini
                simplified_text = query_client(client, formatted_prompt, model)

                # Create a new document that contains both original and simplified text
                enhanced_doc = doc.copy()
                enhanced_doc["text"] = (
                    f"ORIGINAL TEXT:\n{doc['text']}\n\nSIMPLIFIED VERSION:\n{simplified_text}"
                )
                enhanced_doc["original_index"] = idx  # Track which original document this came from

                simplified_documents.append(enhanced_doc)

        # Ask if user wants to continue
        while True:
            continue_choice = input("\nContinue simplifying more abstracts? (yes/no): ").strip().lower()
            if continue_choice in ["yes", "no"]:
                break
            logger.warning("Please enter 'yes' or 'no'")

        continue_simplification = continue_choice == "yes"

        logger.debug(f"Simplified documents: {simplified_documents}")

        return {
            "simplified_documents": simplified_documents,
            "continue_simplification": continue_simplification,
            "current_step": "abstracts_enhanced",
            "last_completed_step": "simplify_abstracts",
        }

    return simplify_node