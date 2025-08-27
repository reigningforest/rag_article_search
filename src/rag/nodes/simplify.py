"""
Abstract simplification functions for the RAG workflow.
"""

from typing import Dict, Any
from tqdm import tqdm

from ..state import RAGState
from ...connections.gemini_query import query_gemini
from ...connections.logger import get_shared_logger

logger = get_shared_logger(__name__)


def create_simplify_abstracts_node(gemini_model, simplifier_prompt: str):
    """Create an abstract simplification node function using Gemini."""

    def simplify_abstracts_node(state: RAGState) -> Dict[str, Any]:
        documents = state["documents"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("simplify_abstracts")

        logger.info(f"Enhancing {len(documents)} abstracts with simplified versions")

        enhanced_documents = []
        for doc in tqdm(documents, desc="Simplifying abstracts"):
            # Format the prompt with the abstract
            formatted_prompt = simplifier_prompt.format(abstract=doc["text"])

            # Get simplified version of the abstract using Gemini
            simplified_text = query_gemini(gemini_model, formatted_prompt)

            # Create a new document that contains both original and simplified text
            enhanced_doc = doc.copy()
            enhanced_doc["text"] = (
                f"ORIGINAL TEXT:\n{doc['text']}\n\nSIMPLIFIED VERSION:\n{simplified_text}"
            )

            enhanced_documents.append(enhanced_doc)

        print(f"Enhanced {len(enhanced_documents)} abstracts with simplified versions")

        return {
            "simplified_documents": enhanced_documents,
            "current_step": "abstracts_enhanced",
            "last_completed_step": "simplify_abstracts",
        }

    return simplify_abstracts_node