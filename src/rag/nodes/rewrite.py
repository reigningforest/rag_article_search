"""
Query processing and rewriting functions for the RAG workflow.
"""

from typing import Dict, Any

from ..state import RAGState
from ...connections.gemini_query import query_gemini


def create_rewrite_node(gemini_model, rewrite_prompt: str):
    """Create a query rewrite node function using Gemini."""

    def rewrite_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("rewrite_query")

        print("Generating query rewrite")

        # Format the prompt with the query
        formatted_prompt = rewrite_prompt.format(query=query)

        # Query Gemini directly
        rewrite = query_gemini(gemini_model, formatted_prompt)

        print(f"Generated query rewrite: {rewrite}")

        return {
            "rewrite": rewrite,
            "current_step": "rewrite_generated",
            "last_completed_step": "rewrite_query",
        }

    return rewrite_node
