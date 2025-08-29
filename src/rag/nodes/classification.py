"""
Classification and routing functions for the RAG workflow.
"""

from typing import Dict, Any, Literal

from ..state import RAGState
from ...connections import query_client, get_shared_logger

logger = get_shared_logger(__name__)


def create_classify_node(client: Any, classification_prompt: str, model: str):
    """Create a classification node function using Gemini."""

    def classify_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("classify")

        logger.info("Classifying query")

        # Format the prompt with the query
        formatted_prompt = classification_prompt.format(query=query)

        # Query Gemini directly
        content = query_client(client, formatted_prompt, model).lower().strip()
        needs_arxiv = content == "yes"

        logger.info(f"Classification result: {needs_arxiv}")

        return {
            "needs_arxiv": needs_arxiv,
            "current_step": "classification_complete",
            "last_completed_step": "classify",
        }

    return classify_node


def route_based_on_classification(
    state: RAGState,
) -> Literal["rewrite_query", "direct_answer"]:
    """Router function for conditional edges."""
    if state["needs_arxiv"]:
        logger.info("Query needs ArXiv data, routing to rewrite_query")
        return "rewrite_query"
    else:
        logger.info("Query doesn't need ArXiv data, routing to direct_answer")
        return "direct_answer"
