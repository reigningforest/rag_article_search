"""
Response generation functions for the RAG workflow.
"""

from typing import Dict, Any

from ..state import RAGState
from ...connections.gemini_query import query_client


def create_generate_response_node(client: Any, final_prompt: str, model: str):
    """Create a RAG response generation node function using Gemini."""

    def generate_response_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        documents = state["documents"]
        rewrite = state["rewrite"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("generate_rag_response")

        print(f"Generating RAG response using {len(documents)} documents")

        # Format documents
        formatted_docs = "\n\n".join(
            f"Title: {doc['title']}\nDate: {doc['date']}\n{doc['text']}"
            for doc in documents
        )

        # Format the final prompt with all variables
        formatted_prompt = final_prompt.format(
            query=query, context=formatted_docs, rewrite=rewrite
        )

        # Query Gemini directly
        response = query_client(client, formatted_prompt, model)

        return {
            "response": response,
            "current_step": "response_generated",
            "last_completed_step": "generate_rag_response",
        }

    return generate_response_node


def create_direct_response_node(client: Any, model: str):
    """Create a direct response node function (without ArXiv) using Gemini."""

    def direct_response_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("direct_answer")

        print(f"Generating direct response for: {query}")

        # Query Gemini directly
        response = query_client(client, query, model)

        return {
            "response": response,
            "current_step": "response_generated",
            "last_completed_step": "direct_answer",
        }

    return direct_response_node
