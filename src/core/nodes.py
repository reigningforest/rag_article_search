"""
Individual node functions for the RAG graph workflow.
"""

from typing import Dict, Any, Literal
from tqdm import tqdm
import re

from ..core.state import RAGState
from ..connections.gemini_query import query_gemini


def create_classify_node(gemini_model, classification_prompt: str):
    """Create a classification node function using Gemini."""

    def classify_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("classify")

        print("Classifying query")

        # Format the prompt with the query
        formatted_prompt = classification_prompt.format(query=query)

        # Query Gemini directly
        content = query_gemini(gemini_model, formatted_prompt).lower().strip()
        needs_arxiv = content == "yes"

        print(f"Classification result: {needs_arxiv}")

        return {
            "needs_arxiv": needs_arxiv,
            "current_step": "classification_complete",
            "last_completed_step": "classify",
        }

    return classify_node


def create_rewrite_node(gemini_model, rewrite_prompt: str):
    """Create a query rewrite node function using Gemini."""

    def rewrite_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("rewrite_query")

        print("Generating query rewrites")

        # Format the prompt with the query
        formatted_prompt = rewrite_prompt.format(query=query)

        # Query Gemini directly
        rewrites_text = query_gemini(gemini_model, formatted_prompt)

        # Handle different possible response formats
        rewrites_text = rewrites_text.replace(
            "\\n", "\n"
        )  # Replace literal \n with actual newlines

        # Split by newlines and clean up
        rewrites = []
        for line in rewrites_text.split("\n"):
            line = line.strip()
            # Remove numbering (1., 2., etc.) and bullet points

            line = re.sub(r"^\d+\.\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)

            if (
                line and len(line) > 10
            ):  # Only keep non-empty lines with reasonable length
                rewrites.append(line)

        # If we still don't have multiple rewrites, try other splitting methods
        if len(rewrites) <= 1:
            # Try splitting by common sentence endings followed by capital letters

            sentences = re.split(r"[.!?]\s*(?=[A-Z])", rewrites_text)
            rewrites = [
                s.strip() + "."
                if not s.strip().endswith((".", "!", "?"))
                else s.strip()
                for s in sentences
                if s.strip() and len(s.strip()) > 10
            ]

        # If we still have only one item, at least ensure it's a list
        if len(rewrites) == 0:
            rewrites = [rewrites_text.strip()]

        print(f"Generated {len(rewrites)} query rewrites: {rewrites}")

        return {
            "rewrites": rewrites,
            "current_step": "rewrites_generated",
            "last_completed_step": "rewrite_query",
        }

    return rewrite_node


def create_retrieve_node(splits, index, embedder, top_k: int):
    """Create a document retrieval node function."""

    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        rewrites = state["rewrites"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("retrieve")

        print("Retrieving documents for concatenated query and rewrites")

        # Concatenate all queries into one string
        concatenated_query = query + " " + " ".join(rewrites)
        print(f"Concatenated query: {concatenated_query}")

        # Generate a single embedding for the concatenated query
        query_vector = embedder.embed_query(concatenated_query)

        # Retrieve documents for the concatenated query
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=False)

        all_results = results.matches

        # Convert to documents as before
        chunk_ids = [int(match.id) for match in all_results]
        docs_df = splits.iloc[chunk_ids]

        # Convert DataFrame to list of dictionaries
        documents = []
        for _, row in docs_df.iterrows():
            doc = {
                "title": row.title,
                "date": str(row.update_date.date()),
                "text": row.chunk_text,
            }
            documents.append(doc)

        print(f"Retrieved {len(documents)} documents")

        return {
            "documents": documents,
            "current_step": "documents_retrieved",
            "last_completed_step": "retrieve",
        }

    return retrieve_node


def create_simplify_abstracts_node(gemini_model, simplifier_prompt: str):
    """Create an abstract simplification node function using Gemini."""

    def simplify_abstracts_node(state: RAGState) -> Dict[str, Any]:
        documents = state["documents"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("simplify_abstracts")

        print(f"Enhancing {len(documents)} abstracts with simplified versions")

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
            "documents": enhanced_documents,
            "current_step": "abstracts_enhanced",
            "last_completed_step": "simplify_abstracts",
        }

    return simplify_abstracts_node


def create_generate_response_node(gemini_model, final_prompt: str):
    """Create a RAG response generation node function using Gemini."""

    def generate_response_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        documents = state["documents"]
        rewrites = state["rewrites"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("generate_rag_response")

        print(f"Generating RAG response using {len(documents)} documents")

        # Format documents
        formatted_docs = "\n\n".join(
            f"Title: {doc['title']}\nDate: {doc['date']}\n{doc['text']}"
            for doc in documents
        )

        formatted_rewrites = "\n- ".join(rewrites)

        # Format the final prompt with all variables
        formatted_prompt = final_prompt.format(
            query=query, context=formatted_docs, rewrites=formatted_rewrites
        )

        # Query Gemini directly
        response = query_gemini(gemini_model, formatted_prompt)

        return {
            "response": response,
            "current_step": "response_generated",
            "last_completed_step": "generate_rag_response",
        }

    return generate_response_node


def create_direct_response_node(gemini_model):
    """Create a direct response node function (without ArXiv) using Gemini."""

    def direct_response_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("direct_answer")

        print(f"Generating direct response for: {query}")

        # Query Gemini directly
        response = query_gemini(gemini_model, query)

        return {
            "response": response,
            "current_step": "response_generated",
            "last_completed_step": "direct_answer",
        }

    return direct_response_node


def route_based_on_classification(
    state: RAGState,
) -> Literal["rewrite_query", "direct_answer"]:
    """Router function for conditional edges."""
    if state["needs_arxiv"]:
        print("Query needs ArXiv data, routing to rewrite_query")
        return "rewrite_query"
    else:
        print("Query doesn't need ArXiv data, routing to direct_answer")
        return "direct_answer"
