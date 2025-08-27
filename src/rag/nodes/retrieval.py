"""
Document retrieval functions for the RAG workflow.
"""

from typing import Dict, Any

from ..state import RAGState


def create_retrieve_node(splits, index, embedder, top_k: int):
    """Create a document retrieval node function."""

    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        rewrite = state["rewrite"]

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("retrieve")

        print("Retrieving documents for concatenated query and rewrite")

        # Concatenate query with rewrite
        concatenated_query = query + " " + rewrite
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