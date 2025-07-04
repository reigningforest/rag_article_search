"""
RAG Graph builder for the LangGraph workflow.
"""

from langgraph.graph import StateGraph, START, END
from .state import RAGState
from .nodes import (
    create_classify_node,
    create_rewrite_node,
    create_retrieve_node,
    create_simplify_abstracts_node,
    create_generate_response_node,
    create_direct_response_node,
    route_based_on_classification,
)


def build_rag_graph(splits, index, gemini_llm, embedder, config, gemini_model):
    """
    Build the RAG graph for the LangGraph system.

    Args:
        splits: Data splits DataFrame
        index: Pinecone index
        gemini_llm: Gemini model for main LLM operations
        embedder: Embedding model
        config: Configuration dictionary
        gemini_model: Gemini model for abstract simplification

    Returns:
        Compiled LangGraph workflow
    """
    # Extract prompts and parameters from config
    classification_prompt = config["classification_prompt"]
    rewrite_prompt = config["rewrite_prompt"]
    final_prompt = config["final_prompt"]
    simplifier_prompt = config["hugging_face_template"]
    top_k = config["top_k"]

    # Create node functions
    classify_node = create_classify_node(gemini_llm, classification_prompt)
    rewrite_node = create_rewrite_node(gemini_llm, rewrite_prompt)
    retrieve_node = create_retrieve_node(splits, index, embedder, top_k)
    simplify_abstracts_node = create_simplify_abstracts_node(
        gemini_model, simplifier_prompt
    )
    generate_response_node = create_generate_response_node(gemini_llm, final_prompt)
    direct_response_node = create_direct_response_node(gemini_llm)

    # Create the StateGraph
    workflow = StateGraph(RAGState)

    # Add nodes to the graph
    workflow.add_node("classify", classify_node)
    workflow.add_node("rewrite_query", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("simplify_abstracts", simplify_abstracts_node)
    workflow.add_node("generate_rag_response", generate_response_node)
    workflow.add_node("direct_answer", direct_response_node)

    # Add edges to connect nodes
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_based_on_classification,
        {"rewrite_query": "rewrite_query", "direct_answer": "direct_answer"},
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "simplify_abstracts")
    workflow.add_edge("simplify_abstracts", "generate_rag_response")
    workflow.add_edge("generate_rag_response", END)
    workflow.add_edge("direct_answer", END)

    # Compile the graph
    graph = workflow.compile()

    return graph
