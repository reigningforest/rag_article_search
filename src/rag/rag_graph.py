"""
RAG Graph builder for the LangGraph workflow.
"""

from langgraph.graph import StateGraph, START, END
from .state import RAGState
from .nodes.classification import (
    create_classify_node,
    route_based_on_classification,
)
from .nodes.retrieval import create_retrieve_node
from .nodes.rewrite import create_rewrite_node
from .nodes.simplify import create_simplify_node
from .nodes.response_generation import (
    create_generate_response_node,
    create_direct_response_node,
)


def build_rag_graph(splits, index, client, embedder, config):
    """
    Build the RAG graph for the LangGraph system.

    Args:
        splits: Data splits DataFrame
        index: Pinecone index
        client: Gemini client
        embedder: Embedding model
        config: Configuration dictionary

    Returns:
        Compiled LangGraph workflow
    """
    # Extract prompts and parameters from config
    classification_prompt = config["classification_prompt"]
    rewrite_prompt = config["rewrite_prompt"]
    final_prompt = config["final_prompt"]
    simplifier_prompt = config["simplify_prompt"]
    top_k = config["top_k"]
    model = config["gemini_model_name"]

    # Create node functions
    classify_node = create_classify_node(client, classification_prompt, model)
    rewrite_node = create_rewrite_node(client, rewrite_prompt, model)
    retrieve_node = create_retrieve_node(splits, index, embedder, top_k)
    simplify_abstracts_node = create_simplify_node(
        client, simplifier_prompt, model
    )
    generate_response_node = create_generate_response_node(client, final_prompt, model)
    direct_response_node = create_direct_response_node(client, model)

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
    workflow.add_edge("retrieve", "generate_rag_response")
    workflow.add_edge("generate_rag_response", "simplify_abstracts")
    workflow.add_edge("simplify_abstracts", END)
    workflow.add_edge("direct_answer", END)

    # Compile the graph
    graph = workflow.compile()

    return graph
