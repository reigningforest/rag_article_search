"""
RAG State definition for the LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional, Callable


class RAGState(TypedDict):
    """State that will be passed between nodes in the RAG graph."""

    query: str
    needs_arxiv: bool
    rewrite: str
    documents: List[Dict[str, Any]]
    simplified_documents: List[Dict[str, Any]]
    response: str
    current_step: str
    progress_callback: Optional[Callable[[str], None]]  # For progress updates
    selected_for_simplification: List[int]
    continue_simplification: bool
