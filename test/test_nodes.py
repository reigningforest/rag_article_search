"""
Tests for RAG graph node functions.
"""

from typing import cast
from unittest.mock import Mock, patch
import pandas as pd
from src.rag.state import RAGState
from src.rag.nodes import (
    create_classify_node,
    create_rewrite_node,
    create_retrieve_node,
    create_simplify_abstracts_node,
    create_generate_response_node,
    create_direct_response_node,
    route_based_on_classification,
)


def create_test_state(**kwargs) -> RAGState:
    """Helper to create test RAGState."""
    default_state = {
        "query": "",
        "needs_arxiv": False,
        "rewrites": [],
        "documents": [],
        "response": "",
        "current_step": "initialized",
    }
    default_state.update(kwargs)
    return cast(RAGState, default_state)


class TestClassifyNode:
    """Test query classification node functionality."""

    @patch("src.core.nodes.query_gemini")
    def test_classify_node_needs_arxiv(self, mock_query_gemini):
        """Test classification node when query needs arXiv retrieval."""
        # Setup
        mock_gemini_model = Mock()
        classification_prompt = "Does this query need arXiv papers? Query: {query}"
        mock_query_gemini.return_value = "yes"

        # Create node
        classify_node = create_classify_node(mock_gemini_model, classification_prompt)

        # Test state
        state = create_test_state(
            query="What are the latest advances in machine learning?"
        )

        # Execute
        result = classify_node(state)

        # Verify
        assert result["needs_arxiv"] is True
        assert result["current_step"] == "classification_complete"
        mock_query_gemini.assert_called_once_with(
            mock_gemini_model,
            "Does this query need arXiv papers? Query: What are the latest advances in machine learning?",
        )

    @patch("src.core.nodes.query_gemini")
    def test_classify_node_no_arxiv_needed(self, mock_query_gemini):
        """Test classification node when query doesn't need arXiv retrieval."""
        # Setup
        mock_gemini_model = Mock()
        classification_prompt = "Does this query need arXiv papers? Query: {query}"
        mock_query_gemini.return_value = "NO"  # Test case insensitive

        # Create node
        classify_node = create_classify_node(mock_gemini_model, classification_prompt)

        # Test state
        state = create_test_state(
            query="What is 2+2?",
            needs_arxiv=True,  # Should be overridden
        )

        # Execute
        result = classify_node(state)

        # Verify
        assert result["needs_arxiv"] is False
        assert result["current_step"] == "classification_complete"


class TestRewriteNode:
    """Test query rewrite node functionality."""

    @patch("src.core.nodes.query_gemini")
    def test_rewrite_node_multiple_rewrites(self, mock_query_gemini):
        """Test rewrite node generates multiple query variations."""
        # Setup
        mock_gemini_model = Mock()
        rewrite_prompt = "Rewrite this query in different ways: {query}"
        mock_query_gemini.return_value = (
            "What is machine learning?\nDefine machine learning\nExplain ML concepts"
        )

        # Create node
        rewrite_node = create_rewrite_node(mock_gemini_model, rewrite_prompt)

        # Test state
        state = create_test_state(
            query="What is machine learning?",
            needs_arxiv=True,
            current_step="classification_complete",
        )

        # Execute
        result = rewrite_node(state)

        # Verify
        expected_rewrites = [
            "What is machine learning?",
            "Define machine learning",
            "Explain ML concepts",
        ]
        assert result["rewrites"] == expected_rewrites
        assert result["current_step"] == "rewrites_generated"
        assert len(result["rewrites"]) == 3


class TestRetrieveNode:
    """Test document retrieval node functionality."""

    def test_retrieve_node_successful_retrieval(self):
        """Test successful document retrieval."""
        # Mock data
        mock_splits = pd.DataFrame(
            {
                "title": ["Paper 1", "Paper 2"],
                "update_date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")],
                "chunk_text": ["Content about ML", "Content about AI"],
            }
        )

        # Mock components
        mock_index = Mock()
        mock_embedder = Mock()

        # Mock embedder response
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Mock Pinecone response
        mock_match1 = Mock()
        mock_match1.id = "0"
        mock_match2 = Mock()
        mock_match2.id = "1"

        mock_results = Mock()
        mock_results.matches = [mock_match1, mock_match2]
        mock_index.query.return_value = mock_results

        # Create node
        retrieve_node = create_retrieve_node(
            mock_splits, mock_index, mock_embedder, top_k=5
        )

        # Test state
        state = create_test_state(
            query="machine learning",
            needs_arxiv=True,
            rewrites=["ML", "artificial intelligence"],
            current_step="rewrites_generated",
        )

        # Execute
        result = retrieve_node(state)

        # Verify
        assert len(result["documents"]) == 2
        assert result["current_step"] == "documents_retrieved"
        assert result["documents"][0]["title"] == "Paper 1"
        assert result["documents"][1]["title"] == "Paper 2"

        # Verify concatenated query was used
        expected_query = "machine learning ML artificial intelligence"
        mock_embedder.embed_query.assert_called_once_with(expected_query)
        mock_index.query.assert_called_once_with(
            vector=[0.1, 0.2, 0.3, 0.4, 0.5], top_k=5, include_metadata=False
        )


class TestSimplifyAbstractsNode:
    """Test abstract simplification node functionality."""

    @patch("src.core.nodes.query_gemini")
    def test_simplify_abstracts_node_execution(self, mock_query_gemini):
        """Test simplification of abstracts."""
        # Setup
        mock_gemini_model = Mock()
        simplification_prompt = "Simplify this abstract: {abstract}"
        mock_query_gemini.return_value = "Simplified abstract content"

        # Create node
        simplify_node = create_simplify_abstracts_node(
            mock_gemini_model, simplification_prompt
        )

        # Test state with documents
        state = create_test_state(
            query="machine learning",
            needs_arxiv=True,
            rewrites=["ML"],
            documents=[
                {
                    "title": "Test Paper",
                    "date": "2023-01-01",
                    "text": "Complex abstract content about neural networks",
                }
            ],
            current_step="documents_retrieved",
        )

        # Execute
        result = simplify_node(state)

        # Verify
        assert len(result["documents"]) == 1
        assert result["current_step"] == "abstracts_enhanced"

        enhanced_text = result["documents"][0]["text"]
        assert "ORIGINAL TEXT:" in enhanced_text
        assert "SIMPLIFIED VERSION:" in enhanced_text
        assert "Complex abstract content about neural networks" in enhanced_text
        assert "Simplified abstract content" in enhanced_text

        mock_query_gemini.assert_called_once_with(
            mock_gemini_model,
            "Simplify this abstract: Complex abstract content about neural networks",
        )


class TestGenerateResponseNode:
    """Test response generation node functionality."""

    @patch("src.core.nodes.query_gemini")
    def test_generate_response_node_with_context(self, mock_query_gemini):
        """Test response generation with document context."""
        # Setup
        mock_gemini_model = Mock()
        final_prompt = "Answer: {query}\nContext: {context}\nRewrites: {rewrites}"
        mock_query_gemini.return_value = "Generated response based on context"

        # Create node
        generate_node = create_generate_response_node(mock_gemini_model, final_prompt)

        # Test state
        state = create_test_state(
            query="What is machine learning?",
            needs_arxiv=True,
            rewrites=["Define ML", "Explain machine learning"],
            documents=[
                {
                    "title": "ML Paper",
                    "date": "2023-01-01",
                    "text": "ML is a subset of AI",
                }
            ],
            current_step="abstracts_enhanced",
        )

        # Execute
        result = generate_node(state)

        # Verify
        assert result["response"] == "Generated response based on context"
        assert result["current_step"] == "response_generated"

        # Verify prompt was formatted correctly
        call_args = mock_query_gemini.call_args[0][1]
        assert "What is machine learning?" in call_args
        assert "Title: ML Paper" in call_args
        assert "Define ML" in call_args


class TestDirectResponseNode:
    """Test direct response node functionality."""

    @patch("src.core.nodes.query_gemini")
    def test_direct_response_node_execution(self, mock_query_gemini):
        """Test direct response without document retrieval."""
        # Setup
        mock_gemini_model = Mock()
        mock_query_gemini.return_value = "Hello! How can I help you today?"

        # Create node
        direct_node = create_direct_response_node(mock_gemini_model)

        # Test state
        state = create_test_state(
            query="Hello", needs_arxiv=False, current_step="classification_complete"
        )

        # Execute
        result = direct_node(state)

        # Verify
        assert result["response"] == "Hello! How can I help you today?"
        assert result["current_step"] == "response_generated"
        mock_query_gemini.assert_called_once_with(mock_gemini_model, "Hello")


class TestRouting:
    """Test routing functionality."""

    def test_route_based_on_classification_arxiv_needed(self):
        """Test routing when arXiv is needed."""
        state = create_test_state(
            query="What is machine learning?",
            needs_arxiv=True,
            current_step="classification_complete",
        )

        result = route_based_on_classification(state)
        assert result == "rewrite_query"

    def test_route_based_on_classification_direct_answer(self):
        """Test routing for direct answer."""
        state = create_test_state(
            query="Hello", needs_arxiv=False, current_step="classification_complete"
        )

        result = route_based_on_classification(state)
        assert result == "direct_answer"
