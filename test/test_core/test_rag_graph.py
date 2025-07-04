"""
Tests for RAG graph builder functionality.
"""

from typing import cast
from unittest.mock import Mock, patch
from src.core.state import RAGState
from src.core.rag_graph import build_rag_graph


def create_test_state(**kwargs) -> RAGState:
    """Helper to create test RAGState."""
    default_state = {
        "query": "",
        "needs_arxiv": False,
        "rewrites": [],
        "documents": [],
        "response": "",
        "current_step": "initialized"
    }
    default_state.update(kwargs)
    return cast(RAGState, default_state)


class TestRAGGraphBuilder:
    """Test RAG graph builder functionality."""
    
    def test_build_rag_graph_creation(self):
        """Test building the RAG graph."""
        # Mock all required components
        mock_splits = Mock()
        mock_index = Mock()
        mock_gemini_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()
        
        # Mock config with required prompts
        mock_config = {
            "top_k": 5,
            "classification_prompt": "Does this query need arXiv papers? Query: {query}",
            "rewrite_prompt": "Rewrite this query: {query}",
            "final_prompt": "Answer: {query}\nContext: {context}\nRewrites: {rewrites}",
            "hugging_face_template": "Simplify this abstract: {abstract}"
        }
        
        # Build the graph
        graph = build_rag_graph(
            mock_splits, mock_index, mock_gemini_llm, mock_embedder, mock_config, mock_gemini_model
        )
        
        # Verify graph was created
        assert graph is not None
        assert hasattr(graph, 'invoke')
    
    @patch('src.core.rag_graph.StateGraph')
    def test_graph_node_addition(self, mock_state_graph_class):
        """Test that all nodes are added to the graph."""
        # Mock StateGraph
        mock_graph = Mock()
        mock_state_graph_class.return_value = mock_graph
        
        # Mock components
        mock_splits = Mock()
        mock_index = Mock()
        mock_gemini_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()
        mock_config = {
            "top_k": 5,
            "classification_prompt": "Test prompt {query}",
            "rewrite_prompt": "Rewrite: {query}",
            "final_prompt": "Answer: {query}",
            "hugging_face_template": "Simplify: {abstract}"
        }
        
        # Build the graph
        build_rag_graph(
            mock_splits, mock_index, mock_gemini_llm, mock_embedder, mock_config, mock_gemini_model
        )
        
        # Verify all nodes were added
        expected_nodes = ["classify", "rewrite_query", "retrieve", "simplify_abstracts", "generate_rag_response", "direct_answer"]
        actual_calls = [call[0][0] for call in mock_graph.add_node.call_args_list]
        
        for node in expected_nodes:
            assert node in actual_calls
    
    @patch('src.core.rag_graph.StateGraph')
    def test_graph_edge_addition(self, mock_state_graph_class):
        """Test that all edges are added to the graph."""
        # Mock StateGraph
        mock_graph = Mock()
        mock_state_graph_class.return_value = mock_graph
        
        # Mock components
        mock_splits = Mock()
        mock_index = Mock()
        mock_gemini_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()
        mock_config = {
            "top_k": 5,
            "classification_prompt": "Test prompt {query}",
            "rewrite_prompt": "Rewrite: {query}",
            "final_prompt": "Answer: {query}",
            "hugging_face_template": "Simplify: {abstract}"
        }
        
        # Build the graph
        build_rag_graph(
            mock_splits, mock_index, mock_gemini_llm, mock_embedder, mock_config, mock_gemini_model
        )
        
        # Verify edges were added
        assert mock_graph.add_edge.call_count > 0
        assert mock_graph.add_conditional_edges.call_count > 0
    
    def test_graph_execution_with_research_query(self):
        """Test graph execution with a research query."""
        # Mock components
        mock_splits = Mock()
        mock_index = Mock()
        mock_gemini_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()
        
        # Mock embedder
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Mock index query
        mock_match = Mock()
        mock_match.id = '0'
        mock_results = Mock()
        mock_results.matches = [mock_match]
        mock_index.query.return_value = mock_results
        
        # Mock splits data
        import pandas as pd
        mock_splits.iloc = pd.DataFrame({
            'title': ['Test Paper'],
            'update_date': [pd.Timestamp('2023-01-01')],
            'chunk_text': ['Machine learning content']
        }).iloc
        
        # Mock config
        mock_config = {
            "top_k": 5,
            "classification_prompt": "Does this query need arXiv papers? Query: {query}",
            "rewrite_prompt": "Rewrite this query: {query}",
            "final_prompt": "Answer: {query}\nContext: {context}\nRewrites: {rewrites}",
            "hugging_face_template": "Simplify this abstract: {abstract}"
        }
        
        # Build the graph
        with patch('src.core.nodes.query_gemini') as mock_query_gemini:
            # Mock Gemini responses for the workflow
            mock_query_gemini.side_effect = [
                "yes",  # classify response
                "What is ML?\nDefine machine learning",  # rewrite response
                "ML is AI that learns from data",  # simplify response
                "Machine learning is a subset of AI..."  # generate response
            ]
            
            graph = build_rag_graph(
                mock_splits, mock_index, mock_gemini_llm, mock_embedder, mock_config, mock_gemini_model
            )
            
            # Test state
            initial_state = create_test_state(
                query="What is machine learning?"
            )
            
            # Execute graph
            result = graph.invoke(initial_state)
            
            # Verify final state has response
            assert "response" in result
            assert result["query"] == "What is machine learning?"
    
    def test_graph_execution_with_simple_query(self):
        """Test graph execution with a simple query that doesn't need research."""
        # Mock components
        mock_splits = Mock()
        mock_index = Mock()
        mock_gemini_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()
        
        # Mock config
        mock_config = {
            "top_k": 5,
            "classification_prompt": "Does this query need arXiv papers? Query: {query}",
            "rewrite_prompt": "Rewrite this query: {query}",
            "final_prompt": "Answer: {query}\nContext: {context}\nRewrites: {rewrites}",
            "hugging_face_template": "Simplify this abstract: {abstract}"
        }
        
        # Build and test graph
        with patch('src.core.nodes.query_gemini') as mock_query_gemini:
            # Mock Gemini responses - classify as no research needed
            mock_query_gemini.side_effect = [
                "no",  # classify response
                "Hello! How can I help you today?"  # direct response
            ]
            
            graph = build_rag_graph(
                mock_splits, mock_index, mock_gemini_llm, mock_embedder, mock_config, mock_gemini_model
            )
            
            # Test state
            initial_state = create_test_state(
                query="Hello"
            )
            
            # Execute graph
            result = graph.invoke(initial_state)
            
            # Verify final state
            assert "response" in result
            assert result["query"] == "Hello"
            # Should not query index for simple greeting
            mock_index.query.assert_not_called()
    
    def test_graph_compilation(self):
        """Test that the graph compiles successfully."""
        # Mock components
        mock_splits = Mock()
        mock_index = Mock()
        mock_gemini_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()
        mock_config = {
            "top_k": 5,
            "classification_prompt": "Test prompt {query}",
            "rewrite_prompt": "Rewrite: {query}",
            "final_prompt": "Answer: {query}",
            "hugging_face_template": "Simplify: {abstract}"
        }
        
        # Build the graph
        graph = build_rag_graph(
            mock_splits, mock_index, mock_gemini_llm, mock_embedder, mock_config, mock_gemini_model
        )
        
        # Verify graph has necessary methods
        assert hasattr(graph, 'invoke')
        assert hasattr(graph, 'get_graph')
        
        # Test that graph can be visualized
        try:
            graph_def = graph.get_graph()
            assert graph_def is not None
        except Exception:
            # Some graph implementations might not support get_graph
            # This is acceptable for our testing purposes
            pass
    
    def test_graph_with_custom_config(self):
        """Test graph building with custom configuration."""
        # Mock components
        mock_splits = Mock()
        mock_index = Mock()
        mock_gemini_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()
        
        # Custom config
        custom_config = {
            "top_k": 10,  # Custom top_k
            "classification_prompt": "Custom classify prompt {query}",
            "rewrite_prompt": "Custom rewrite: {query}",
            "final_prompt": "Custom answer: {query}",
            "hugging_face_template": "Custom simplify: {abstract}",
            "custom_param": "test_value"
        }
        
        # Build the graph
        graph = build_rag_graph(
            mock_splits, mock_index, mock_gemini_llm, mock_embedder, custom_config, mock_gemini_model
        )
        
        # Verify graph was created with custom config
        assert graph is not None
        
        # Test execution to verify custom top_k is used
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_match = Mock()
        mock_match.id = '0'
        mock_results = Mock()
        mock_results.matches = [mock_match]
        mock_index.query.return_value = mock_results
        
        # Mock splits
        import pandas as pd
        mock_splits.iloc = pd.DataFrame({
            'title': ['Test Paper'],
            'update_date': [pd.Timestamp('2023-01-01')],
            'chunk_text': ['Test content']
        }).iloc
        
        with patch('src.core.nodes.query_gemini') as mock_query_gemini:
            mock_query_gemini.side_effect = [
                "yes",  # classify
                "test query",  # rewrite
                "simplified",  # simplify
                "final response"  # generate
            ]
            
            initial_state = create_test_state(
                query="Test question"
            )
            
            graph.invoke(initial_state)
            
            # Verify index was queried with custom top_k
            mock_index.query.assert_called()
            call_args = mock_index.query.call_args
            assert call_args[1]["top_k"] == 10  # Custom top_k value
