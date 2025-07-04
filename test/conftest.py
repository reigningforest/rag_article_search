"""
Pytest configuration and fixtures for the Agentic RAG test suite.
"""

import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import Mock
from typing import Dict, Any

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def mock_config():
    """Mock configuration dictionary for testing."""
    return {
        "gemini_model_name": "gemini-2.0-flash",
        "fast_embed_name": "BAAI/bge-small-en-v1.5",
        "top_k": 5,
        "pc_index": "test-index",
        "distance_metric": "cosine",
        "data_dir": "test_data",
        "output_dir": "test_output",
        "embedding_cache_dir": "test_cache",
        "env_file": ".env",
        "chunk_file_name": "test_chunks.pkl",
    }


@pytest.fixture
def mock_gemini_api_key():
    """Mock Gemini API key for testing."""
    return "test_gemini_api_key_123"


@pytest.fixture
def mock_pinecone_api_key():
    """Mock Pinecone API key for testing."""
    return "test_pinecone_api_key_123"


@pytest.fixture
def mock_gemini_model():
    """Mock Gemini model for testing."""
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = "Test response from Gemini"
    mock_model.generate_content.return_value = mock_response
    return mock_model


@pytest.fixture
def mock_pinecone_index():
    """Mock Pinecone index for testing."""
    mock_index = Mock()
    mock_index.query.return_value = {
        'matches': [
            {
                'id': 'test_id_1',
                'score': 0.9,
                'metadata': {'text': 'Test abstract 1'}
            },
            {
                'id': 'test_id_2', 
                'score': 0.8,
                'metadata': {'text': 'Test abstract 2'}
            }
        ]
    }
    return mock_index


@pytest.fixture
def mock_embedder():
    """Mock embedder for testing."""
    mock_embedder = Mock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    return mock_embedder


@pytest.fixture
def sample_abstracts():
    """Sample abstract data for testing."""
    return [
        {
            'id': 'test_id_1',
            'title': 'Test Paper 1',
            'abstract': 'This is a test abstract about machine learning.',
            'categories': 'cs.LG'
        },
        {
            'id': 'test_id_2',
            'title': 'Test Paper 2', 
            'abstract': 'This is another test abstract about neural networks.',
            'categories': 'cs.NE'
        }
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "What is machine learning?",
        "Explain neural networks",
        "How does deep learning work?",
        "What are the applications of AI?"
    ]


@pytest.fixture
def sample_rag_state():
    """Sample RAG state for testing."""
    return {
        "query": "What is machine learning?",
        "needs_arxiv": True,
        "rewrites": [
            "What is machine learning?",
            "Define machine learning",
            "Explain ML concepts"
        ],
        "documents": [
            "Machine learning is a subset of artificial intelligence...",
            "ML algorithms learn patterns from data..."
        ],
        "simplified_docs": [
            "Machine learning helps computers learn from data.",
            "ML finds patterns in information."
        ],
        "response": "Machine learning is a field of AI that enables computers to learn and improve from experience without being explicitly programmed.",
        "current_step": "complete"
    }


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_prompts():
    """Mock prompts for testing."""
    return {
        "classification_prompt": "Does this query need arXiv papers? Query: {query}",
        "rewrite_prompt": "Rewrite this query in 3 different ways: {query}",
        "simplification_prompt": "Simplify this text: {text}",
        "generation_prompt": "Answer this question using the context: {query}\n\nContext: {context}"
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("PINECONE_API_KEY", "test_pinecone_key")


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    return "cpu"


class MockLangGraphState:
    """Mock LangGraph state for testing."""
    
    def __init__(self, initial_state: Dict[str, Any]):
        self._state = initial_state
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def update(self, updates: Dict[str, Any]):
        self._state.update(updates)


@pytest.fixture
def mock_langgraph_state(sample_rag_state):
    """Mock LangGraph state object."""
    return MockLangGraphState(sample_rag_state)
