"""
Tests for model loading functionality.
"""

from unittest.mock import Mock, patch


class TestModelLoader:
    """Test model loading functionality."""

    @patch("src.models.model_loader.Pinecone")
    @patch("src.models.model_loader.setup_gemini")
    def test_load_gemini_model(self, mock_setup_gemini, mock_pinecone):
        """Test loading Gemini model."""
        from src.models.model_loader import load_gemini_model

        mock_model = Mock()
        mock_setup_gemini.return_value = mock_model

        api_key = "test_api_key"
        model_name = "gemini-2.0-flash"

        result = load_gemini_model(api_key, model_name)

        mock_setup_gemini.assert_called_once_with(api_key, model_name)
        assert result == mock_model

    @patch("src.models.model_loader.Pinecone")
    @patch("src.models.model_loader.HuggingFaceEmbeddings")
    @patch("src.models.model_loader.LocalFileStore")
    @patch("src.models.model_loader.CacheBackedEmbeddings.from_bytes_store")
    def test_load_embedder(
        self, mock_cached_embeddings, mock_file_store, mock_hf_embeddings, mock_pinecone
    ):
        """Test loading embedder with caching."""
        from src.models.model_loader import load_embedder

        config = {
            "fast_embed_name": "BAAI/bge-small-en-v1.5",
            "embedding_cache_dir": "test_cache",
        }
        device = "cpu"

        mock_base_embedder = Mock()
        mock_hf_embeddings.return_value = mock_base_embedder

        mock_store = Mock()
        mock_file_store.return_value = mock_store

        mock_cached_embedder = Mock()
        mock_cached_embeddings.return_value = mock_cached_embedder

        result = load_embedder(config, device)

        # Verify HuggingFace embeddings were created with correct parameters
        mock_hf_embeddings.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Verify file store was created
        mock_file_store.assert_called_once_with("test_cache/embeddings_cache")

        # Verify cached embeddings were created
        mock_cached_embeddings.assert_called_once_with(
            mock_base_embedder, mock_store, namespace="BAAI/bge-small-en-v1.5"
        )

        assert result == mock_cached_embedder

    @patch("src.models.model_loader.pd.read_pickle")
    @patch("src.models.model_loader.Pinecone")
    def test_load_data_and_index(self, mock_pinecone_class, mock_read_pickle):
        """Test loading data and index."""
        from src.models.model_loader import load_data_and_index

        mock_data = Mock()
        mock_read_pickle.return_value = mock_data

        mock_pinecone_instance = Mock()
        mock_index = Mock()
        mock_pinecone_instance.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_pinecone_instance

        config = {"chunk_file_name": "test_chunks.pkl", "pc_index": "test-index"}
        data_dir = "test_data"

        splits, index = load_data_and_index(config, data_dir)

        # Use os.path.join for cross-platform compatibility
        import os

        expected_path = os.path.join("test_data", "test_chunks.pkl")
        mock_read_pickle.assert_called_once_with(expected_path)
        mock_pinecone_class.assert_called_once()
        mock_pinecone_instance.Index.assert_called_once_with("test-index")
        assert splits == mock_data
        assert index == mock_index

    @patch("src.models.model_loader.Pinecone")
    @patch("src.models.model_loader.setup_gemini")
    def test_load_llm(self, mock_setup_gemini, mock_pinecone):
        """Test loading LLM."""
        from src.models.model_loader import load_llm

        mock_model = Mock()
        mock_setup_gemini.return_value = mock_model

        config = {"gemini_model_name": "gemini-2.0-flash"}
        api_key = "test_api_key"

        result = load_llm(config, api_key)

        mock_setup_gemini.assert_called_once_with(api_key, "gemini-2.0-flash")
        assert result == mock_model

    @patch("src.models.model_loader.Pinecone")
    @patch("src.models.model_loader.load_gemini_model")
    @patch("src.models.model_loader.load_llm")
    @patch("src.models.model_loader.load_embedder")
    @patch("src.models.model_loader.load_data_and_index")
    def test_load_all_components(
        self,
        mock_load_data_index,
        mock_load_embedder,
        mock_load_llm,
        mock_load_gemini,
        mock_pinecone,
    ):
        """Test loading all components."""
        from src.models.model_loader import load_all_components

        # Setup mocks
        mock_splits = Mock()
        mock_index = Mock()
        mock_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()

        mock_load_data_index.return_value = (mock_splits, mock_index)
        mock_load_llm.return_value = mock_llm
        mock_load_embedder.return_value = mock_embedder
        mock_load_gemini.return_value = mock_gemini_model

        config = {"gemini_model_name": "gemini-2.0-flash"}
        data_dir = "data"
        device = "cpu"
        gemini_api_key = "test_key"

        result = load_all_components(config, data_dir, device, gemini_api_key)

        # Verify all components were loaded
        assert result == (
            mock_splits,
            mock_index,
            mock_llm,
            mock_embedder,
            mock_gemini_model,
        )

        # Verify individual function calls
        mock_load_data_index.assert_called_once_with(config, data_dir)
        mock_load_llm.assert_called_once_with(config, gemini_api_key)
        mock_load_embedder.assert_called_once_with(config, device)
        mock_load_gemini.assert_called_once_with(gemini_api_key)
