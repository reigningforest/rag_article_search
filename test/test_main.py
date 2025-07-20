"""
Tests for the main application.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
import yaml

CONFIG_DATA = '''env_file: ".env"
gemini_model_name: "gemini-2.0-flash"
data_dir: "data"
output_dir: "output"
chunk_file_name: "chunks.pkl"
pc_index: "test-index"
fast_embed_name: "BAAI/bge-small-en-v1.5"
top_k: 5
fine_tuned_graph_file_name: "fine_tuned_graph.png"'''


class TestMainApplication:
    """Test main application functionality."""

    @patch("builtins.open", new_callable=mock_open, read_data=CONFIG_DATA)
    @patch("main.load_dotenv")
    @patch("main.os.getenv")
    @patch("main.torch.cuda.is_available")
    @patch("main.load_all_components")
    @patch("main.build_rag_graph")
    @patch("main.visualize_graph")
    @patch("main.os.makedirs")
    @patch("main.os.path.exists")
    def test_main_successful_execution(
        self,
        mock_exists,
        mock_makedirs,
        mock_visualize,
        mock_build_graph,
        mock_load_components,
        mock_cuda,
        mock_getenv,
        mock_load_dotenv,
        mock_file,
    ):
        """Test successful main application execution."""
        from main import main

        # Setup mocks
        mock_exists.return_value = False  # Output directory doesn't exist
        mock_cuda.return_value = True  # CUDA available
        mock_getenv.return_value = "test_gemini_api_key"

        # Mock component loading
        mock_splits = Mock()
        mock_index = Mock()
        mock_gemini_llm = Mock()
        mock_embedder = Mock()
        mock_gemini_model = Mock()
        mock_load_components.return_value = (
            mock_splits,
            mock_index,
            mock_gemini_llm,
            mock_embedder,
            mock_gemini_model,
        )

        # Mock graph building
        mock_graph = Mock()
        mock_build_graph.return_value = mock_graph

        # Execute main with mocked input
        with patch("builtins.input", side_effect=["test query", "exit"]):
            with patch("builtins.print"):
                main()

        # Verify calls
        mock_load_dotenv.assert_called_once()
        mock_getenv.assert_called_with("GEMINI_API_KEY")
        mock_makedirs.assert_called_once_with("output")
        mock_load_components.assert_called_once()
        mock_build_graph.assert_called_once()
        mock_visualize.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data=CONFIG_DATA)
    @patch("main.load_dotenv")
    @patch("main.os.getenv")
    def test_main_missing_gemini_api_key(
        self, mock_getenv, mock_load_dotenv, mock_file
    ):
        """Test main application with missing Gemini API key."""
        from main import main

        # Setup mocks
        mock_getenv.return_value = None  # No API key

        # Execute main and capture print output
        with patch("builtins.print") as mock_print:
            main()

        # Verify error message was printed
        mock_print.assert_called_with(
            "GEMINI_API_KEY not found in environment variables."
        )

    @patch("builtins.open", new_callable=mock_open, read_data=CONFIG_DATA)
    @patch("main.load_dotenv")
    @patch("main.os.getenv")
    @patch("main.torch.cuda.is_available")
    @patch("main.os.path.exists")
    def test_main_cuda_not_available(
        self, mock_exists, mock_cuda, mock_getenv, mock_load_dotenv, mock_file
    ):
        """Test main application when CUDA is not available."""
        from main import main

        # Setup mocks
        mock_exists.return_value = True  # Output directory exists
        mock_cuda.return_value = False  # No CUDA
        mock_getenv.return_value = "test_key"

        with patch("main.load_all_components") as mock_load_components:
            with patch("main.build_rag_graph"):
                with patch("main.visualize_graph"):
                    with patch("builtins.input", side_effect=["exit"]):
                        with patch("builtins.print") as mock_print:
                            # Mock load_all_components to return expected values
                            mock_load_components.return_value = (
                                Mock(),
                                Mock(),
                                Mock(),
                                Mock(),
                                Mock(),
                            )
                            main()

        # Verify CPU message was printed
        mock_print.assert_any_call("CUDA not available. Using CPU for embeddings.")

    @patch("builtins.open", new_callable=mock_open, read_data="invalid: yaml: content")
    def test_main_invalid_config_file(self, mock_file):
        """Test main application with invalid config file."""
        from main import main

        # This should raise a YAML parsing error
        with pytest.raises(yaml.YAMLError):
            main()


class TestConfigurationLoading:
    """Test configuration loading functionality."""

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
gemini_model_name: "gemini-2.0-flash"
fast_embed_name: "BAAI/bge-small-en-v1.5"
top_k: 5
pc_index: "abstract-index"
distance_metric: "cosine"
data_dir: "data"
output_dir: "output"
embedding_cache_dir: "embedding_cache"
env_file: ".env"
""",
    )
    def test_config_loading(self, mock_file):
        """Test loading configuration from YAML file."""
        import yaml

        # This simulates the config loading in main()
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Verify configuration values
        assert config["gemini_model_name"] == "gemini-2.0-flash"
        assert config["fast_embed_name"] == "BAAI/bge-small-en-v1.5"
        assert config["top_k"] == 5
        assert config["pc_index"] == "abstract-index"
        assert config["data_dir"] == "data"
        assert config["output_dir"] == "output"


class TestUserInteraction:
    """Test user interaction functionality."""

    @patch("main.load_all_components")
    @patch("main.build_rag_graph")
    @patch("main.visualize_graph")
    @patch("main.os.makedirs")
    @patch("main.os.path.exists")
    @patch("main.torch.cuda.is_available")
    @patch("main.os.getenv")
    @patch("main.load_dotenv")
    @patch("builtins.open", new_callable=mock_open, read_data=CONFIG_DATA)
    def test_user_query_processing(
        self,
        mock_file,
        mock_load_dotenv,
        mock_getenv,
        mock_cuda,
        mock_exists,
        mock_makedirs,
        mock_visualize,
        mock_build_graph,
        mock_load_components,
    ):
        """Test processing user queries in the interactive loop."""
        from main import main

        # Setup mocks
        mock_exists.return_value = True
        mock_cuda.return_value = True
        mock_getenv.return_value = "test_key"

        # Mock graph and invoke
        mock_graph = Mock()
        mock_result = {"response": "Test response"}
        mock_graph.invoke.return_value = mock_result
        mock_build_graph.return_value = mock_graph

        mock_load_components.return_value = (Mock(), Mock(), Mock(), Mock(), Mock())

        # Test query processing
        user_inputs = ["What is machine learning?", "Explain neural networks", "exit"]

        with patch("builtins.input", side_effect=user_inputs):
            with patch("builtins.print") as mock_print:
                main()

        # Verify graph was invoked for each non-exit query
        assert mock_graph.invoke.call_count == 2

        # Verify responses were printed (they are printed as separate calls)
        mock_print.assert_any_call("\nResponse:")
        mock_print.assert_any_call("Test response")


class TestErrorHandling:
    """Test error handling in main application."""

    @patch("builtins.open", side_effect=FileNotFoundError("config.yaml not found"))
    def test_config_file_not_found(self, mock_file):
        """Test handling when config file is not found."""
        from main import main

        with pytest.raises(FileNotFoundError):
            main()

    @patch("builtins.open", new_callable=mock_open, read_data=CONFIG_DATA)
    @patch("main.load_dotenv")
    @patch("main.os.getenv")
    @patch("main.torch.cuda.is_available")
    @patch("main.load_all_components")
    @patch("main.os.path.exists")
    def test_component_loading_failure(
        self,
        mock_exists,
        mock_load_components,
        mock_cuda,
        mock_getenv,
        mock_load_dotenv,
        mock_file,
    ):
        """Test handling when component loading fails."""
        from main import main

        # Setup mocks
        mock_exists.return_value = True
        mock_cuda.return_value = True
        mock_getenv.return_value = "test_key"
        mock_load_components.side_effect = Exception("Component loading failed")

        # Should raise the exception
        with pytest.raises(Exception, match="Component loading failed"):
            main()
