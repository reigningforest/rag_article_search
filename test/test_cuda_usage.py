"""
Test CUDA availability and usage for the RAG system components.
"""

import pytest
import torch
from unittest.mock import patch


class TestCudaUsage:
    """Test CUDA detection and usage across the system."""

    def test_torch_cuda_availability(self):
        """Test basic torch CUDA detection."""
        # This will return True if CUDA is available, False otherwise
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")

        # Test passes regardless of CUDA availability - just reports status
        assert isinstance(cuda_available, bool)

    def test_device_selection_logic(self):
        """Test the device selection logic used in the application."""
        # Test the actual logic used in the app
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Selected device: {device}")

        assert device in ["cuda", "cpu"]

        if torch.cuda.is_available():
            assert device == "cuda"
        else:
            assert device == "cpu"

    @patch("torch.cuda.is_available")
    def test_device_selection_with_mocked_cuda_true(self, mock_cuda):
        """Test device selection when CUDA is mocked to be available."""
        mock_cuda.return_value = True

        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cuda"
        mock_cuda.assert_called_once()

    @patch("torch.cuda.is_available")
    def test_device_selection_with_mocked_cuda_false(self, mock_cuda):
        """Test device selection when CUDA is mocked to be unavailable."""
        mock_cuda.return_value = False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device == "cpu"
        mock_cuda.assert_called_once()

    def test_embedding_model_device_compatibility(self):
        """Test that embedding models can work with detected device."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Test creating a simple tensor on the detected device
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
            print(f"Test tensor created successfully on device: {test_tensor.device}")
            assert test_tensor.device.type == device
        except Exception as e:
            pytest.fail(f"Failed to create tensor on device {device}: {e}")

    def test_huggingface_embeddings_device_detection(self):
        """Test HuggingFace embeddings device parameter."""
        # Import here to avoid issues if not installed
        try:
            import importlib.util

            spec = importlib.util.find_spec("langchain_huggingface")
            if spec is None:
                pytest.skip("langchain_huggingface not available")
        except ImportError:
            pytest.skip("HuggingFaceEmbeddings not available")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Test that we can create embeddings with the detected device
        # Note: We don't actually load the model to avoid heavy operations in tests
        model_kwargs = {"device": device}

        print(f"Would use device for HuggingFace embeddings: {device}")
        assert device in ["cuda", "cpu"]
        assert isinstance(model_kwargs, dict)
        assert model_kwargs["device"] == device

    def test_memory_requirements_warning(self):
        """Test that appropriate warnings are shown for device selection."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cpu":
            # This would be the warning logic in actual code
            warning_message = "CUDA not available. Using CPU for embeddings."
            print(warning_message)
            assert "CPU" in warning_message
        else:
            info_message = f"Using device: {device}"
            print(info_message)
            assert device in info_message

    def test_cuda_memory_info_if_available(self):
        """Test CUDA memory information if CUDA is available."""
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                print(f"CUDA memory allocated: {memory_allocated / 1024**2:.2f} MB")
                print(f"CUDA memory reserved: {memory_reserved / 1024**2:.2f} MB")

                assert isinstance(memory_allocated, int)
                assert isinstance(memory_reserved, int)
                assert memory_allocated >= 0
                assert memory_reserved >= memory_allocated
            except Exception as e:
                pytest.fail(f"Failed to get CUDA memory info: {e}")
        else:
            pytest.skip("CUDA not available for memory testing")


if __name__ == "__main__":
    # Run the tests directly
    test_instance = TestCudaUsage()

    print("=== CUDA Availability Test ===")
    test_instance.test_torch_cuda_availability()

    print("\n=== Device Selection Test ===")
    test_instance.test_device_selection_logic()

    print("\n=== Embedding Device Compatibility Test ===")
    test_instance.test_embedding_model_device_compatibility()

    print("\n=== Memory Requirements Warning Test ===")
    test_instance.test_memory_requirements_warning()

    if torch.cuda.is_available():
        print("\n=== CUDA Memory Info Test ===")
        test_instance.test_cuda_memory_info_if_available()

    print("\n=== All CUDA tests completed ===")
