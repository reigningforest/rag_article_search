"""
Tests for Gemini query functionality.
"""

import pytest
from unittest.mock import Mock, patch

from src.connections.gemini_query import _configure_gemini, setup_gemini, query_gemini


class TestGeminiConfiguration:
    """Test Gemini API configuration."""
    
    @patch('src.connections.gemini_query.genai.configure')
    def test_configure_gemini_valid_key(self, mock_configure):
        """Test configuring Gemini with valid API key."""
        api_key = "valid_test_key"
        _configure_gemini(api_key)
        mock_configure.assert_called_once_with(api_key=api_key)
    
    @patch('src.connections.gemini_query.genai.configure')
    def test_configure_gemini_empty_key(self, mock_configure):
        """Test configuring Gemini with empty API key."""
        api_key = ""
        _configure_gemini(api_key)
        mock_configure.assert_called_once_with(api_key=api_key)
    
    @patch('src.connections.gemini_query.genai.configure')
    def test_configure_gemini_none_key(self, mock_configure):
        """Test configuring Gemini with None API key."""
        api_key = None
        # Note: This test checks the behavior, even though it's not ideal usage
        _configure_gemini(api_key)  # type: ignore
        mock_configure.assert_called_once_with(api_key=api_key)


class TestGeminiSetup:
    """Test Gemini model setup."""
    
    @patch('src.connections.gemini_query._configure_gemini')
    @patch('src.connections.gemini_query.genai.GenerativeModel')
    def test_setup_gemini_default_model(self, mock_model, mock_configure):
        """Test setting up Gemini with default model."""
        api_key = "test_key"
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        result = setup_gemini(api_key)
        
        mock_configure.assert_called_once_with(api_key)
        mock_model.assert_called_once_with("gemini-2.0-flash")
        assert result == mock_model_instance
    
    @patch('src.connections.gemini_query._configure_gemini')
    @patch('src.connections.gemini_query.genai.GenerativeModel')
    def test_setup_gemini_custom_model(self, mock_model, mock_configure):
        """Test setting up Gemini with custom model."""
        api_key = "test_key"
        model_name = "gemini-1.5-pro"
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        result = setup_gemini(api_key, model_name)
        
        mock_configure.assert_called_once_with(api_key)
        mock_model.assert_called_once_with(model_name)
        assert result == mock_model_instance
    
    @patch('src.connections.gemini_query._configure_gemini')
    @patch('src.connections.gemini_query.genai.GenerativeModel')
    def test_setup_gemini_model_creation_error(self, mock_model, mock_configure):
        """Test setup_gemini when model creation fails."""
        api_key = "test_key"
        mock_model.side_effect = Exception("Model creation failed")
        
        with pytest.raises(Exception, match="Model creation failed"):
            setup_gemini(api_key)


class TestGeminiQuery:
    """Test Gemini query functionality."""
    
    def test_query_gemini_successful_response(self):
        """Test successful Gemini query."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "This is a test response"
        mock_model.generate_content.return_value = mock_response
        
        prompt = "Test prompt"
        result = query_gemini(mock_model, prompt)
        
        mock_model.generate_content.assert_called_once_with(prompt)
        assert result == "This is a test response"
    
    def test_query_gemini_empty_prompt(self):
        """Test Gemini query with empty prompt."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = ""
        mock_model.generate_content.return_value = mock_response
        
        prompt = ""
        result = query_gemini(mock_model, prompt)
        
        mock_model.generate_content.assert_called_once_with(prompt)
        assert result == ""
    
    def test_query_gemini_long_prompt(self):
        """Test Gemini query with long prompt."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Response to long prompt"
        mock_model.generate_content.return_value = mock_response
        
        prompt = "This is a very long prompt " * 100
        result = query_gemini(mock_model, prompt)
        
        mock_model.generate_content.assert_called_once_with(prompt)
        assert result == "Response to long prompt"
    
    def test_query_gemini_api_error(self):
        """Test Gemini query when API returns error."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        
        prompt = "Test prompt"
        result = query_gemini(mock_model, prompt)
        
        mock_model.generate_content.assert_called_once_with(prompt)
        assert "Error querying Gemini" in result
    
    def test_query_gemini_no_text_attribute(self):
        """Test Gemini query when response has no text attribute."""
        mock_model = Mock()
        mock_response = Mock(spec=[])  # Mock without text attribute
        mock_model.generate_content.return_value = mock_response
        
        prompt = "Test prompt"
        result = query_gemini(mock_model, prompt)
        
        assert "Error querying Gemini" in result
    
    def test_query_gemini_none_response(self):
        """Test Gemini query when response is None."""
        mock_model = Mock()
        mock_model.generate_content.return_value = None
        
        prompt = "Test prompt"
        result = query_gemini(mock_model, prompt)
        
        assert "Error querying Gemini" in result
    
    def test_query_gemini_special_characters(self):
        """Test Gemini query with special characters in prompt."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Handled special characters"
        mock_model.generate_content.return_value = mock_response
        
        prompt = "Test with special chars: @#$%^&*()[]{}|\\:;\"'<>,.?/`~"
        result = query_gemini(mock_model, prompt)
        
        mock_model.generate_content.assert_called_once_with(prompt)
        assert result == "Handled special characters"
    
    def test_query_gemini_unicode_prompt(self):
        """Test Gemini query with unicode characters."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Handled unicode: αβγδε"
        mock_model.generate_content.return_value = mock_response
        
        prompt = "Test with unicode: αβγδε 中文 العربية"
        result = query_gemini(mock_model, prompt)
        
        mock_model.generate_content.assert_called_once_with(prompt)
        assert result == "Handled unicode: αβγδε"


class TestGeminiIntegration:
    """Integration tests for Gemini functionality."""
    
    @patch('src.connections.gemini_query.genai.configure')
    @patch('src.connections.gemini_query.genai.GenerativeModel')
    def test_full_gemini_workflow(self, mock_model_class, mock_configure):
        """Test complete Gemini setup and query workflow."""
        api_key = "test_key"
        model_name = "gemini-2.0-flash"
        prompt = "What is AI?"
        expected_response = "AI is artificial intelligence"
        
        # Setup mocks
        mock_model_instance = Mock()
        mock_response = Mock()
        mock_response.text = expected_response
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance
        
        # Test workflow
        model = setup_gemini(api_key, model_name)
        result = query_gemini(model, prompt)
        
        # Verify calls
        mock_configure.assert_called_once_with(api_key=api_key)
        mock_model_class.assert_called_once_with(model_name)
        mock_model_instance.generate_content.assert_called_once_with(prompt)
        assert result == expected_response
    
    @patch('src.connections.gemini_query.genai.configure')
    @patch('src.connections.gemini_query.genai.GenerativeModel')
    def test_gemini_workflow_with_error_recovery(self, mock_model_class, mock_configure):
        """Test Gemini workflow with error and recovery."""
        api_key = "test_key"
        prompt = "Test prompt"
        
        # Setup mocks - first call fails, second succeeds
        mock_model_instance = Mock()
        mock_model_instance.generate_content.side_effect = [
            Exception("First call fails"),
            Mock(text="Second call succeeds")
        ]
        mock_model_class.return_value = mock_model_instance
        
        # Test workflow
        model = setup_gemini(api_key)
        
        # First call should return error message
        result1 = query_gemini(model, prompt)
        assert "Error querying Gemini" in result1
        
        # Second call should succeed
        result2 = query_gemini(model, prompt)
        assert result2 == "Second call succeeds"
