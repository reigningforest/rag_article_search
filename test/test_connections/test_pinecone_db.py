"""Tests for Pinecone database functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


@patch('pinecone.ServerlessSpec')
@patch('pinecone.Pinecone')
class TestPineconeIndexManagement:
    """Test Pinecone index creation and management."""
    
    def test_check_and_create_index_new_index(self, mock_pinecone, mock_serverless):
        """Test creating a new Pinecone index."""
        from src.connections.pinecone_db import _check_and_create_index
        
        mock_pc = Mock()
        mock_pc.list_indexes.return_value.names.return_value = []
        
        pc_index = "test-index"
        embeddings = np.random.random((100, 384))
        distance_metric = "cosine"
        pc_cloud = "aws"
        pc_region = "us-east-1"
        
        _check_and_create_index(
            mock_pc, pc_index, embeddings, distance_metric, pc_cloud, pc_region
        )
        
        mock_pc.list_indexes.assert_called_once()
    
    def test_check_and_create_index_existing_index(self, mock_pinecone, mock_serverless):
        """Test handling existing Pinecone index."""
        from src.connections.pinecone_db import _check_and_create_index
        
        mock_pc = Mock()
        mock_pc.list_indexes.return_value.names.return_value = ["test-index"]
        
        pc_index = "test-index"
        embeddings = np.random.random((100, 384))
        distance_metric = "cosine"
        pc_cloud = "aws"
        pc_region = "us-east-1"
        
        _check_and_create_index(
            mock_pc, pc_index, embeddings, distance_metric, pc_cloud, pc_region
        )
        
        mock_pc.list_indexes.assert_called_once()
        mock_pc.delete_index.assert_called_once_with(pc_index)


@patch('pinecone.Pinecone')
class TestVectorPreparation:
    """Test vector preparation for Pinecone upload."""
    
    def test_prepare_vectors_small_batch(self, mock_pinecone):
        """Test vector preparation with small batch."""
        from src.connections.pinecone_db import _prepare_vectors
        
        embeddings = np.random.random((5, 384))
        
        vectors = _prepare_vectors(embeddings)
        
        assert len(vectors) == 5
        for i, vector in enumerate(vectors):
            assert vector.id == str(i)
            assert len(vector.values) == 384
    
    def test_prepare_vectors_empty_array(self, mock_pinecone):
        """Test vector preparation with empty input."""
        from src.connections.pinecone_db import _prepare_vectors
        
        embeddings = np.array([]).reshape(0, 384)
        
        vectors = _prepare_vectors(embeddings)
        
        assert len(vectors) == 0


@patch('src.connections.pinecone_db.ServerlessSpec')
@patch('src.connections.pinecone_db.Pinecone')
class TestPineconeUpload:
    """Test Pinecone data upload functionality."""
    
    def test_pinecone_upload_successful(self, mock_pinecone_class, mock_serverless):
        """Test successful upload to Pinecone."""
        from src.connections.pinecone_db import pinecone_upload
        
        # Setup mocks
        mock_pc = Mock()
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        mock_pc.list_indexes.return_value.names.return_value = []
        mock_pinecone_class.return_value = mock_pc
        
        embeddings = np.random.random((5, 384))
        
        pinecone_upload(
            pc_index="test-index",
            embeddings=embeddings,
            batch_size=10,
            distance_metric="cosine",
            pc_cloud="aws",
            pc_region="us-east-1"
        )
        
        # Verify that upsert was called at least once
        assert mock_index.upsert.call_count > 0, f"Expected upsert to be called, but call_count is {mock_index.upsert.call_count}"
    
    @patch('src.connections.pinecone_db._check_and_create_index')
    def test_pinecone_upload_index_creation_failure(self, mock_check_and_create, mock_pinecone_class, mock_serverless):
        """Test upload failure during index creation."""
        from src.connections.pinecone_db import pinecone_upload
        
        # Setup mock to raise exception during index creation
        mock_check_and_create.side_effect = Exception("API Error")
        mock_pc = Mock()
        mock_pinecone_class.return_value = mock_pc
        
        embeddings = np.random.random((5, 384))
        
        with pytest.raises(Exception, match="API Error"):
            pinecone_upload(
                pc_index="test-index",
                embeddings=embeddings,
                batch_size=10,
                distance_metric="cosine",
                pc_cloud="aws",
                pc_region="us-east-1"
            )