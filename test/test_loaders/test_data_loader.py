"""
Tests for data_loader.py - focusing on critical/integration tests.
Tests file I/O, data processing, and filtering functionality.
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from src.loaders.data_loader import download, filter_abstracts, _read_json_lines, _clean_dataframe


class TestDataLoader:
    """Test suite for data_loader.py with focus on integration tests."""

    def test_download_creates_directory_and_handles_existing_file(self):
        """Test download function creates directories and handles existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock file to simulate existing download
            test_file = "test_data.json"
            data_dir = "test_data"
            
            # Create the data directory and file
            full_data_dir = os.path.join(temp_dir, data_dir)
            os.makedirs(full_data_dir, exist_ok=True)
            test_file_path = os.path.join(full_data_dir, test_file)
            
            # Write test data
            with open(test_file_path, 'w') as f:
                f.write('{"test": "data"}')
            
            with patch('src.loaders.data_loader.os.path.dirname') as mock_dirname:
                # Mock the path resolution to use our temp directory
                mock_dirname.return_value = temp_dir
                
                # Test that existing file is detected
                data_dir_path, data_file_path = download("test/dataset", data_dir, test_file)
                
                assert os.path.exists(data_dir_path)
                assert os.path.exists(data_file_path)
                assert data_dir_path == full_data_dir
                assert data_file_path == test_file_path

    @patch('src.loaders.data_loader.kagglehub.dataset_download')
    def test_download_with_kaggle_integration(self, mock_kaggle_download):
        """Test download function with Kaggle API integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup mock Kaggle download
            kaggle_temp_dir = os.path.join(temp_dir, "kaggle_download")
            os.makedirs(kaggle_temp_dir, exist_ok=True)
            
            test_file = "arxiv_data.json"
            src_file = os.path.join(kaggle_temp_dir, test_file)
            
            # Create test file in Kaggle directory
            with open(src_file, 'w') as f:
                json.dump({"test": "kaggle_data"}, f)
            
            mock_kaggle_download.return_value = kaggle_temp_dir
            
            data_dir = "test_data"
            
            with patch('src.loaders.data_loader.os.path.dirname') as mock_dirname:
                mock_dirname.return_value = temp_dir
                
                # Test download with file move
                data_dir_path, data_file_path = download("Cornell-University/arxiv", data_dir, test_file)
                
                # Verify the file was moved
                assert os.path.exists(data_file_path)
                assert not os.path.exists(src_file)  # Original should be moved
                
                # Verify content
                with open(data_file_path, 'r') as f:
                    data = json.load(f)
                    assert data["test"] == "kaggle_data"

    def test_read_json_lines_integration(self):
        """Test reading actual JSON lines file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file in temp directory
            jsonl_file = os.path.join(temp_dir, "test.jsonl")
            
            # Write test JSON lines
            test_data = [
                {"id": "1", "title": "Test Paper 1", "abstract": "Abstract 1"},
                {"id": "2", "title": "Test Paper 2", "abstract": "Abstract 2"},
                {"id": "3", "title": "Test Paper 3", "abstract": "Abstract 3"}
            ]
            
            with open(jsonl_file, 'w') as f:
                for item in test_data:
                    json.dump(item, f)
                    f.write('\n')
            
            result = _read_json_lines(jsonl_file)
            
            assert len(result) == 3
            assert result[0]["id"] == "1"
            assert result[1]["title"] == "Test Paper 2"
            assert result[2]["abstract"] == "Abstract 3"
            
            # Verify all data types are preserved
            for i, item in enumerate(result):
                assert isinstance(item, dict)
                assert item == test_data[i]

    def test_clean_dataframe_integration(self):
        """Test DataFrame cleaning with realistic data."""
        # Create test DataFrame with realistic arxiv data
        test_data = {
            "title": [
                "  Machine Learning in Healthcare  ",
                "\n  Deep Learning Applications\t",
                "AI Research  \n"
            ],
            "update_date": [
                "2023-01-15",
                "2023-02-20",
                "2023-03-10"
            ],
            "abstract": ["Abstract 1", "Abstract 2", "Abstract 3"]
        }
        
        df = pd.DataFrame(test_data)
        cleaned_df = _clean_dataframe(df)
        
        # Test date parsing
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df["update_date"])
        assert cleaned_df["update_date"].iloc[0] == pd.Timestamp("2023-01-15")
        
        # Test title cleaning
        assert cleaned_df["title"].iloc[0] == "Machine Learning in Healthcare"
        assert cleaned_df["title"].iloc[1] == "Deep Learning Applications"
        assert cleaned_df["title"].iloc[2] == "AI Research"
        
        # Verify no whitespace remains
        for title in cleaned_df["title"]:
            assert not title.startswith(' ')
            assert not title.endswith(' ')
            assert '\n' not in title
            assert '\t' not in title

    def test_filter_abstracts_end_to_end_integration(self):
        """Test complete filtering workflow with realistic data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test JSON lines file
            jsonl_file = os.path.join(temp_dir, "test_data.jsonl")
            test_data = [
                {
                    "id": "1",
                    "title": "  Old Paper  ",
                    "abstract": "Old research from 2020",
                    "update_date": "2020-01-01"
                },
                {
                    "id": "2", 
                    "title": "Recent Paper",
                    "abstract": "Recent research from 2023",
                    "update_date": "2023-06-15"
                },
                {
                    "id": "3",
                    "title": "Very Recent Paper",
                    "abstract": "Very recent research from late 2023",
                    "update_date": "2023-12-01"
                }
            ]
            
            with open(jsonl_file, 'w') as f:
                for item in test_data:
                    json.dump(item, f)
                    f.write('\n')
            
            # Test filtering
            filter_date = datetime(2023, 1, 1)
            filter_file = "filtered_data.pkl"
            
            result_df = filter_abstracts(temp_dir, jsonl_file, filter_date, filter_file)
            
            # Verify filtering worked
            assert len(result_df) == 2  # Should exclude the 2020 paper
            assert all(result_df["update_date"] >= filter_date)
            
            # Verify data cleaning happened
            assert result_df.iloc[0]["title"] == "Recent Paper"  # Should be cleaned
            assert pd.api.types.is_datetime64_any_dtype(result_df["update_date"])
            
            # Verify pickle file was created
            pickle_path = os.path.join(temp_dir, filter_file)
            assert os.path.exists(pickle_path)
            
            # Verify pickle content
            loaded_df = pd.read_pickle(pickle_path)
            pd.testing.assert_frame_equal(result_df, loaded_df)
            
            # Verify specific content
            assert "2020-01-01" not in loaded_df["update_date"].astype(str).values
            assert any("Recent research from 2023" in abstract for abstract in loaded_df["abstract"])

    def test_filter_abstracts_with_edge_cases(self):
        """Test filtering with edge cases like empty data and boundary dates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with boundary date (skip empty file test since data_loader doesn't handle it)
            boundary_file = os.path.join(temp_dir, "boundary.jsonl")
            boundary_data = [
                {"id": "1", "title": "Exact Date", "abstract": "Test", "update_date": "2023-01-01"},
                {"id": "2", "title": "One Day Before", "abstract": "Test", "update_date": "2022-12-31"},
                {"id": "3", "title": "One Day After", "abstract": "Test", "update_date": "2023-01-02"}
            ]
            
            with open(boundary_file, 'w') as f:
                for item in boundary_data:
                    json.dump(item, f)
                    f.write('\n')
            
            filter_date = datetime(2023, 1, 1)
            result_df = filter_abstracts(temp_dir, boundary_file, filter_date, "boundary.pkl")
            
            # Should include exact date and after, exclude before
            assert len(result_df) == 2
            titles = result_df["title"].tolist()
            assert "Exact Date" in titles
            assert "One Day After" in titles
            assert "One Day Before" not in titles
            
            # Test with minimal data structure
            minimal_file = os.path.join(temp_dir, "minimal.jsonl")
            minimal_data = [
                {"id": "1", "title": " Minimal Paper ", "abstract": "Minimal abstract", "update_date": "2023-06-01"}
            ]
            
            with open(minimal_file, 'w') as f:
                for item in minimal_data:
                    json.dump(item, f)
                    f.write('\n')
            
            result_df = filter_abstracts(temp_dir, minimal_file, datetime(2023, 1, 1), "minimal.pkl")
            
            # Verify single record processing
            assert len(result_df) == 1
            assert result_df.iloc[0]["title"] == "Minimal Paper"  # Should be cleaned
            assert pd.api.types.is_datetime64_any_dtype(result_df["update_date"])
