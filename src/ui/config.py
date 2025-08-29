"""
Configuration and validation utilities for the Streamlit UI.
"""

import os
import yaml
import streamlit as st
from pathlib import Path


def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config") / "config.yaml"
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file: {e}")
        st.stop()


def check_data_exists(config):
    """Check if processed data files exist."""
    data_dir = config["data_dir"]
    required_files = [
        config["embeddings_selected_file_name"],
        config["chunk_selected_file_name"],
    ]

    existing_files = []
    for file in required_files:
        file_path = Path(data_dir) / file
        if file_path.exists():
            existing_files.append(file)

    return len(existing_files) == len(required_files), existing_files


def check_dependencies():
    """Check if required dependencies are available."""
    dependencies = {
        "torch": False,
        "pinecone": False,
        "fastembed": False,
        "langchain": False,
    }

    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            pass

    # Check environment variables
    env_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY"]
    env_status = {}
    for var in env_vars:
        env_status[var] = bool(os.getenv(var))

    return dependencies, env_status


def load_and_validate_config():
    """Load config and check system status."""
    config = load_config()
    dependencies, env_status = check_dependencies()
    data_exists, existing_files = check_data_exists(config)
    
    return config, {
        "dependencies": dependencies,
        "env_status": env_status,
        "data_exists": data_exists,
        "existing_files": existing_files
    }
