"""
Model loading utilities for embeddings and Gemini models.
"""

import os
import pandas as pd
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from ..connections.gemini_query import setup_gemini


def load_gemini_model(api_key: str, model_name: str = "gemini-2.0-flash"):
    """
    Load the Gemini model for abstract simplification.
    
    Args:
        api_key (str): Gemini API key
        model_name (str): Name of the Gemini model to use
        
    Returns:
        Gemini model instance
    """
    return setup_gemini(api_key, model_name)


def load_embedder(config: dict, device: str):
    """
    Load the embedding model with caching.

    Args:
        config (dict): Configuration dictionary
        device (str): Device to load the embedder on

    Returns:
        CacheBackedEmbeddings: Cached embedding model
    """
    fast_embed_name = config["fast_embed_name"]
    cache_dir = config["embedding_cache_dir"]

    # Initialize base embeddings
    base_embedder = HuggingFaceEmbeddings(
        model_name=fast_embed_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create a cached embedder
    store = LocalFileStore(f"{cache_dir}/embeddings_cache")
    embedder = CacheBackedEmbeddings.from_bytes_store(
        base_embedder, store, namespace=fast_embed_name
    )

    return embedder


def load_llm(config: dict, gemini_api_key: str):
    """
    Load the Gemini LLM for main RAG operations.

    Args:
        config (dict): Configuration dictionary
        gemini_api_key (str): Gemini API key

    Returns:
        Gemini model instance
    """
    # Use Gemini model for all LLM operations
    model_name = config.get("gemini_model_name", "gemini-2.0-flash")
    return setup_gemini(gemini_api_key, model_name)


def load_data_and_index(config: dict, data_dir: str):
    """
    Load the data chunks and Pinecone index.

    Args:
        config (dict): Configuration dictionary
        data_dir (str): Directory containing data files

    Returns:
        tuple: (splits DataFrame, Pinecone index)
    """
    # Load the data chunks(splits)
    chunk_file_name = config["chunk_file_name"]
    splits = pd.read_pickle(os.path.join(data_dir, chunk_file_name))

    # Load the Pinecone index
    pc_index = config["pc_index"]
    pinecone = Pinecone()
    index = pinecone.Index(pc_index)

    return splits, index


def load_all_components(config: dict, data_dir: str, device: str, gemini_api_key: str):
    """
    Load all necessary components for the RAG system.

    Args:
        config (dict): Configuration dictionary
        data_dir (str): Directory containing data files
        device (str): Device to load models on
        gemini_api_key (str): Gemini API key for all LLM operations

    Returns:
        tuple: (splits, index, llm, embedder, gemini_model)
    """
    print("Loading components...")

    # Load data and index
    splits, index = load_data_and_index(config, data_dir)
    print("Index loaded")

    # Create main LLM (Gemini)
    llm = load_llm(config, gemini_api_key)
    print("Main LLM (Gemini) loaded")

    # Load the Gemini model for simplification (same as main LLM)
    gemini_model = load_gemini_model(gemini_api_key)
    print("Simplification model (Gemini) loaded")

    # Load embedder
    embedder = load_embedder(config, device)
    print("Cached embedder loaded")

    return splits, index, llm, embedder, gemini_model
