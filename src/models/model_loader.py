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
from src.connections.logger import get_logger

logger = get_logger(__name__)


def load_gemini_model(api_key: str, model_name: str = "gemini-2.0-flash"):
    """
    Load the Gemini model for abstract simplification.

    Args:
        api_key (str): Gemini API key
        model_name (str): Name of the Gemini model to use

    Returns:
        Gemini model instance
    """
    logger.info(f"Loading Gemini model: {model_name}")
    model = setup_gemini(api_key, model_name)
    logger.info(f"✅ Gemini model ({model_name}) loaded successfully")
    return model


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

    logger.info(f"Loading embedding model: {fast_embed_name} on device: {device}")

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

    logger.info(f"✅ Cached embedder ({fast_embed_name}) loaded successfully")
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
    logger.info(f"Loading main LLM: {model_name}")
    llm = setup_gemini(gemini_api_key, model_name)
    logger.info(f"✅ Main LLM ({model_name}) loaded successfully")
    return llm


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
    logger.info(f"Loading data chunks from: {chunk_file_name}")
    splits = pd.read_pickle(os.path.join(data_dir, chunk_file_name))
    logger.info(f"✅ Data chunks loaded: {len(splits)} chunks")

    # Load the Pinecone index
    pc_index = config["pc_index"]
    logger.info(f"Connecting to Pinecone index: {pc_index}")
    pinecone = Pinecone()
    index = pinecone.Index(pc_index)
    logger.info(f"✅ Pinecone index ({pc_index}) connected successfully")

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
    logger.info("🚀 Starting component loading process...")

    # Load data and index
    splits, index = load_data_and_index(config, data_dir)

    # Create main LLM (Gemini)
    llm = load_llm(config, gemini_api_key)

    # Load the Gemini model for simplification (same as main LLM)
    gemini_model = load_gemini_model(gemini_api_key)

    # Load embedder
    embedder = load_embedder(config, device)

    logger.info("🎉 All components loaded successfully!")
    return splits, index, llm, embedder, gemini_model
