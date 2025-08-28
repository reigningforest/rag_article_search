"""
Model loading utilities for embeddings and Gemini models.
"""

import os
import pandas as pd
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from ..connections.gemini_query import setup_client
from src.connections.logger import get_shared_logger

logger = get_shared_logger(__name__)


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

    logger.info(f"Cached embedder ({fast_embed_name}) loaded successfully")
    return embedder


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
    logger.info(f"Data chunks loaded: {len(splits)} chunks")

    # Load the Pinecone index
    pc_index = config["pc_index"]
    logger.info(f"Connecting to Pinecone index: {pc_index}")
    pinecone = Pinecone()
    index = pinecone.Index(pc_index)
    logger.info(f"Pinecone index ({pc_index}) connected successfully")

    return splits, index


def load_all_components(config: dict, data_dir: str, device: str):
    """
    Load all necessary components for the RAG system.

    Args:
        config (dict): Configuration dictionary
        data_dir (str): Directory containing data files
        device (str): Device to load models on

    Returns:
        tuple: (splits, index, client, embedder)
    """
    logger.info("🚀 Starting component loading process...")

    # Load data and index
    splits, index = load_data_and_index(config, data_dir)

    # Create main LLM (Gemini)
    client = setup_client()

    # Load embedder
    embedder = load_embedder(config, device)

    logger.info("🎉 All components loaded successfully!")
    return splits, index, client, embedder
