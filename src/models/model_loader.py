"""
Model loading utilities for embeddings and Gemini models.
"""

import os
import pandas as pd
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from ..connections.gemini_query import setup_client
from ..connections.logger import get_shared_logger

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


def load_simplification_model(config: dict, device: str):
    """
    Load the fine-tuned simplification model using PEFT.

    Args:
        config (dict): Configuration dictionary containing simplification_model settings
        device (str): Device to load the model on

    Returns:
        tuple: (model, tokenizer) or (None, None) if loading fails
    """
    try:
        simplification_config = config.get("simplification_model", {})
        if not simplification_config.get("use_finetuned", False):
            logger.info("Fine-tuned model disabled in config, skipping load")
            return None, None

        # Check device requirement
        if device == "cpu":
            logger.error("Fine-tuned simplification model requires GPU. CPU is not supported.")
            logger.warning("Falling back to Gemini for simplification")
            return None, None

        base_model_name = simplification_config.get("base_model_name")
        model_path = simplification_config.get("model_path", "models/")
        
        if not base_model_name:
            logger.error("base_model_name not specified in config")
            return None, None
        
        logger.info(f"Loading fine-tuned simplification model from: {model_path}")
        logger.info(f"Base model: {base_model_name}")
        
        # Load base model and tokenizer
        logger.info(f"Loading base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load PEFT adapter
        logger.info(f"Loading PEFT adapter from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        logger.info(f"Fine-tuned simplification model loaded successfully on {device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load fine-tuned simplification model: {e}")
        logger.warning("Falling back to Gemini for simplification")
        return None, None


def load_all_components(config: dict, data_dir: str, device: str):
    """
    Load all necessary components for the RAG system.

    Args:
        config (dict): Configuration dictionary
        data_dir (str): Directory containing data files
        device (str): Device to load models on

    Returns:
        tuple: (splits, index, client, embedder, simplification_model, simplification_tokenizer)
    """
    logger.info("Starting component loading process...")

    # Load data and index
    splits, index = load_data_and_index(config, data_dir)

    # Create main LLM (Gemini)
    client = setup_client()

    # Load embedder
    embedder = load_embedder(config, device)

    # Load simplification model (optional)
    simplification_model, simplification_tokenizer = load_simplification_model(config, device)

    logger.info("All components loaded successfully!")
    return splits, index, client, embedder, simplification_model, simplification_tokenizer
