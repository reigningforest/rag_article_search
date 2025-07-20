import numpy as np
from pinecone import Pinecone, ServerlessSpec, Vector
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger(__name__)


def _check_and_create_index(
    pc: Pinecone,
    pc_index: str,
    embeddings: np.ndarray,
    distance_metric: str,
    pc_cloud: str,
    pc_region: str,
) -> None:
    """
    Check if a Pinecone index exists, delete if it does, and create a new one.

    Args:
        pc (Pinecone): Pinecone client instance
        pc_index (str): Index name
        embeddings (np.ndarray): Embeddings array to determine dimension
        distance_metric (str): Metric for the index
        pc_cloud (str): Cloud provider
        pc_region (str): Cloud region
    """
    logger.info(f"Checking for index: {pc_index}")
    existing_indexes = pc.list_indexes().names()
    if pc_index in existing_indexes:
        logger.info(f"Deleting index: '{pc_index}'")
        pc.delete_index(pc_index)
    logger.info(f"Creating index: '{pc_index}'")
    pc.create_index(
        name=pc_index,
        dimension=embeddings.shape[1],
        metric=distance_metric,
        spec=ServerlessSpec(cloud=pc_cloud, region=pc_region),
    )


def _prepare_vectors(embeddings: np.ndarray) -> list[Vector]:
    """
    Prepare vectors for upsert to Pinecone.

    Args:
        embeddings (np.ndarray): Embeddings array

    Returns:
        list[Vector]: List of vectors with id and values
    """
    vectors = []
    for i in tqdm(range(len(embeddings)), desc="Preparing vectors"):
        vectors.append(Vector(id=str(i), values=embeddings[i].tolist()))
    return vectors


def pinecone_upload(
    pc_index: str,
    embeddings: np.ndarray,
    batch_size: int,
    distance_metric: str,
    pc_cloud: str,
    pc_region: str,
) -> None:
    """
    Upload embeddings to Pinecone index in batches.

    Args:
        pc_index (str): Index name
        embeddings (np.ndarray): Embeddings array
        batch_size (int): Batch size for upsert
        distance_metric (str): Metric for the index
        pc_cloud (str): Cloud provider
        pc_region (str): Cloud region
    """
    logger.info("PINECONE UPLOAD START!")
    pc = Pinecone()
    _check_and_create_index(
        pc, pc_index, embeddings, distance_metric, pc_cloud, pc_region
    )
    index = pc.Index(pc_index)
    vectors = _prepare_vectors(embeddings)
    for i in tqdm(range(0, len(vectors), batch_size), desc="Upserting batches"):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
    logger.info("Index upserted successfully!")
