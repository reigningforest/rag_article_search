import json
import os

import kagglehub
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


def download(dataset_kaggle: str, data_dir: str, dl_file_name: str) -> tuple[str, str]:
    """
    Download the arxiv dataset from Kaggle if not already present.

    Args:
        dataset_kaggle (str): Kaggle dataset identifier (e.g., 'Cornell-University/arxiv')
        data_dir (str): Directory to store the dataset
        dl_file_name (str): Name of the file to download

    Returns:
        tuple[str, str]: Tuple of (data_dir_path, data_file_path)
    """
    logger.info("DOWNLOAD START!")
    # Get root directory
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # data directory is in the specified folder in the script directory
    data_dir_path = os.path.join(root_dir, data_dir)

    # Create data directory if it doesn't exist
    os.makedirs(data_dir_path, exist_ok=True)

    # Download the dataset and get the path
    # Check if data set is already downloaded
    data_file_path = os.path.join(data_dir_path, dl_file_name)
    if os.path.exists(data_file_path):
        logger.info(f"Dataset already downloaded to {data_dir_path}")

        return data_dir_path, data_file_path

    dl_dir = kagglehub.dataset_download(dataset_kaggle)

    logger.info(f"Files downloaded to {dl_dir}")

    # Move files from download directory to data directory
    src_file = os.path.join(dl_dir, dl_file_name)
    os.rename(src_file, data_file_path)

    logger.info(f"Files moved to {data_dir_path}")

    return data_dir_path, data_file_path


def _read_json_lines(data_file_path: str) -> list[dict]:
    """
    Read a JSON lines file and return a list of dicts.

    Args:
        data_file_path (str): Path to the JSON lines file

    Returns:
        list[dict]: List of parsed JSON objects
    """
    data = []
    with open(data_file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by parsing dates and stripping titles.

    Args:
        df (pd.DataFrame): DataFrame to clean

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df["update_date"] = pd.to_datetime(df["update_date"])
    df["title"] = df["title"].apply(lambda x: x.strip())
    return df


def filter_abstracts(
    data_dir_path: str, data_file_path: str, filter_date, filter_file_name: str
) -> pd.DataFrame:
    """
    Filter the arxiv dataset by date and save as a pickle file.

    Args:
        data_dir_path (str): Directory to save the filtered file
        data_file_path (str): Path to the raw data file
        filter_date: Minimum update_date to keep (datetime or str)
        filter_file_name (str): Name of the pickle file to save

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    logger.info("FILTERING START!")
    logger.info(f"Reading abstracts from {data_file_path}...")
    data = _read_json_lines(data_file_path)
    df = pd.DataFrame(data)
    df = _clean_dataframe(df)
    logger.info(f"Read {len(df)} abstracts from {data_file_path}")
    df = df[df["update_date"] >= filter_date]
    df.to_pickle(os.path.join(data_dir_path, filter_file_name))
    logger.info(f"Filtered abstracts saved to {data_dir_path}")

    return df
