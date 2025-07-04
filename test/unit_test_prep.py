import numpy as np
from dotenv import load_dotenv
import yaml
import os

from download_filter_embed_upsert import pinecone_upload


def prepare_pinecone_index(config):
    """Handle the embedding and Pinecone upload process separately"""
    print("Starting Pinecone index preparation...")

    # Load the embedded data
    data_dir = config["data_dir"]
    embeddings_file_name = config["embeddings_selected_file_name"]

    embeddings = np.load(os.path.join(data_dir, embeddings_file_name))

    # upsert the embeddings to pinecone
    pc_index = config["pc_index_selected"]
    pc_batch_size = config["pc_batch_size"]
    distance_metric = config["distance_metric"]
    pc_cloud = config["pc_cloud"]
    pc_region = config["pc_region"]

    pinecone_upload(
        pc_index, embeddings, pc_batch_size, distance_metric, pc_cloud, pc_region
    )

    # Return the index name for use in the second function
    return pc_index


def main():
    # Load the config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the environment variables
    load_dotenv(dotenv_path=config["env_file"])

    # Prepare Pinecone index (TensorFlow heavy)
    pc_index = prepare_pinecone_index(config)

    print(f"Embeddings Upserted to Pinecone index {pc_index}")


if __name__ == "__main__":
    main()
