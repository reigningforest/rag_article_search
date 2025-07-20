import os
import numpy as np
import yaml
from dotenv import load_dotenv

from src.loaders.data_loader import download, filter_abstracts
from src.loaders.embedding import chunk_texts_with_index, embed_chunks
from src.connections.pinecone_db import pinecone_upload


def main():
    """
    Main pipeline
    1. Downloads the arxiv dataset
    2. Filters the dataset by date
    3. Chunks the dataset
    4. Embeds the dataset
    5. Upserts the dataset to pinecone
    """
    # Load the configuration file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load environment variables
    env_file_name = config["env_file"]
    load_dotenv(dotenv_path=env_file_name)

    # Download the abstracts
    dataset_kaggle = config["dataset_kaggle"]
    data_dir = config["data_dir"]
    dl_file_name = config["dl_file_name"]
    data_dir_path, data_file_path = download(dataset_kaggle, data_dir, dl_file_name)

    # Filter the abstracts
    date = config["date"]
    filter_file_name = config["filter_file_name"]
    df = filter_abstracts(data_dir_path, data_file_path, date, filter_file_name)

    # Chunk the abstracts
    chunk_file_name = config["chunk_file_name"]
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]
    min_text_len = config["min_text_len"]
    chunk_df = chunk_texts_with_index(
        data_dir_path, df, chunk_file_name, chunk_size, chunk_overlap, min_text_len
    )
    del df

    # Embed the chunks
    fast_embed_name = config["fast_embed_name"]
    ch_batch_size = config["ch_batch_size"]
    save_every = config["save_every"]
    save_checkpoints = config["save_checkpoints"]
    embeddings_file_name = config["embeddings_file_name"]
    embeddings = embed_chunks(
        data_dir_path,
        chunk_df["chunk_text"].values,
        ch_batch_size,
        save_every,
        save_checkpoints,
        fast_embed_name,
        embeddings_file_name,
    )

    # set seed
    np.random.seed(42)

    # Check if embeddings were successfully generated
    if embeddings is None:
        print("Embeddings generation failed. Exiting pipeline.")
        return

    # Randomly select embeddings
    chunks_selected = chunk_df.sample(n=10000, random_state=42)
    embeddings_selected = embeddings[chunks_selected.index]

    # save the embeddings
    np.save(
        os.path.join(data_dir_path, config["embeddings_selected_file_name"]),
        embeddings_selected,
    )
    chunks_selected.to_pickle(
        os.path.join(data_dir_path, config["chunk_selected_file_name"])
    )

    del chunks_selected
    del embeddings_selected
    del chunk_df

    # upsert the embeddings to pinecone
    pc_index = config["pc_index"]
    pc_batch_size = config["pc_batch_size"]
    distance_metric = config["distance_metric"]
    pc_cloud = config["pc_cloud"]
    pc_region = config["pc_region"]

    if embeddings is not None:
        pinecone_upload(
            pc_index, embeddings, pc_batch_size, distance_metric, pc_cloud, pc_region
        )


if __name__ == "__main__":
    # Authenticate if required
    # kagglehub.login()

    # Run the main function
    main()
