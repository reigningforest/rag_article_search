import os
import kagglehub
import json
from dotenv import load_dotenv
import yaml

import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

import torch  # Must come first
import onnxruntime as ort

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding


def download(dataset_kaggle, data_dir, dl_file_name):
    '''
    Downloads the arxiv dataset
    1. Creates a data directory under the root directory
    2. Checks if the dataset is already downloaded
        1. If the data set is not downloaded, downloads the arxiv dataset from Kaggle
        2. Moves the dataset to the data directory
    3. Filters the dataset
    '''
    print("DOWNLOAD START!")
    # Get root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # data directory is in the specified folder in the script directory
    data_dir_path = os.path.join(root_dir, data_dir)
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir_path, exist_ok=True)

    # Download the dataset and get the path
    # Check if data set is already downloaded
    if os.path.exists(os.path.join(data_dir_path, dl_file_name)):

        print(f"Dataset already downloaded to {data_dir_path}")
        
        return data_dir_path, os.path.join(data_dir_path, dl_file_name)
    
    dl_dir = kagglehub.dataset_download(dataset_kaggle)

    print(f"Files downloaded to {dl_dir}")

    # Move files from download directory to data directory
    src_file = os.path.join(dl_dir, dl_file_name)
    data_file_path = os.path.join(data_dir_path, dl_file_name)
    os.rename(src_file, data_file_path)

    print(f"Files moved to {data_dir_path}")

    return data_dir_path, data_file_path


def filter_abstracts(data_dir_path, data_file_path, filter_date, filter_file_name):
    '''
    Filters the arxiv dataset and saves as a pickle file
    1. Reads in the JSON dataset
    2. Converts the dataset to dataframe
    3. Filters the dataset by date
    4. Saves the filtered dataset as a pickle file
    '''
    print("FILTERING START!")
    # Read the JSON file line by line
    data = []

    print(f"Reading abstracts from {data_file_path}...")

    with open(data_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert to DataFrame and clean columns as needed
    df = pd.DataFrame(data)
    df['update_date'] = pd.to_datetime(df['update_date'])
    df['title'] = df['title'].apply(lambda x: x.strip())

    print(f"Read {len(df)} abstracts from {data_file_path}")

    # Filter by date
    df = df[df['update_date'] >= filter_date]
    
    # Save the filtered abstracts as a pickle file
    df.to_pickle(os.path.join(data_dir_path, filter_file_name))
    
    print(f"Filtered abstracts saved to {data_dir_path}")

    return df


def chunk_texts_with_index(data_dir_path, df, chunk_file_name, chunk_size, chunk_overlap, min_text_len):
    '''
    Chunk the abstracts in the dataframe and save the chunks to a pickle file.
    Parameters:
        data_dir_path (str): The directory where the chunked abstracts pickle file will be saved.
        df (pd.DataFrame): The dataframe containing the abstracts to be chunked.
        chunk_file_name (str): The name of the pickle file to save the chunked abstracts.
        chunk_size (int, optional): The maximum size of each chunk.
        chunk_overlap (int, optional): The number of characters that overlap between chunks.
        min_text_len (int, optional): The minimum length of text to be considered for chunking.
    Returns:
        pd.DataFrame: A dataframe containing the chunked text, the original index of the text, and the chunk ID.
    Notes:
        If the chunked abstracts already exist in the data directory ('filtered_abstracts.pkl'), they will be loaded from the pickle file.
        The text is split into chunks using the RecursiveCharacterTextSplitter.
        The resulting chunks and their corresponding original indexes are saved to a new dataframe, which is then saved to a pickle file.
    '''
    print("CHUNKING START!")
    # Create a text splitter that splits text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    original_indexes = []
    titles = []
    update_dates = []

    # Iterate over the rows in the dataframe and split the text into chunks
    for row in df.itertuples(index=True):
        text = row.abstract
        # Skip empty or very short texts
        if pd.isna(text) or len(str(text)) < min_text_len:
            continue
            
        # Split the text into chunks
        text_chunks = text_splitter.split_text(str(text))
        chunks.extend(text_chunks)

        # Associate the original index with each chunk
        original_indexes.extend([row.Index] * len(text_chunks))
        titles.extend([row.title] * len(text_chunks))
        update_dates.extend([row.update_date] * len(text_chunks))

    chunks_df = pd.DataFrame({
        'chunk_text': chunks,
        'original_index': original_indexes,
        'chunk_id': range(len(chunks)),
        'title': titles,
        'update_date': update_dates
    })

    print(f"Created {len(chunks)} chunks from {len(df)} abstracts")

    chunks_df.to_pickle(os.path.join(data_dir_path, chunk_file_name))

    print(f"Saved chunked abstracts to {data_dir_path}")

    return chunks_df


def embed_chunks(data_dir_path, chunks, batch_size, save_every, save_checkpoints, fast_embed_name, embeddings_file_name):
    '''
    Embeds text chunks using a specified text embedding model and saves the embeddings to a file.
    Parameters:
        data_dir_path (str): Directory where the embeddings will be saved.
        chunks (np.ndarray): Array of text chunks to be embedded.
        batch_size (int): Number of texts to embed at once.
        save_every (int): Number of chunks to process before saving a checkpoint.
        save_checkpoints (bool): Whether to save checkpoints after processing save_every chunks.
        fast_embed_name (str): Name of the fastembed text embedding model to use.
        embeddings_file_name (str): Name of the file to save the final embeddings.
    Returns:
        np.ndarray: Array of embedded text chunks.
    Notes:
        If embeddings already exist in the data directory, they will be loaded from the file.
        The function processes the chunks in batches and can save intermediate checkpoints if specified.
        You must use a text embedding model that is compatible with the fastembed library.
        Final embeddings are saved to 'embeddings_final.npy' in the data directory.
    '''
    print("EMBEDDING START!")
    # Check for CUDA availability
    providers = ort.get_available_providers()
    print("Available providers:", providers)
    if "CUDAExecutionProvider" in providers:
        print("CUDA is available for text embedding")
    else:
        print("CUDA is not available, please install a GPU-enabled version of onnxruntime-gpu to use CUDA for text embedding. Also ensure that torch is installed with CUDA support.")
        return None

    # Create the text embedding model
    embedding_model = TextEmbedding(
        model_name=fast_embed_name,
        batch_size=batch_size,  # This controls how many texts are embedded at once
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    
    # Embed the chunks in batches and save checkpoints
    all_embeddings = []
    chunks_list = chunks.tolist()
    
    for i in tqdm(range(0, len(chunks_list), save_every)):
        checkpoint_chunks = chunks_list[i:i + save_every]
        # The embedding model will internally process these in batches of batch_size
        batch_embeddings = list(embedding_model.embed(checkpoint_chunks))

        # Normalize embeddings here
        batch_embeddings = [vec / np.linalg.norm(vec) for vec in batch_embeddings]

        all_embeddings.extend(batch_embeddings)
        
        if save_checkpoints:
            # Save checkpoint after each save_every chunks
            if i > 0:
                np.save(os.path.join(data_dir_path, f'embeddings_checkpoint_{i}.npy'), np.array(all_embeddings))
    
    print("Finished embedding chunks")

    # Save final embeddings
    final_embeddings = np.array(all_embeddings)
    np.save(os.path.join(data_dir_path, embeddings_file_name), final_embeddings)

    print(f"Saved final embeddings to {data_dir_path}")
    
    return final_embeddings


def pinecone_upload(pc_index, embeddings, batch_size, distance_metric, pc_cloud, pc_region):
    '''
    Upload embeddings to Pinecone index
    1. Initialize Pinecone
    2. Check if index exists and create a new Pinecone index
    3. Connect to existing index
    4. Upsert data by id in batches
    '''
    print("PINECONE UPLOAD START!")
    # Initialize Pinecone
    pc = Pinecone()

    # Check if index exists, otherwise create a new one
    '''
    Check if an index exists:
    If it does: Delete and create a new Pinecone index
    If it doesn't: Create a new Pinecone index
    '''

    print(f"Checking for index: {pc_index}")
    
    # Check for existing indexes
    existing_indexes = pc.list_indexes().names()
    if pc_index in existing_indexes:
        print(f"Deleting index: '{pc_index}'")
        pc.delete_index(pc_index)
        
    # requery existing indexes if you deleted the index
    print(f"Creating index: '{pc_index}'")
    pc.create_index(
        name=pc_index,
        dimension=embeddings.shape[1],
        metric=distance_metric,
        spec=ServerlessSpec(cloud=pc_cloud, region=pc_region)
    )

    # Connect to existing index
    index = pc.Index(pc_index)

    # Prepare data for upsert using position as ID
    vectors = []
    for i in tqdm(range(len(embeddings)), desc='Preparing vectors'):
        vectors.append({
            'id': str(i),
            'values': embeddings[i].tolist()
        })

    # Upsert in batches
    for i in tqdm(range(0, len(vectors), batch_size), desc='Upserting batches'):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print('Index upserted successfully!')


def main():
    '''
    Main pipeline
    1. Downloads the arxiv dataset
    2. Filters the dataset by date
    3. Chunks the dataset
    4. Embeds the dataset
    5. Upserts the dataset to pinecone
    '''
    # Load the configuration file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Load environment variables
    env_file_name = config['env_file']
    load_dotenv(dotenv_path=env_file_name)

    # Download the abstracts
    dataset_kaggle = config['dataset_kaggle']
    data_dir = config['data_dir']
    dl_file_name = config['dl_file_name']
    data_dir_path, data_file_path = download(dataset_kaggle, data_dir, dl_file_name)

    # Filter the abstracts
    date = config['date']
    filter_file_name = config['filter_file_name']
    df = filter_abstracts(data_dir_path, data_file_path, date, filter_file_name)

    # Chunk the abstracts
    chunk_file_name = config['chunk_file_name']
    chunk_size = config['chunk_size']
    chunk_overlap = config['chunk_overlap']
    min_text_len = config['min_text_len']
    chunk_df = chunk_texts_with_index(data_dir_path, df, chunk_file_name, chunk_size, chunk_overlap, min_text_len)
    del df

    # Embed the chunks
    fast_embed_name = config['fast_embed_name']
    ch_batch_size = config['ch_batch_size']
    save_every = config['save_every']
    save_checkpoints = config['save_checkpoints']
    embeddings_file_name = config['embeddings_file_name']
    embeddings = embed_chunks(data_dir_path, chunk_df['chunk_text'].values, ch_batch_size, save_every, save_checkpoints, fast_embed_name, embeddings_file_name)
    
    # set seed
    np.random.seed(42)

    # Randomly select embeddings
    chunks_selected = chunk_df.sample(n=10000, random_state=42)
    embeddings_selected = embeddings[chunks_selected.index]

    # save the embeddings
    np.save(os.path.join(data_dir_path, config['embeddings_selected_file_name']), embeddings_selected)
    chunks_selected.to_pickle(os.path.join(data_dir_path, config['chunk_selected_file_name']))
    
    del chunks_selected
    del embeddings_selected
    del chunk_df

    # upsert the embeddings to pinecone
    pc_index = config['pc_index']
    pc_batch_size = config['pc_batch_size']
    distance_metric = config['distance_metric']
    pc_cloud = config['pc_cloud']
    pc_region = config['pc_region']

    if embeddings is not None:
        pinecone_upload(pc_index, embeddings, pc_batch_size, distance_metric, pc_cloud, pc_region)

if __name__ == "__main__":
    # Authenticate if required
    # kagglehub.login()

    # Run the main function
    main()