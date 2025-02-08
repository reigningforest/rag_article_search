import os
import kagglehub
import json
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

import torch  # Must come first
import onnxruntime as ort

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

def get_env():
    '''Load environment variables'''
    print("Loading environment variables")
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    env_path = os.path.join(current_file_dir, '.env')
    load_dotenv(dotenv_path=env_path)

def download(dataset, filename):
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
    data_dir = os.path.join(root_dir, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Download the dataset and get the path
    # Check if data set is already downloaded
    if os.path.exists(os.path.join(data_dir, filename)):

        print(f"Dataset already downloaded to {data_dir}")
        
        return data_dir, os.path.join(data_dir, filename)
    
    dl_dir = kagglehub.dataset_download(dataset)

    print(f"Files downloaded to {dl_dir}")

    # Move files from download directory to data directory
    src_file = os.path.join(dl_dir, filename)
    data_file = os.path.join(data_dir, filename)
    os.rename(src_file, data_file)

    print(f"Files moved to {data_dir}")

    return data_dir, data_file


def filter_abstracts(data_dir, data_file, date):
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

    print(f"Reading abstracts from {data_file}...")

    with open(data_file, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Convert to DataFrame and clean columns as needed
    df = pd.DataFrame(data)
    df['update_date'] = pd.to_datetime(df['update_date'])
    df['title'] = df['title'].apply(lambda x: x.strip())

    print(f"Read {len(df)} abstracts from {data_file}")

    # Filter by date
    df = df[df['update_date'] >= date]
    
    # Save the filtered abstracts as a pickle file
    df.to_pickle(os.path.join(data_dir, 'filtered_abstracts.pkl'))
    
    print(f"Filtered abstracts saved to {data_dir}")

    return df


def chunk_texts_with_index(data_dir, df, chunk_size=2000, chunk_overlap=200, min_text_len = 50):
    '''
    Chunk the abstracts in the dataframe and save the chunks to a pickle file.
    Parameters:
        data_dir (str): The directory where the chunked abstracts pickle file will be saved.
        chunk_size (int, optional): The maximum size of each chunk. Default is 2000.
        chunk_overlap (int, optional): The number of characters that overlap between chunks. Default is 200.
        min_text_len (int, optional): The minimum length of text to be considered for chunking. Default is 50.
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

    chunks_df.to_pickle(os.path.join(data_dir, 'chunked_abstracts.pkl'))

    print(f"Saved chunked abstracts to {data_dir}")

    return chunks_df


def embed_chunks(data_dir, chunks, batch_size=256, save_every=40000, save_checkpoints=False, fast_embed_name="BAAI/bge-small-en-v1.5"):
    '''
    Embeds text chunks using a specified text embedding model and saves the embeddings to a file.
    Parameters:
        data_dir (str): Directory where the embeddings will be saved.
        chunks (np.ndarray): Array of text chunks to be embedded.
        batch_size (int, optional): Number of texts to embed at once. Default is 256.
        save_every (int, optional): Number of chunks to process before saving a checkpoint. Default is 40000.
        save_checkpoints (bool, optional): Whether to save checkpoints after processing save_every chunks. Default is False.
        fast_embed_name (str, optional): Name of the fastembed text embedding model to use. Default is "BAAI/bge-small-en-v1.5".
    Returns:
        np.ndarray: Array of embedded text chunks.
    Notes:
        If embeddings already exist in the data directory, they will be loaded from the file.
        The function processes the chunks in batches and can save intermediate checkpoints if specified.
        You must use a text embedding model that is compatible with the fastembed library.
        Final embeddings are saved to 'embeddings_final.npy' in the data directory.
    '''
    print("EMBEDDING START!")
    # Create the text embedding model
    embedding_model = TextEmbedding(
        model_name=fast_embed_name,
        batch_size=batch_size,  # This controls how many texts are embedded at once
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        parallel=0
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
                np.save(os.path.join(data_dir, f'embeddings_checkpoint_{i}.npy'), np.array(all_embeddings))
    
    print("Finished embedding chunks")

    # Save final embeddings
    final_embeddings = np.array(all_embeddings)
    np.save(os.path.join(data_dir, 'embeddings_final.npy'), final_embeddings)

    print(f"Saved final embeddings to {data_dir}")
    
    return final_embeddings


def pinecone_upload(index_name, embeddings, batch_size = 500):
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

    print(f"Checking for index: {index_name}")
    
    # Check for existing indexes
    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        print(f"Deleting index: '{index_name}'")
        pc.delete_index(index_name)
        
    # requery existing indexes if you deleted the index
    print(f"Creating index: '{index_name}'")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Connect to existing index
    index = pc.Index(index_name)

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
    dataset = "Cornell-University/arxiv"
    filename = "arxiv-metadata-oai-snapshot.json" # This is expected!
    date = '2021-10-01'
    pc_index = 'abstract-index'

    # Load environment variables
    get_env()

    # Download the dataset and process the abstracts
    data_dir, data_file = download(dataset, filename)
    df = filter_abstracts(data_dir, data_file, date)

    # chunk and embed the abstracts
    df = chunk_texts_with_index(data_dir, df)
    embeddings = embed_chunks(data_dir, df['chunk_text'].values)

    # save on memory
    del df

    # upsert the embeddings to pinecone
    pinecone_upload(pc_index, embeddings, batch_size=500)

if __name__ == "__main__":
    # Authenticate if required
    # kagglehub.login()

    # Run the main function
    main()