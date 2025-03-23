import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from pinecone import Pinecone
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import yaml


def loading(data_dir, pc_index, chunk_file_name, fast_embed_name):
    '''
    Load the data, index and embedder
    '''
    print("Loading started...")
    # Load the data directory
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir_path = os.path.join(current_file_dir, data_dir)

    # Load the data chunks(splits)
    splits = pd.read_pickle(os.path.join(data_dir_path, chunk_file_name))
    print('- Chunks loaded')

    # Load the Pinecone index
    pinecone = Pinecone()
    index = pinecone.Index(pc_index)
    print('- Index loaded')

    # Load the embedder
    embedder = HuggingFaceEmbeddings(
        model_name=fast_embed_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print('- Embedder loaded')

    print("Loading completed")

    return current_file_dir, splits, index, embedder

def retrieve_docs(current_file_dir, query_str, embedder, index, splits, top_k, output_dir, output_db_query_file_name):
    '''
    Embeds the query and retrieves the top-k similar chunks
    '''
    # if the output folder doesn't exist, create it
    output_path = os.path.join(current_file_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Embed the query
    query_vector = embedder.embed_query(query_str)

    # Query the index
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=False,
        include_values=False
    )

    # Identify and return chunks from the splits DataFrame
    chunk_ids = [int(match.id) for match in results.matches]
    selected_chunks = splits.iloc[chunk_ids]

    # Save original question and similar chunks to a file
    with open(os.path.join(output_path, output_db_query_file_name), 'w') as f:
        f.write(f"Original Input: {query_str}\n\n")
        f.write("Similar Chunks:\n")
        for match in results.matches:
            idx = int(match.id)
            f.write(f"Similarity: {match.score:.4f}\n")
            f.write(f"Text: {selected_chunks.loc[idx]['chunk_text']}\n\n")
    print(f"Saved similar chunks to 'output_db_query.txt'")

def main():
    # Text query input
    query_str = input('Enter the text query: ')
    query_top_k = input('Enter the number of similar chunks to retrieve: ')

    # Load in config variables
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load environment variables
    env_file_name = config['env_file']
    load_dotenv(dotenv_path=env_file_name)

    # Load everything
    data_dir = config['data_dir']
    pc_index = config['pc_index']
    chunk_file_name = config['chunk_file_name']
    fast_embed_name = config['fast_embed_name']
    current_file_dir, splits, index, embedder = loading(data_dir, pc_index, chunk_file_name, fast_embed_name)

    # Retrieve docs
    output_dir = config['output_dir']
    output_db_query_file_name = config['output_db_query_file_name']
    query_top_k = int(query_top_k)
    retrieve_docs(current_file_dir, query_str, embedder, index, splits, query_top_k, output_dir, output_db_query_file_name)


if __name__ == '__main__':
    main()

