import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from pinecone import Pinecone
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


def get_env():
    '''Load environment variables'''
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    env_path = os.path.join(current_file_dir, '.env')
    load_dotenv(dotenv_path=env_path)


def loading(folder_name, index_name='abstract-index'):
    '''
    Load the data, index and embedder
    '''
    print("Loading started...")
    # Load the data directory
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(current_file_dir, folder_name)

    # Load the data chunks(splits)
    splits = pd.read_pickle(os.path.join(data_dir, 'chunked_abstracts.pkl'))
    print('- Chunks loaded')

    # Load the Pinecone index
    pinecone = Pinecone()
    index = pinecone.Index(index_name)
    print('- Index loaded')

    # Load the embedder
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print('- Embedder loaded')

    print("Loading completed")

    return current_file_dir, splits, index, embedder

def retrieve_docs(current_file_dir, query_str, embedder, index, splits, top_k=3):
    '''
    Embeds the query and retrieves the top-k similar chunks
    '''
    # if the output folder doesn't exist, create it
    output_path = os.path.join(current_file_dir, 'output')
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
    with open(os.path.join(output_path, 'output_db_query.txt'), 'w') as f:
        f.write(f"Original Input: {query_str}\n\n")
        f.write("Similar Chunks:\n")
        for match in results.matches:
            idx = int(match.id)
            f.write(f"Similarity: {match.score:.4f}\n")
            f.write(f"Text: {selected_chunks.loc[idx]['chunk_text']}\n\n")
    print(f"Saved similar chunks to 'output_db_query.txt'")

def main(folder_name, query_str):
    # Load environment variables
    get_env()

    # Load everything
    current_file_dir, splits, index, embedder = loading(folder_name)

    # Retrieve docs
    retrieve_docs(current_file_dir, query_str, embedder, index, splits, top_k=3)


if __name__ == '__main__':
    query_str = input('Enter the query: ')

    main('data', query_str)

