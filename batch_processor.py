import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import csv
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from abstract_llm_only import query_llm
from abstract_rag_simple import query_rag_simple
from abstract_rag_classifier import query_rag_classifier
from abstract_rag_rewrite import query_rag_rewrite
from abstract_rag_2_advanced import query_rag_2_adv


def get_env():
    '''Load environment variables'''
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    env_path = os.path.join(current_file_dir, '.env')
    load_dotenv(dotenv_path=env_path)


def loading(index_name):
    # Load the environment variables
    get_env()

    # Get the data directory
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(current_file_dir, 'data')

    # Load the data chunks(splits)
    splits = pd.read_pickle(os.path.join(data_dir, 'chunked_abstracts.pkl'))

    # Load the Pinecone index
    pinecone = Pinecone()
    index = pinecone.Index(index_name)
    print('Index loaded')

    # Create LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    print('LLM loaded')

    # Load the embedder
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print('Embedder loaded')

    return current_file_dir, splits, index, llm, embedder


def write_txt(output_dir, output_file, prompts_list, responses):
    '''Write the prompts and responses to a text file'''
    with open(os.path.join(output_dir, output_file), 'w') as f:
        for query_str, response in zip(prompts_list, responses):
            f.write(f"Question: {query_str}\n")
            f.write(f"Answer: {response}\n\n")


def main():
    # vars
    index_name = 'abstract-index'
    prompts_file = 'prompts.txt'

    # Load the environment variables
    get_env()

    # Load the splits, index, llm, and embedder
    current_file_dir, splits, index, llm, embedder = loading(index_name)

    # Get the output directory
    output_dir = os.path.join(current_file_dir, 'output')

    # Process the prompts
    with open(os.path.join(current_file_dir, prompts_file), 'r') as f:
        prompts_list = [line.strip() for line in f if line.strip()]

    # initialize lists
    llm_responses = []
    rag_simple_responses = []
    rag_classifier_responses = []
    rag_rewrite_responses = []
    rag_2_adv_responses = []

    # Process the each prompt
    for query_str in tqdm(prompts_list, desc="Processing prompts"):
        # LLM batch
        llm_response = query_llm(llm, query_str)
        llm_responses.append(llm_response)

        # RAG simple batch
        rag_simple_response = query_rag_simple(query_str, splits, index, llm, embedder)
        rag_simple_responses.append(rag_simple_response)

        # RAG classifier batch
        rag_classifier_response = query_rag_classifier(query_str, splits, index, llm, embedder, sup_print=True)
        rag_classifier_responses.append(rag_classifier_response)

        # RAG rewrite batch
        rag_rewrite_response = query_rag_rewrite(query_str, splits, index, llm, embedder)
        rag_rewrite_responses.append(rag_rewrite_response)

        # RAG 2 advanced batch
        rag_2_adv_response = query_rag_2_adv(query_str, splits, index, llm, embedder, sup_print=True)
        rag_2_adv_responses.append(rag_2_adv_response)

    # Write responses to txt files
    write_txt(output_dir, 'output_llm.txt', prompts_list, llm_responses)
    write_txt(output_dir, 'output_rag_simple.txt', prompts_list, rag_simple_responses)
    write_txt(output_dir, 'output_rag_classifier.txt', prompts_list, rag_classifier_responses)
    write_txt(output_dir, 'output_rag_rewrite.txt', prompts_list, rag_rewrite_responses)
    write_txt(output_dir, 'output_rag_2_adv.txt', prompts_list, rag_2_adv_responses)

    # Write responses to a CSV file
    with open(os.path.join(output_dir, 'output_responses.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Prompt', 'LLM', 'RAG Simple', 'RAG Classifier', 'RAG Rewrite', 'RAG 2 Advanced'])
        for i in range(len(prompts_list)):
            writer.writerow([
                prompts_list[i],
                llm_responses[i],
                rag_simple_responses[i],
                rag_classifier_responses[i],
                rag_rewrite_responses[i],
                rag_2_adv_responses[i]
            ])

if __name__ == "__main__":
    main()