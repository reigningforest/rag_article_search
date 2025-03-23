import os
import yaml
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from torch.cuda import is_available

def loading(config, data_dir, device):
    '''Load the Pinecone index, data chunks, LLM, and embedder'''
    pc_index = config['pc_index_selected']
    llm_model_name = config['llm_model_name']
    fast_embed_name = config['fast_embed_name']
    chunk_file_name = config['chunk_selected_file_name']

    # Load the data chunks(splits)
    splits = pd.read_pickle(os.path.join(data_dir, chunk_file_name))

    # Load the Pinecone index
    pinecone = Pinecone()
    index = pinecone.Index(pc_index)
    print('Index loaded')

    # Create LLM
    llm = ChatOpenAI(model=llm_model_name)
    print('LLM loaded')

    # Load the embedder
    embedder = HuggingFaceEmbeddings(
        model_name=fast_embed_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    print('Embedder loaded')

    return splits, index, llm, embedder

def query_rag_simple(splits, index, llm, embedder, config, query):
    top_k = config['top_k']
    base_prompt = config['base_prompt']

    # Create retriever function
    def retrieve_docs(inputs):
        q = inputs["query"]  # Extract the string from the dictionary
        query_vector = embedder.embed_query(q)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=False
        )
        chunk_ids = [int(match.id) for match in results.matches]
        return splits.iloc[chunk_ids]  # Now returns DataFrame rows
    
    # Format documents function
    def format_docs(docs):
        return "\n\n".join(
            f"Title: {row.title}\nDate: {row.update_date.date()}\n{row.chunk_text}"
            for row in docs.itertuples()
        )

    # Create prompt template
    template = PromptTemplate(
        input_variables=["query", "context"],
        template = base_prompt
    )

    # Build the RAG chain
    context = RunnableLambda(retrieve_docs) | format_docs
    rag_chain_prompt = (
        {"context": context, "query": RunnableLambda(lambda x: x["query"])}
        | template
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain_prompt.invoke({"query": query})

    return response


def main():
    # load in the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data_dir = config['data_dir']

    # Set the device
    device = "cuda" if is_available() else "cpu"

    # Load the environment variables
    load_dotenv(dotenv_path=config['env_file'])

    # Load the splits, index, llm, and embedder
    splits, index, llm, embedder = loading(config, data_dir, device)

    print("Enter 'exit' to quit the program.")
    while True:
        query = input("Enter a query or type 'exit': ")
        if query == "exit":
            break
        answer = query_rag_simple(splits, index, llm, embedder, config, query)
        print(answer)


if __name__ == "__main__":
    main()