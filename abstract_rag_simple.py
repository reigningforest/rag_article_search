import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pandas as pd
from dotenv import load_dotenv
import yaml
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def loading(pc_index, chunk_file_name, llm_model_name, fast_embed_name, data_dir):
    # Get the data directory
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(current_file_dir, data_dir)

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
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print('Embedder loaded')

    return splits, index, llm, embedder


def query_rag_simple(query_str, splits, index, llm, embedder, top_k):
    # Create retriever function
    def retrieve_docs(inputs):
        question = inputs["question"]  # Extract the string from the dictionary
        query_vector = embedder.embed_query(question)
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=False,
            include_values=False
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
        input_variables=["question", "context"],
        template = """Answer {question} using this context:

        {context}

        Consider these guidelines:
        1. Verify dates match the query's temporal requirements
        2. Use article titles to disambiguate similar concepts
        3. Prioritize recent papers (2023-2024) unless historical context is requested"""
    )

    # Build the RAG chain
    context = RunnableLambda(retrieve_docs) | format_docs
    rag_chain_prompt = (
        {"context": context, "question": RunnableLambda(lambda x: x["question"])}
        | template
        | llm
        | StrOutputParser()
    )

    # Execute the chain
    answer = rag_chain_prompt.invoke({"question": query_str})
    # context_chunks = retrieve_docs({"question": query_str})
    
    return answer


def main():
    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the environment variables
    env_file_name = config['env_file']
    load_dotenv(dotenv_path=env_file_name)

    # Load the splits, index, llm, and embedder
    pc_index = config['pc_index']
    chunk_file_name = config['chunk_file_name']
    llm_model_name = config['llm_model_name']
    fast_embed_name = config['fast_embed_name']
    data_dir = config['data_dir']
    splits, index, llm, embedder = loading(pc_index, chunk_file_name, llm_model_name, fast_embed_name, data_dir)

    # Set the top_k value
    top_k = config['top_k']
    print("Enter 'exit' to quit the program.")
    while True:
        query_str = input("Enter a question: ")
        if query_str == "exit":
            break
        answer = query_rag_simple(query_str, splits, index, llm, embedder, top_k)
        print(answer)
        # print(context_chunks)


if __name__ == "__main__":
    main()