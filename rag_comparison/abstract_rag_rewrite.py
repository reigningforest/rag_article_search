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
from langchain_core.prompts import ChatPromptTemplate


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


def query_rag_rewrite(query_str, splits, index, llm, embedder, top_k):
    '''Query the RAG model with a given question'''
    # 1. Define rewrite chain outside retrieval
    rewrite_prompt = ChatPromptTemplate.from_template(
        """Generate 3 alternative search queries for: {question}
        
        Focus on different aspects of the question and use synonyms or related terms.
        """
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # 2. Generate query variants
    rewrites = rewrite_chain.invoke({"question": query_str}).split("\n")
    
    # 3. Create unified retriever function
    def retrieve_docs(inputs):
        queries = [inputs["question"]] + inputs["rewrites"]
        all_results = []
        
        for query in queries:
            query_vector = embedder.embed_query(query)
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=False
            )
            all_results.extend(results.matches)
        
        # Deduplication with score-based ranking
        unique_chunks = {match.id: match.score for match in all_results}
        sorted_chunks = sorted(unique_chunks.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:top_k*2]
        chunk_ids = [int(chunk[0]) for chunk in sorted_chunks]
        
        return splits.iloc[chunk_ids]

    # 4. Document formatting
    def format_docs(docs):
        return "\n\n".join(
            f"Title: {row.title}\nDate: {row.update_date.date()}\n{row.chunk_text}"
            for row in docs.itertuples()
        )

    # 5. Build RAG chain with proper input handling
    context = (
        RunnableLambda(lambda x: {"question": x["question"], "rewrites": x["rewrites"]})
        | RunnableLambda(retrieve_docs)
        | format_docs
    )
    
    template = PromptTemplate(
        input_variables=["question", "context", "rewrites"],
        template="""Consider these query perspectives: {rewrites}
        Answer {question} using context:
        {context}
        Guidelines:
        1. Resolve ambiguities using multiple query versions
        2. Weight recent papers higher
        3. Cross-validate findings across query variants
        4. Make sure not to repeat information across query variants"""
    )
    
    rag_chain = (
        {"question": RunnableLambda(lambda x: x["question"]), 
         "context": context,
         "rewrites": RunnableLambda(lambda x: "\n- ".join(x["rewrites"]))}
        | template
        | llm
        | StrOutputParser()
    )

    # 6. Execute with proper ordering
    answer = rag_chain.invoke({
        "question": query_str,
        "rewrites": rewrites
    })
    # context_chunks = retrieve_docs({"question": query_str, "rewrites": rewrites})
    
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
        answer = query_rag_rewrite(query_str, splits, index, llm, embedder, top_k)
        print(answer)
        # print(context_chunks)

if __name__ == "__main__":
    main()