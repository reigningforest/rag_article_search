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
from langchain.retrievers.multi_query import MultiQueryRetriever


def get_env(env_file):
    '''Load environment variables'''
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    env_path = os.path.join(current_file_dir, env_file)
    load_dotenv(dotenv_path=env_path)


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


def classify_with_gpt(query, llm):
    """Classify whether a query requires information from vectorDB"""
    classification_prompt = f"""Analyze this query: "{query}"

Decide whether to retrieve 2022-2024 ArXiv papers using these criteria:

**Use ArXiv Data (YES) if:**
1. Explicit time references: "recent", "last 3 years", "2022-2024", "current", "latest"
2. Emerging technologies: "new developments in...", "recent breakthroughs"
3. Active research fields examples: "LLM quantization", "CRISPR delivery systems"
4. Comparative analysis requests: "how has X changed since Sept 2021?"

**Skip ArXiv Data (NO) if:**
1. Historical facts: "discovery of...", "original paper about..."
2. Fundamental concepts: "basic principles of...", "what is...?"
3. Pre Sept 2021 references: "research before 2022", "traditional methods"

**Uncertainty Handling:**
- If unclear whether recency matters, default to YES
- If conflicting indicators exist, default to YES

Examples:
Query: "Latest advancements in mRNA vaccine delivery (2023)" = YES
Query: "Explain CRISPR-Cas9's basic mechanism" = NO
Query: "What is CRISPR-Cas9?" = NO

Output ONLY: <YES> or <NO>"""
    response = llm.invoke(classification_prompt)
    content = response.content.lower().strip()
    if content == "<yes>":
        return True
    elif content == "<no>":
        return False
    

def query_rag_rewrite(query_str, splits, index, llm, embedder, top_k=10):
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
        4. Summarize across query variants"""
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


def query_rag_2_adv(query_str, splits, index, llm, embedder, top_k, sup_print):
    '''Take in a query and determine whether Arxiv is needed. Then rewrite the prompt and query the LLM.'''
    # Classify query first to see if it needs arxiv
    needs_arxiv = classify_with_gpt(query_str, llm)
    if needs_arxiv:
        if sup_print == False:
            print("Query needs ArXiv data")
        response = query_rag_rewrite(query_str, splits, index, llm, embedder, top_k)
    else:
        if sup_print == False:
            print("Query does not need ArXiv data")
        response = llm.invoke(query_str).content  # Direct response
    return response


def main():
    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the environment variables
    env_file = config['env_file']
    get_env(env_file)

    # Load the splits, index, llm, and embedder
    pc_index = config['pc_index']
    chunk_file_name = config['chunk_file_name']
    llm_model_name = config['llm_model_name']
    fast_embed_name = config['fast_embed_name']
    data_dir = config['data_dir']
    splits, index, llm, embedder = loading(pc_index, chunk_file_name, llm_model_name, fast_embed_name, data_dir)

    # Set the top_k value
    top_k = config['top_k']
    
    # Set whether to suppress needs arxiv print
    # sup_print = config['sup_print']
    sup_print = False # for debug purposes!

    print("Enter 'exit' to quit the program.")
    while True:
        query_str = input("Enter a question: ")
        if query_str == "exit":
            break
        answer = query_rag_2_adv(query_str, splits, index, llm, embedder, top_k, sup_print)
        print(answer)
        # print(context_chunks)

if __name__ == "__main__":
    main()