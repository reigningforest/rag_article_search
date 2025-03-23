# System and environment configuration
import os

# Core libraries
import yaml
from typing import TypedDict, List, Dict, Any, Literal

# Deep learning and CUDA
from torch.cuda import is_available

# Data handling
import pandas as pd

# Environment and configuration
from dotenv import load_dotenv

# Vector database
from pinecone import Pinecone

# LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# Graph components
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import MermaidDrawMethod

# Define the state that will be passed between nodes
class RAGState(TypedDict):
    query: str
    needs_arxiv: bool
    rewrites: List[str]
    documents: List[Dict[str, Any]]
    response: str
    current_step: str


def loading(config, data_dir, device):
    '''
    Load the Pinecone index, data chunks, LLM, and embedder
    '''
    pc_index = config['pc_index']
    chunk_file_name = config['chunk_file_name']
    llm_model_name = config['llm_model_name']
    fast_embed_name = config['fast_embed_name']
    
    # Load the data chunks(splits)
    splits = pd.read_pickle(os.path.join(data_dir, chunk_file_name))
    
    # Load the Pinecone index
    pinecone = Pinecone()
    index = pinecone.Index(pc_index)

    print('Index loaded')
    
    # Create LLM
    llm = ChatOpenAI(model=llm_model_name)

    print('LLM loaded')
    
    # Load embeddings
    fast_embed_name = config['fast_embed_name']
    cache_dir = config['embedding_cache_dir']  # Already in your config.yaml
    
    # Initialize base embeddings
    base_embedder = HuggingFaceEmbeddings(
        model_name=fast_embed_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create a cached embedder
    store = LocalFileStore(f"{cache_dir}/embeddings_cache")
    embedder = CacheBackedEmbeddings.from_bytes_store(
        base_embedder,
        store,
        namespace=fast_embed_name
    )
    
    print('Cached embedder loaded')
    
    return splits, index, llm, embedder


def build_rag_graph(splits, index, llm, embedder, config):
    '''
    Build the RAG graph for the LangGraph system
    '''
    classification_prompt = config['classification_prompt']
    rewrite_prompt = config['rewrite_prompt']
    final_prompt = config['final_prompt']
    top_k = config['top_k']

    # Node to classify if query needs ArXiv data
    def classify_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        print(f"Classifying query")
    
        prompt = ChatPromptTemplate.from_template(classification_prompt)

        classify_chain = prompt | llm | StrOutputParser()
        content = classify_chain.invoke({"query": query}).lower().strip()
        needs_arxiv = content == "yes"
        
        print(f"Classification result: {needs_arxiv}")
        
        return {"needs_arxiv": needs_arxiv, "current_step": "classification_complete"}
    
    # Node to generate query rewrites
    def rewrite_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        print(f"Generating query rewrites")

        prompt = ChatPromptTemplate.from_template(rewrite_prompt)
        
        chain = prompt | llm | StrOutputParser()
        rewrites_text = chain.invoke({"query": query})
        rewrites = [r.strip() for r in rewrites_text.split("\n") if r.strip()]
        
        print(f"Generated {len(rewrites)} query rewrites: {rewrites}")
        
        return {"rewrites": rewrites, "current_step": "rewrites_generated"}
    
    # Node to retrieve documents
    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        rewrites = state["rewrites"]
        print(f"Retrieving documents for concatenated query and rewrites")
        
        # Concatenate all queries into one string
        concatenated_query = query + " " + " ".join(rewrites)
        print(f"Concatenated query: {concatenated_query}")
        
        # Generate a single embedding for the concatenated query
        query_vector = embedder.embed_query(concatenated_query)
        
        # Retrieve documents for the concatenated query
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=False
        )
        
        all_results = results.matches
        
        # Convert to documents as before
        chunk_ids = [int(match.id) for match in all_results]
        docs_df = splits.iloc[chunk_ids]
        
        # Convert DataFrame to list of dictionaries
        documents = []
        for _, row in docs_df.iterrows():
            doc = {
                "title": row.title,
                "date": str(row.update_date.date()),
                "text": row.chunk_text
            }
            documents.append(doc)
        
        print(f"Retrieved {len(documents)} documents")
        return {"documents": documents, "current_step": "documents_retrieved"}
    
    # Node to generate RAG response
    def generate_response_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        documents = state["documents"]
        rewrites = state["rewrites"]
        
        print(f"Generating RAG response using {len(documents)} documents")
        
        # Format documents
        formatted_docs = "\n\n".join(
            f"Title: {doc['title']}\nDate: {doc['date']}\n{doc['text']}"
            for doc in documents
        )
        
        prompt = PromptTemplate(
            input_variables=["question", "context", "rewrites"],
            template=final_prompt
        )
        
        formatted_rewrites = "\n- ".join(rewrites)
        
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "query": query,
            "context": formatted_docs,
            "rewrites": formatted_rewrites
        })
        
        return {"response": response, "current_step": "response_generated"}
    
    # Node to generate direct response without ArXiv
    def direct_response_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        print(f"Generating direct response for: {query}")
        
        response = llm.invoke(query).content
        
        return {"response": response, "current_step": "response_generated"}
    
    # Router function for conditional edges
    def route_based_on_classification(state: RAGState) -> Literal["rewrite_query", "direct_answer"]:
        if state["needs_arxiv"]:
            print("Query needs ArXiv data, routing to rewrite_query")
            return "rewrite_query"
        else:
            print("Query doesn't need ArXiv data, routing to direct_answer")
            return "direct_answer"
    
    # Create the StateGraph
    workflow = StateGraph(RAGState)
    
    # Add nodes to the graph
    workflow.add_node("classify", classify_node)
    workflow.add_node("rewrite_query", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_rag_response", generate_response_node)
    workflow.add_node("direct_answer", direct_response_node)
    
    # Add edges to connect nodes
    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_based_on_classification,
        {
            "rewrite_query": "rewrite_query",
            "direct_answer": "direct_answer"
        }
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "generate_rag_response")
    workflow.add_edge("generate_rag_response", END)
    workflow.add_edge("direct_answer", END)
    
    # Compile the graph with memory saver for persistence
    graph = workflow.compile()

    return graph


def visualize_graph(graph, output_path):
    """Visualize the graph and save it as a PNG file"""
    # check if the output path already exists
    if os.path.exists(output_path):
        print(f"graph visualization already exists at '{output_path}'")
        return False
    try:
        
        # Generate PNG data from the graph
        png_data = graph.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
            background_color="white",
            padding=10
        )
        
        # Save the PNG data to a file
        with open(output_path, "wb") as f:
            f.write(png_data)
        
        print(f"Graph visualization saved as '{output_path}'")
        return True
    except Exception as e:
        print(f"Error generating graph visualization: {e}")
        return False


def main():

    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the environment variables
    load_dotenv(dotenv_path=config['env_file'])
    
    # Load Directories
    data_dir = config['data_dir']
    output_dir = config['output_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the device
    device = "cuda" if is_available() else "cpu"
    if is_available() == False:
        return "CUDA not available. Please enable CUDA for better performance."
    else:
        print(f"Using device: {device}")

    # Load the splits, index, llm, and embedder
    pc_index = config['pc_index']
    chunk_file_name = config['chunk_file_name']
    llm_model_name = config['llm_model_name']
    fast_embed_name = config['fast_embed_name']
    data_dir = config['data_dir']
    
    splits, index, llm, embedder = loading(config, data_dir, device)
    
    # Set the top_k value
    top_k = config['top_k']
    
    # Build the RAG graph
    rag_graph = build_rag_graph(splits, index, llm, embedder, config)

    # Visualize the graph
    output_path = os.path.join(output_dir, config['base_graph_file_name'])
    visualize_graph(rag_graph, output_path)
    
    print("LangGraph RAG System initialized")
    print("Enter 'exit' to quit the program.")
    
    while True:
        query_str = input("Enter a question: ")
        if query_str == "exit":
            break
        
        # Initialize state
        initial_state = {
            "query": query_str, 
            "needs_arxiv": False, 
            "rewrites": [], 
            "documents": [], 
            "response": "",
            "current_step": "initialized"
        }

        # Execute the graph with the query
        try:
            # Get the final result
            result = rag_graph.invoke(initial_state)
            
            # Print the response
            print("\nResponse:")
            print(result["response"])
            
        except Exception as e:
            print(f"Error during execution: {e}")


if __name__ == "__main__":
    main()

    
