# System and environment configuration
import os

# Core libraries
import yaml
from typing import TypedDict, List, Dict, Any, Literal
from tqdm import tqdm

# Deep learning and model related
import torch
from torch.cuda import is_available
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Data handling
import pandas as pd

# Environment configuration
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


def loading(config, data_dir, model_dir, device):
    '''
    Load the Pinecone index, data chunks, LLM, and embedder
    '''
    pc_index = config['pc_index']
    chunk_file_name = config['chunk_file_name']
    llm_model_name = config['llm_model_name']
    fast_embed_name = config['fast_embed_name']

    def load_simplifier_model(model_dir, device):
        # Load the tokenizer from model path
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load the PEFT configuration to get base model name
        peft_config = PeftConfig.from_pretrained(model_dir)
        base_model_name = peft_config.base_model_name_or_path
        
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Load the LoRA adapter onto the base model
        lora_model = PeftModel.from_pretrained(base_model, model_dir)
        
        # Move to device and set to evaluation mode
        lora_model.to(device)
        lora_model.eval()
        
        return lora_model, tokenizer
    
    # Load the data chunks(splits)
    splits = pd.read_pickle(os.path.join(data_dir, chunk_file_name))
    
    # Load the Pinecone index
    pinecone = Pinecone()
    index = pinecone.Index(pc_index)
    print('Index loaded')
    
    # Create LLM
    llm = ChatOpenAI(model=llm_model_name)
    print('LLM loaded')
    
    # Load the simplifier model
    simplifier_model, simplifier_tokenizer = load_simplifier_model(model_dir, device)
    
    print('Simplifier model loaded')

    # Load embedder
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

    return splits, index, llm, embedder, simplifier_model, simplifier_tokenizer


def query_llm(abstract, question_template, model, tokenizer, device):
    '''Queries the LLM model with the given abstracts and question template'''
    # Format prompt and tokenize
    question = question_template.format(abstract=abstract)
    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.7,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Extract only new generated text (excluding prompt)
    input_length = inputs.input_ids.shape[1]
    response = tokenizer.decode(
        outputs[0][input_length:], 
        skip_special_tokens=True
    )
    
    return response.strip()


def build_rag_graph(splits, index, llm, embedder, config, simplifier_model, simplifier_tokenizer, device):
    '''
    Build the RAG graph for the LangGraph system
    '''
    classification_prompt = config['classification_prompt']
    rewrite_prompt = config['rewrite_prompt']
    final_prompt = config['final_prompt']
    simplifier_prompt = config['hugging_face_template']
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


    # Node to simplify abstracts
    def simplify_abstracts_node(state: RAGState) -> Dict[str, Any]:
        documents = state["documents"]
        print(f"Enhancing {len(documents)} abstracts with simplified versions")
        
        enhanced_documents = []
        for doc in tqdm(documents, desc="Simplifying abstracts"):
            # Get simplified version of the abstract
            simplified_text = query_llm(doc['text'], simplifier_prompt, simplifier_model, simplifier_tokenizer, device)
            
            # Create a new document that contains both original and simplified text
            enhanced_doc = doc.copy()
            enhanced_doc['text'] = f"ORIGINAL TEXT:\n{doc['text']}\n\nSIMPLIFIED VERSION:\n{simplified_text}"
            
            enhanced_documents.append(enhanced_doc)
        
        print(f"Enhanced {len(enhanced_documents)} abstracts with simplified versions")
        return {"documents": enhanced_documents, "current_step": "abstracts_enhanced"}

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
    workflow.add_node("simplify_abstracts", simplify_abstracts_node)
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
    workflow.add_edge("retrieve", "simplify_abstracts")
    workflow.add_edge("simplify_abstracts", "generate_rag_response")
    workflow.add_edge("generate_rag_response", END)
    workflow.add_edge("direct_answer", END)
    
    # Compile the graph with memory saver for persistence
    graph = workflow.compile()

    return graph


def visualize_graph(graph, output_path):
    """Visualize the graph and save it as a PNG file"""
    # check if the output path already exists
    if os.path.exists(output_path):
        print(f"File '{output_path}' already exists. Please delete it or choose a different path.")
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
    model_dir = config['model_dir']
    output_dir = config['output_dir']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set the device
    device = "cuda" if is_available() else "cpu"
    if is_available() == False:
        return "CUDA not available. Please enable CUDA for better performance."
    else:
        print(f"Using device: {device}")

    # Load the Pinecone index, data chunks, LLM, and embedder
    splits, index, llm, embedder, simplifier_model, simplifier_tokenizer = loading(config, data_dir, model_dir, device)
    
    # Build the RAG graph
    rag_graph = build_rag_graph(splits, index, llm, embedder, config, simplifier_model, simplifier_tokenizer, device)

    # Visualize the graph
    graph_output_path = os.path.join(output_dir, config['fine_tuned_graph_file_name'])
    visualize_graph(rag_graph, graph_output_path)
    
    print("LangGraph RAG System initialized")
    print("Enter 'exit' to quit the program.")
    
    while True:
        query_str = input("Enter a question or type 'exit': ")
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

    
