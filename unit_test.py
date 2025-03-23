# System and environment configuration
import os

# Core libraries
import yaml

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

# YAML configuration
import yaml
from torch.cuda import is_available

# LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# RAG system
from agentic_rag_finetune import build_rag_graph as build_rag_graph_finetune


def loading(config, data_dir, model_dir, device):
    '''
    Load the Pinecone index, data chunks, LLM, and embedder
    '''
    pc_index = config['pc_index_selected']
    chunk_file_name = config['chunk_selected_file_name']
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


def process_prompts(config, device):
    """Process prompts using the RAG system"""
    print("Starting prompt processing with RAG system...")

    data_dir = config['data_dir']
    model_dir = config['model_dir']
    output_dir = config['output_dir']

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, config['unit_test_file_name'])
    
    splits, index, llm, embedder, simplifier_model, simplifier_tokenizer = loading(config, data_dir, model_dir, device)
    
    # Build the RAG graph
    rag_graph = build_rag_graph_finetune(splits, index, llm, embedder, config, simplifier_model, simplifier_tokenizer, device)
    
    print("LangGraph RAG System initialized")
    
    # Read in prompts.txt
    with open('prompts.txt', 'r') as file:
        prompts = [p.strip() for p in file.readlines() if p.strip()]
    
    prompts = [prompts[-1]] # For testing purposes, only process 1 prompts

    # Process each prompt
    with open(output_path, 'w') as results_file:
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")
            
            # Initialize state for this prompt
            initial_state = {
                "query": prompt,
                "needs_arxiv": False,
                "rewrites": [],
                "documents": [],
                "response": "",
                "current_step": "initialized"
            }
            
            try:
                # Execute the graph with the query
                print(f"Running prompt through the RAG system...")
                
                # Get the final result
                result = rag_graph.invoke(initial_state)
                
                # Print the response
                print(f"\nResponse for prompt {i+1}:")
                print(result["response"])
                
                # Save to file
                results_file.write(f"Prompt {i+1}: {prompt}\n\n")
                results_file.write(f"Response {i+1}:\n{result['response']}\n\n")
                results_file.write("-" * 80 + "\n\n")
                
            except Exception as e:
                print(f"Error processing prompt {i+1}: {e}")
            
    print(f"\nAll prompts processed. Results saved to {output_path}")


def main():
    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Load the environment variables
    load_dotenv(dotenv_path=config['env_file'])
    
    device = "cuda" if is_available() else "cpu"
    if is_available() == False:
        return "CUDA not available. Please enable CUDA for better performance."
    else:
        print(f"Using device: {device}")

    try:
        process_prompts(config, device)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()