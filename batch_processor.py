import os
import csv
import yaml
import torch
from datetime import datetime
from dotenv import load_dotenv

# Import from base_llm.py
from base_llm import query_llm

# Import from base_rag.py
from base_rag import query_rag_simple

# Import from agentic_rag.py
from agentic_rag import build_rag_graph as build_agentic_rag_graph

# Import from agentic_rag_finetune.py
from agentic_rag_finetune import loading, build_rag_graph as build_finetune_rag_graph

def get_timestamp():
    """Get a timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Load environment variables
    load_dotenv(dotenv_path=config['env_file'])
    
    # Set up directories
    data_dir = config['data_dir']
    model_dir = config['model_dir']
    output_dir = config['output_dir']
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Read in sample questions
    with open(os.path.join(config['prompt_file_name']), 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Set up CSV file
    timestamp = get_timestamp()
    csv_filename = os.path.join(output_dir, f"batch_results_{timestamp}.csv")
    
    # Load models
    splits, index, llm, embedder, simplifier_model, simplifier_tokenizer = loading(config, data_dir, model_dir, device)

    agentic_graph = build_agentic_rag_graph(splits, index, llm, embedder, config)
    
    finetune_graph = build_finetune_rag_graph(splits, index, llm, embedder, config, simplifier_model, simplifier_tokenizer, device)
    
    # Create CSV file and write header
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Question', 'Base LLM', 'Base RAG', 'Agentic RAG', 'Agentic RAG (Fine-tuned)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each question
        for i, question in enumerate(questions, 1):
            print(f"\nProcessing question {i}/{len(questions)}: {question}")
            row = {'Question': question}
            
            # Process with Base LLM
            try:
                print("Processing with Base LLM...")
                row['Base LLM'] = query_llm(llm, question)
            except Exception as e:
                print(f"Error with Base LLM: {e}")
                row['Base LLM'] = f"ERROR: {str(e)}"
            
            # Process with Base RAG
            try:
                print("Processing with Base RAG...")
                row['Base RAG'] = query_rag_simple(splits, index, llm, embedder, config, question)
            except Exception as e:
                print(f"Error with Base RAG: {e}")
                row['Base RAG'] = f"ERROR: {str(e)}"
            
            # Process with Agentic RAG
            try:
                print("Processing with Agentic RAG...")
                initial_state = {
                    "query": question,
                    "needs_arxiv": False,
                    "rewrites": [],
                    "documents": [],
                    "response": "",
                    "current_step": "initialized"
                }
                result = agentic_graph.invoke(initial_state)
                row['Agentic RAG'] = result["response"]
            except Exception as e:
                print(f"Error with Agentic RAG: {e}")
                row['Agentic RAG'] = f"ERROR: {str(e)}"
            
            # Process with Agentic RAG (Fine-tuned)
            try:
                print("Processing with Agentic RAG (Fine-tuned)...")
                initial_state = {
                    "query": question,
                    "needs_arxiv": False,
                    "rewrites": [],
                    "documents": [],
                    "response": "",
                    "current_step": "initialized"
                }
                result = finetune_graph.invoke(initial_state)
                row['Agentic RAG (Fine-tuned)'] = result["response"]
            except Exception as e:
                print(f"Error with Agentic RAG (Fine-tuned): {e}")
                row['Agentic RAG (Fine-tuned)'] = f"ERROR: {str(e)}"
            
            # Write row to CSV
            writer.writerow(row)
            csvfile.flush()
            
            print(f"Completed question {i}/{len(questions)}")
    
    print(f"\nAll questions processed. Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
