"""
Abstract simplification functions for the RAG workflow.
"""

from typing import Dict, Any
from tqdm import tqdm
import torch

from ..state import RAGState
from ...connections.gemini_query import query_client
from ...connections.logger import get_shared_logger

logger = get_shared_logger(__name__)


def create_finetune_simplify_node(model: Any, tokenizer: Any, config: dict):
    """Create an interactive abstract simplification node function using fine-tuned SmolLM2."""

    def simplify_node(state: RAGState) -> Dict[str, Any]:
        documents = state["documents"]
        simplified_documents = state.get("simplified_documents", [])

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("simplify_abstracts")

        # Display available documents for selection
        logger.info("\nAvailable documents for simplification:")
        for i, doc in enumerate(documents):
            # Show title if available, otherwise show first 100 chars of text
            title = doc.get("title", doc["text"][:100] + "...")
            simplified_status = " (ALREADY SIMPLIFIED)" if any(
                sd.get("original_index") == i for sd in simplified_documents
            ) else ""
            logger.info(f"{i}: {title}{simplified_status}")

        # Get user selection
        while True:
            selection = input("\nEnter document numbers to simplify (e.g., 0,1,3 or 'all'): ").strip()
            
            if selection.lower() == "all":
                selected_indices = list(range(len(documents)))
                break
            else:
                try:
                    # Parse comma-separated indices
                    selected_indices = [int(x.strip()) for x in selection.split(",") if x.strip()]
                    # Validate indices
                    if all(0 <= idx < len(documents) for idx in selected_indices):
                        break
                    else:
                        logger.warning(f"Please enter valid document numbers (0-{len(documents)-1})")
                except ValueError:
                    logger.warning("Please enter valid numbers separated by commas, or 'all'")

        # Filter out already simplified documents
        new_indices = [idx for idx in selected_indices 
                      if not any(sd.get("original_index") == idx for sd in simplified_documents)]
        
        if not new_indices:
            logger.info("All selected documents are already simplified.")
        else:
            logger.info(f"Simplifying {len(new_indices)} new abstracts using fine-tuned model")

            # Get generation parameters from config
            gen_params = config.get("simplification_model", {}).get("generation_params", {})
            
            # Simplify selected documents
            for idx in tqdm(new_indices, desc="Simplifying selected abstracts"):
                doc = documents[idx]
                
                # Format the prompt with SmolLM2 chat template
                system_message = """You are a technical abstract explainer. Your explanation should:
    - Use clear, accessible language
    - Retain all technical terms from the original
    - Give a high-level overview of the background of abstract's topic
    - Briefly define any specialized jargon when first mentioned
    - Maintain the core meaning and relationships between concepts
    - Organize information in a logical flow"""
                
                prompt = f"""<|im_start|>system
{system_message}
<|im_end|>
<|im_start|>user
{doc['text']}
<|im_end|>
<|im_start|>assistant"""

                # Tokenize and generate
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=gen_params.get("max_length", 1000),
                        temperature=gen_params.get("temperature", 0.7),
                        do_sample=gen_params.get("do_sample", True),
                        top_p=gen_params.get("top_p", 0.9),
                        pad_token_id=gen_params.get("pad_token_id", 2),
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode response
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract only the assistant's response
                simplified_text = full_response.split("<|im_start|>assistant")[-1].strip()

                # Create a new document that contains both original and simplified text
                enhanced_doc = doc.copy()
                enhanced_doc["text"] = (
                    f"ORIGINAL TEXT:\n{doc['text']}\n\nSIMPLIFIED VERSION:\n{simplified_text}"
                )
                enhanced_doc["original_index"] = idx  # Track which original document this came from

                simplified_documents.append(enhanced_doc)

        # Ask if user wants to continue
        while True:
            continue_choice = input("\nContinue simplifying more abstracts? (yes/no): ").strip().lower()
            if continue_choice in ["yes", "no"]:
                break
            logger.warning("Please enter 'yes' or 'no'")

        continue_simplification = continue_choice == "yes"

        logger.debug(f"Simplified documents: {simplified_documents}")

        return {
            "simplified_documents": simplified_documents,
            "continue_simplification": continue_simplification,
            "current_step": "abstracts_enhanced",
            "last_completed_step": "simplify_abstracts",
        }

    return simplify_node


def create_gemini_simplify_node(client: Any, simplifier_prompt: str, model: str):
    """Create an interactive abstract simplification node function using Gemini."""

    def simplify_node(state: RAGState) -> Dict[str, Any]:
        documents = state["documents"]
        simplified_documents = state.get("simplified_documents", [])

        # Call progress callback at START of node
        if "progress_callback" in state and callable(state["progress_callback"]):
            state["progress_callback"]("simplify_abstracts")

        # Display available documents for selection
        logger.info("\nAvailable documents for simplification:")
        for i, doc in enumerate(documents):
            # Show title if available, otherwise show first 100 chars of text
            title = doc.get("title", doc["text"][:100] + "...")
            simplified_status = " (ALREADY SIMPLIFIED)" if any(
                sd.get("original_index") == i for sd in simplified_documents
            ) else ""
            logger.info(f"{i}: {title}{simplified_status}")

        # Get user selection
        while True:
            selection = input("\nEnter document numbers to simplify (e.g., 0,1,3 or 'all'): ").strip()
            
            if selection.lower() == "all":
                selected_indices = list(range(len(documents)))
                break
            else:
                try:
                    # Parse comma-separated indices
                    selected_indices = [int(x.strip()) for x in selection.split(",") if x.strip()]
                    # Validate indices
                    if all(0 <= idx < len(documents) for idx in selected_indices):
                        break
                    else:
                        logger.warning(f"Please enter valid document numbers (0-{len(documents)-1})")
                except ValueError:
                    logger.warning("Please enter valid numbers separated by commas, or 'all'")

        # Filter out already simplified documents
        new_indices = [idx for idx in selected_indices 
                      if not any(sd.get("original_index") == idx for sd in simplified_documents)]
        
        if not new_indices:
            logger.info("All selected documents are already simplified.")
        else:
            logger.info(f"Simplifying {len(new_indices)} new abstracts")

            # Simplify selected documents
            for idx in tqdm(new_indices, desc="Simplifying selected abstracts"):
                doc = documents[idx]
                
                # Format the prompt with the abstract
                formatted_prompt = simplifier_prompt.format(abstract=doc["text"])

                # Get simplified version of the abstract using Gemini
                simplified_text = query_client(client, formatted_prompt, model)

                # Create a new document that contains both original and simplified text
                enhanced_doc = doc.copy()
                enhanced_doc["text"] = (
                    f"ORIGINAL TEXT:\n{doc['text']}\n\nSIMPLIFIED VERSION:\n{simplified_text}"
                )
                enhanced_doc["original_index"] = idx  # Track which original document this came from

                simplified_documents.append(enhanced_doc)

        # Ask if user wants to continue
        while True:
            continue_choice = input("\nContinue simplifying more abstracts? (yes/no): ").strip().lower()
            if continue_choice in ["yes", "no"]:
                break
            logger.warning("Please enter 'yes' or 'no'")

        continue_simplification = continue_choice == "yes"

        logger.debug(f"Simplified documents: {simplified_documents}")

        return {
            "simplified_documents": simplified_documents,
            "continue_simplification": continue_simplification,
            "current_step": "abstracts_enhanced",
            "last_completed_step": "simplify_abstracts",
        }

    return simplify_node


def create_simplify_node(config: dict, client: Any = None, simplifier_prompt: str | None = None, 
                        model: str | None = None, simplification_model: Any = None, 
                        simplification_tokenizer: Any = None):
    """
    Create a simplification node based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        client (Any): Gemini client (required if using Gemini)
        simplifier_prompt (str): Prompt for Gemini (required if using Gemini)
        model (str): Gemini model name (required if using Gemini)
        simplification_model (Any): Fine-tuned model (required if using fine-tuned)
        simplification_tokenizer (Any): Fine-tuned tokenizer (required if using fine-tuned)
    
    Returns:
        function: Simplification node function
    """
    simplification_config = config.get("simplification_model", {})
    use_finetuned = simplification_config.get("use_finetuned", False)
    
    if use_finetuned and simplification_model is not None and simplification_tokenizer is not None:
        logger.info("Using fine-tuned model for simplification")
        return create_finetune_simplify_node(simplification_model, simplification_tokenizer, config)
    else:
        if use_finetuned:
            logger.warning("Fine-tuned model requested but not available, falling back to Gemini")
        else:
            logger.info("Using Gemini model for simplification")
        
        if client is None or simplifier_prompt is None or model is None:
            raise ValueError("Gemini client, prompt, and model name are required when not using fine-tuned model")
        
        return create_gemini_simplify_node(client, simplifier_prompt, model)
