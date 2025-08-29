"""
Simple CLI for testing Gemini API prompts.
Uses existing functions from the codebase.
"""

import os
import yaml
from dotenv import load_dotenv

# Import existing functions
from src.connections.gemini_query import setup_client, query_client
from src.connections.logger import get_shared_logger

logger = get_shared_logger(__name__)


def load_config():
    """Load configuration to get model name."""
    with open("config/config.yaml", "r") as file:
        return yaml.safe_load(file)


def test_prompt():
    """Interactive prompt testing function."""
    # Load environment variables
    load_dotenv()
    
    # Load config to get model name
    config = load_config()
    model_name = config.get("gemini_model_name", "gemini-2.0-flash")
    
    # Setup Gemini client using existing function
    logger.info("Setting up Gemini client...")
    client = setup_client()
    
    print(f"🤖 Gemini Prompt Tester (Model: {model_name})")
    print("=" * 50)
    print("Enter your prompt below. Type 'quit' to exit.")
    print("Type 'hyde' to test the current HyDE prompt from config.")
    print("Type 'classify' to test the classification prompt.")
    print("Type 'simplify' to test the simplification prompt.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\n📝 Enter prompt (or command): ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'hyde':
            # Test HyDE prompt from config
            hyde_prompt = config.get("rewrite_prompt", "")
            test_query = input("Enter test query: ").strip()
            prompt = hyde_prompt.format(query=test_query)
            print(f"\n🔍 Testing HyDE prompt with query: '{test_query}'")
        elif user_input.lower() == 'classify':
            # Test classification prompt from config
            classification_prompt = config.get("classification_prompt", "")
            test_query = input("Enter test query: ").strip()
            prompt = classification_prompt.format(query=test_query)
            print(f"\n🧠 Testing classification prompt with query: '{test_query}'")
        elif user_input.lower() == 'simplify':
            # Test simplification prompt from config
            simplify_prompt = config.get("simplify_prompt", "")
            test_abstract = input("Enter test abstract: ").strip()
            prompt = simplify_prompt.format(abstract=test_abstract)
            print(f"\n✨ Testing simplification prompt")
        else:
            # Use raw prompt
            prompt = user_input
            print(f"\n💬 Testing custom prompt")
        
        if not prompt.strip():
            print("❌ Empty prompt, please try again.")
            continue
        
        print("\n" + "="*50)
        print("📤 PROMPT:")
        print("-" * 25)
        print(prompt)
        print("\n📥 RESPONSE:")
        print("-" * 25)
        
        # Query using existing function
        try:
            response = query_client(client, prompt, model_name)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("="*50)


def test_specific_prompt(prompt: str, query: str | None = None):
    """Test a specific prompt with optional query formatting."""
    load_dotenv()
    config = load_config()
    model_name = config.get("gemini_model_name", "gemini-2.0-flash")
    
    client = setup_client()
    
    # Format prompt if query is provided
    if query:
        formatted_prompt = prompt.format(query=query)
    else:
        formatted_prompt = prompt
    
    print(f"🤖 Testing with {model_name}")
    print("="*50)
    print("📤 PROMPT:")
    print(formatted_prompt)
    print("\n📥 RESPONSE:")
    print("-"*25)
    
    response = query_client(client, formatted_prompt, model_name)
    print(response)
    print("="*50)


if __name__ == "__main__":
    # Interactive mode
    test_prompt()
    
    # Example of testing specific prompt:
    # test_specific_prompt("What is machine learning?")
    
    # Example of testing with query formatting:
    # test_specific_prompt("Explain {query} in simple terms", "neural networks")
