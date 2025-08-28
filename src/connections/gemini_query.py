"""
Gemini API Client
"""

from typing import Any
from google import genai
import dotenv
import os
from src.connections.logger import get_shared_logger

logger = get_shared_logger(__name__)


def setup_client() -> Any:
    """
    Set up the Gemini generative model.

    Args:
        model_name (str): Name of the Gemini model to use

    Returns:
        Any: Gemini GenerativeModel instance
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables.")
        exit(1)
    
    # For google-genai, set the API key directly
    client = genai.Client(api_key=api_key)

    logger.info("LLM Client Loaded")

    return client


def query_client(client: Any, prompt: str, model_name: str = "gemini-2.0-flash") -> str:
    """
    Query a Gemini Model with a prompt.

    Args:
        client (Any): Gemini Client object
        prompt (str): Prompt to send to the model
        model_name (str): Gemini Model to use

    Returns:
        str: Model response text or error message
    """
    try:

        logger.debug(f"Querying {model_name}")

        response = client.models.generate_content(
            model=model_name,
            contents=prompt)
        
        usage = response.usage_metadata
        prompt_token_count = usage.prompt_token_count if usage else None
        candidates_token_count = usage.candidates_token_count if usage else None

        logger.debug(f"Prompt token count: {prompt_token_count}")
        logger.debug(f"Output token count: {candidates_token_count}")

        return response.text
    except Exception as e:
        logger.error(f"Error querying: {str(e)}")
        return f"Error querying: {str(e)}"


if __name__ == "__main__":
    # Load environment variables from .env file
    dotenv.load_dotenv()

    # Initialize the client
    client = setup_client()

    # Example query
    prompt = "What is the capital of France?"
    result = query_client(client, prompt)
    logger.info(f"Prompt: {prompt}\nResponse: {result}")