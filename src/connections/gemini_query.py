from typing import Any
import google.generativeai as genai
import dotenv
import os
from utils.logger import get_logger

logger = get_logger(__name__)


def _configure_gemini(api_key: str) -> None:
    """
    Configure the Gemini API with the provided API key.

    Args:
        api_key (str): Gemini API key
    """
    genai.configure(api_key=api_key)  # type: ignore
    logger.info("Gemini API configured.")


def setup_gemini(api_key: str, model_name: str = "gemini-2.0-flash") -> Any:
    """
    Set up the Gemini generative model.

    Args:
        api_key (str): Gemini API key
        model_name (str): Name of the Gemini model to use

    Returns:
        Any: Gemini GenerativeModel instance
    """
    _configure_gemini(api_key)
    model = genai.GenerativeModel(model_name)  # type: ignore
    logger.info(f"Gemini model '{model_name}' initialized.")
    return model


def query_gemini(model: Any, prompt: str) -> str:
    """
    Query the Gemini model with a prompt.

    Args:
        model (Any): Gemini GenerativeModel instance
        prompt (str): Prompt to send to the model

    Returns:
        str: Model response text or error message
    """
    try:
        response = model.generate_content(prompt)
        logger.info("Gemini query successful.")
        return response.text
    except Exception as e:
        logger.error(f"Error querying Gemini: {str(e)}")
        return f"Error querying Gemini: {str(e)}"


if __name__ == "__main__":
    # Load environment variables from .env file
    dotenv.load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found in environment variables.")
        exit(1)

    # Initialize the model
    model = setup_gemini(GEMINI_API_KEY)

    # Example query
    prompt = "What is the capital of France?"
    result = query_gemini(model, prompt)
    print(result)
