"""
Streamlit application for the Agentic RAG Article Search System.
Modular design using existing functions and UI components.
"""

import streamlit as st
from dotenv import load_dotenv

# Import UI modules that reuse existing functions
from src.ui.config import load_and_validate_config
from src.ui.components import render_sidebar, render_header
from src.ui.data_pipeline import handle_data_pipeline
from src.ui.query_handler import handle_query_interface
from src.ui.display import display_results


def main():
    """Main Streamlit application."""
    # Load environment variables at module level
    load_dotenv()

    st.set_page_config(
        page_title="Agentic RAG Article Search", 
        page_icon="🔍", 
        layout="wide"
    )

    # Load config and check system status
    config, system_status = load_and_validate_config()

    # Render header and sidebar
    render_header()
    render_sidebar(system_status)

    # Main application flow
    if not system_status["data_exists"]:
        handle_data_pipeline(config)
    else:
        handle_query_interface(config)
        display_results()


if __name__ == "__main__":
    main()
