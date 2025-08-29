"""
RAG system initialization with progress tracking.
Uses existing load_all_components from src.models.model_loader.
"""

import os
import time
import torch
import streamlit as st
from src.models import load_all_components  # Reuse existing function
from src.rag import build_rag_graph  # Reuse existing function


def initialize_rag_system_with_progress(config):
    """
    Initialize RAG system with detailed progress tracking.
    Reuses existing load_all_components() function.
    """
    # Create progress tracking elements
    progress_container = st.empty()

    with progress_container.container():
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Define initialization steps
        steps = [
            {"name": "🔧 Loading configuration...", "progress": 10},
            {"name": "🔑 Validating API keys...", "progress": 20},
            {"name": "🖥️ Setting up device...", "progress": 30},
            {"name": "📊 Loading data chunks...", "progress": 45},
            {"name": "🗂️ Connecting to Pinecone index...", "progress": 60},
            {"name": "🤖 Loading Gemini models...", "progress": 75},
            {"name": "🧮 Loading embedding model...", "progress": 85},
            {"name": "🔗 Building RAG workflow graph...", "progress": 95},
            {"name": "✅ System ready!", "progress": 100},
        ]

        def update_progress(step_name, progress):
            progress_bar.progress(progress)
            status_text.text(step_name)
            time.sleep(0.3)  # Brief pause to show progress

        try:
            # Step 1-2: Basic setup
            update_progress(steps[0]["name"], steps[0]["progress"])
            update_progress(steps[1]["name"], steps[1]["progress"])
            
            # Step 3: Check API keys
            update_progress(steps[2]["name"], steps[2]["progress"])
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                progress_container.empty()
                st.error("GEMINI_API_KEY not found in environment variables.")
                return None

            # Step 4: Set up device
            update_progress(steps[3]["name"], steps[3]["progress"])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            data_dir = config["data_dir"]

            # Steps 5-7: Load components using existing function
            update_progress(steps[4]["name"], steps[4]["progress"])
            time.sleep(0.5)
            
            update_progress(steps[5]["name"], steps[5]["progress"])
            time.sleep(0.5)
            
            update_progress(steps[6]["name"], steps[6]["progress"])
            
            # Use existing load_all_components function
            splits, index, client, embedder = load_all_components(
                config, data_dir, device
            )

            # Step 8: Build graph using existing function
            update_progress(steps[7]["name"], steps[7]["progress"])
            rag_graph = build_rag_graph(
                splits, index, client, embedder, config
            )

            # Final step
            update_progress(steps[8]["name"], steps[8]["progress"])
            time.sleep(0.5)

            # Clear progress elements after completion
            progress_container.empty()
            return rag_graph

        except Exception as e:
            # Show error in progress tracking
            progress_bar.progress(100)
            status_text.text("❌ System initialization failed")
            time.sleep(1)
            progress_container.empty()
            st.error(f"Failed to initialize RAG system: {str(e)}")
            return None
