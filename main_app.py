"""
Streamlit application for the Agentic RAG Article Search System.
Single entry point - uses existing pipeline functions when needed.
"""

import streamlit as st
import os
import yaml
import time
import torch
from pathlib import Path
from dotenv import load_dotenv

# Internal imports
from scripts.get_data_pipeline import main as run_pipeline_main
from src.models import load_all_components
from src.rag import build_rag_graph

# Load environment variables at module level
load_dotenv()


def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config") / "config.yaml"
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file: {e}")
        st.stop()


def check_data_exists(config):
    """Check if processed data files exist."""
    data_dir = config["data_dir"]
    required_files = [
        config["embeddings_selected_file_name"],
        config["chunk_selected_file_name"],
    ]

    existing_files = []
    for file in required_files:
        file_path = Path(data_dir) / file
        if file_path.exists():
            existing_files.append(file)

    return len(existing_files) == len(required_files), existing_files


def check_dependencies():
    """Check if required dependencies are available."""
    dependencies = {
        "torch": False,
        "pinecone": False,
        "fastembed": False,
        "langchain": False,
    }

    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            pass

    # Check environment variables
    env_vars = ["GEMINI_API_KEY", "PINECONE_API_KEY"]
    env_status = {}
    for var in env_vars:
        env_status[var] = bool(os.getenv(var))

    return dependencies, env_status


def handle_progress_error(progress_bar, status_text, error_msg):
    """Consolidated error handling for progress tracking."""
    progress_bar.progress(100)
    status_text.text("❌ Query processing failed")
    return {"error": f"Query execution failed: {error_msg}"}


def run_data_pipeline():
    """Run the data processing pipeline using existing function."""
    try:
        run_pipeline_main()
        return True
    except Exception as e:
        st.error(f"Data pipeline failed: {str(e)}")
        return False


def initialize_rag_system():
    """Initialize RAG components using existing function."""
    try:
        config = load_config()

        # Check for required API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("GEMINI_API_KEY not found in environment variables.")
            return None

        # Set up device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load components
        data_dir = config["data_dir"]
        splits, index, client, embedder = load_all_components(
            config, data_dir, device
        )

        # Build RAG graph
        rag_graph = build_rag_graph(
            splits, index, client, embedder, config
        )

        return rag_graph

    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None


def initialize_rag_system_with_progress():
    """Initialize RAG system with detailed progress tracking."""

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
            # Step 1: Import components
            update_progress(steps[0]["name"], steps[0]["progress"])

            # Step 2: Load config
            update_progress(steps[1]["name"], steps[1]["progress"])
            config = load_config()

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

            # Step 5-8: Load components (this is where the heavy lifting happens)
            update_progress(steps[4]["name"], steps[4]["progress"])
            data_dir = config["data_dir"]

            # We'll advance progress during component loading
            update_progress(steps[5]["name"], steps[5]["progress"])
            time.sleep(0.5)  # Simulate loading time

            update_progress(steps[6]["name"], steps[6]["progress"])
            splits, index, client, embedder = load_all_components(
                config, data_dir, device
            )

            # Step 9: Build graph
            update_progress(steps[7]["name"], steps[7]["progress"])
            rag_graph = build_rag_graph(
                splits, index, client, embedder, config
            )

            # Final step
            update_progress(steps[8]["name"], steps[8]["progress"])
            time.sleep(0.5)  # Brief pause to show completion

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


def query_rag_system(rag_graph, query):
    """Query the RAG system."""
    try:
        initial_state = {
            "query": query,
            "needs_arxiv": False,
            "rewrite": "",
            "documents": [],
            "simplified_documents": [],
            "response": "",
            "current_step": "initialized",
            "last_completed_step": "",
        }

        result = rag_graph.invoke(initial_state)
        return result

    except Exception as e:
        return {"error": f"Query execution failed: {str(e)}"}


def query_rag_system_with_progress(rag_graph, query, progress_bar, status_text):
    """Query the RAG system with simple callback-based progress tracking."""
    try:
        # Initialize
        progress_bar.progress(0)
        status_text.text("🚀 Starting query processing...")

        # Force Streamlit to update immediately
        time.sleep(0.1)

        # Progress mapping using simple tuples - showing current activity
        progress_steps = {
            "classify": (15, "🧠 Analyzing query..."),
            "rewrite_query": (35, "📝 Generating query variations..."),
            "retrieve": (55, "🔍 Retrieving relevant documents..."),
            "generate_rag_response": (75, "🤖 Generating response..."),
            "simplify_abstracts": (95, "✨ Simplifying abstracts..."),
            "direct_answer": (95, "🤖 Generating direct response..."),
        }

        def update_progress(step_name):
            """Simple callback to update progress directly."""
            print(f"Progress callback called for: {step_name}")  # Debug print
            if step_name in progress_steps:
                progress, status = progress_steps[step_name]
                progress_bar.progress(progress)
                status_text.text(status)
                # Force Streamlit to update
                time.sleep(0.2)
                print(f"Updated progress to {progress}% - {status}")  # Debug print
            else:
                print(f"Unknown step: {step_name}")  # Debug print

        # Create initial state with callback
        initial_state = {
            "query": query,
            "needs_arxiv": False,
            "rewrite": "",
            "documents": [],
            "simplified_documents": [],
            "response": "",
            "current_step": "initialized",
            "last_completed_step": "",
            "progress_callback": update_progress,  # Add callback to state
        }

        print("Starting RAG graph execution with progress callback...")  # Debug print

        # Single execution - no threading, no double execution
        result = rag_graph.invoke(initial_state)

        print(
            f"RAG graph completed. Final step: {result.get('last_completed_step', '')}"
        )  # Debug print

        # Show final completion
        final_step = result.get("last_completed_step", "")
        if final_step == "direct_answer":
            status_text.text("🤖 Generated direct response!")
        else:
            status_text.text("🤖 Generated intelligent response!")

        progress_bar.progress(100)
        return result

    except Exception as e:
        print(f"Error in query_rag_system_with_progress: {str(e)}")  # Debug print
        return handle_progress_error(progress_bar, status_text, str(e))


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Agentic RAG Article Search", page_icon="🔍", layout="wide"
    )

    st.title("🔍 Agentic RAG Article Search System")

    # Load configuration
    config = load_config()

    # Check system status
    dependencies, env_status = check_dependencies()
    data_exists, existing_files = check_data_exists(config)

    # Sidebar status
    with st.sidebar:
        st.header("System Status")

        st.subheader("Dependencies")
        for dep, status in dependencies.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {dep}")

        st.subheader("Environment Variables")
        for var, status in env_status.items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {var}")

        st.subheader("Data Files")
        if data_exists:
            st.success("✅ Processed data available")
        else:
            st.warning("⚠️ No processed data found")

    # Main interface
    if not data_exists:
        st.warning("⚠️ No processed data found. Please run the data pipeline first.")

        st.subheader("Data Pipeline")
        st.info(
            "This will download ArXiv papers, process them, and create embeddings. This may take several hours and requires significant storage space."
        )

        if st.button("🚀 Run Data Pipeline", type="primary"):
            # Check for kaggle.json file
            kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
            if not kaggle_json_path.exists():
                st.error(
                    "Kaggle authentication required. Please place your kaggle.json file in ~/.kaggle/kaggle.json"
                )
            else:
                with st.spinner(
                    "Running data pipeline... This may take several hours."
                ):
                    success = run_data_pipeline()
                    if success:
                        st.success("✅ Data pipeline completed successfully!")
                        st.rerun()

    else:
        # Data exists, show RAG interface
        st.success("✅ System ready for queries!")

        # Initialize RAG system if not already done
        if "rag_graph" not in st.session_state:
            st.session_state.rag_graph = initialize_rag_system_with_progress()

        if st.session_state.rag_graph is None:
            st.error(
                "❌ Failed to initialize RAG system. Check your environment variables."
            )
        else:
            # Query interface
            st.subheader("💬 Ask Questions")

            # Create a form to handle Enter key press
            with st.form(key="search_form", clear_on_submit=False):
                query = st.text_input(
                    "Enter your question:",
                    placeholder="What are the recent developments in transformer architectures?",
                    key="query_input",
                )

                # Submit button for the form (handles Enter key)
                form_submit = st.form_submit_button("🔍 Search", type="primary")

            # Process search when form is submitted
            search_triggered = form_submit

            if search_triggered:
                if not query or query.strip() == "":
                    st.error("⚠️ Please enter a proper question before searching!")
                else:
                    # Create progress tracking elements
                    progress_container = st.empty()
                    status_container = st.empty()

                    with progress_container.container():
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        result = query_rag_system_with_progress(
                            st.session_state.rag_graph, query, progress_bar, status_text
                        )

                        if result and "error" in result:
                            st.error(f"❌ {result['error']}")
                        elif result:
                            st.session_state.last_result = result
                            st.session_state.last_query = query
                        else:
                            st.error("❌ Query processing failed - no result returned")

                        # Clear progress elements after completion
                        progress_container.empty()
                        status_container.empty()

            # Refresh Data button
            if st.button("🔄 Refresh Data"):
                # Show warning modal
                st.warning("⚠️ **Data Refresh Warning**")
                st.markdown("""
                **This will completely refresh your dataset and may take several hours.**
                
                **The process will:**
                1. 🗑️ Delete existing processed data files
                2. 📥 Re-download the entire ArXiv dataset from Kaggle (~4GB)
                3. 🔄 Re-process and chunk all papers
                4. 🧮 Re-generate embeddings for all chunks
                5. 📤 Re-upload embeddings to Pinecone vector database
                6. 💾 Save new processed files locally
                
                **Requirements:**
                - Stable internet connection
                - ~20GB free disk space
                - Several hours of processing time
                - Valid Kaggle and Pinecone API credentials
                """)

                col_confirm, col_cancel = st.columns(2)

                with col_confirm:
                    if st.button(
                        "⚠️ YES, Refresh Everything",
                        type="secondary",
                        help="This will take several hours",
                    ):
                        # Check for kaggle.json file
                        kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
                        if not kaggle_json_path.exists():
                            st.error(
                                "Kaggle authentication required. Please place your kaggle.json file in ~/.kaggle/kaggle.json"
                            )
                        else:
                            with st.spinner(
                                "Refreshing data... This may take several hours."
                            ):
                                success = run_data_pipeline()
                                if success:
                                    st.success("✅ Data refreshed successfully!")
                                    # Clear cached RAG system to reload with new data
                                    if "rag_graph" in st.session_state:
                                        del st.session_state.rag_graph
                                    st.rerun()

                with col_cancel:
                    if st.button("❌ Cancel", type="primary"):
                        st.rerun()

            # Display results
            if (
                "last_result" in st.session_state
                and "last_query" in st.session_state
                and st.session_state.last_result
            ):
                st.subheader("📝 Response")
                st.write(f"**Question:** {st.session_state.last_query}")

                response = (
                    st.session_state.last_result.get(
                        "response", "No response generated"
                    )
                    if isinstance(st.session_state.last_result, dict)
                    else "No response generated"
                )
                st.write(response)

                # Show query rewrite first (above source documents)
                if (
                    isinstance(st.session_state.last_result, dict)
                    and "rewrite" in st.session_state.last_result
                    and st.session_state.last_result["rewrite"]
                ):
                    with st.expander(
                        "🔄 Enhanced Query Perspective",
                        expanded=False,
                    ):
                        st.markdown("**🎯 Original Query:**")
                        st.info(f"_{st.session_state.last_query}_")

                        st.markdown("**🔄 Enhanced Query Perspective:**")
                        st.markdown(
                            "*This enhanced perspective helps find more relevant papers by optimizing the search query.*"
                        )

                        # Display the single rewrite
                        rewrite = st.session_state.last_result["rewrite"]
                        st.markdown(f"• {rewrite}")

                # Show additional info in expandable sections - prefer simplified documents
                display_documents = st.session_state.last_result.get("simplified_documents", [])
                if not display_documents:
                    display_documents = st.session_state.last_result.get("documents", [])
                
                if (
                    isinstance(st.session_state.last_result, dict)
                    and display_documents
                ):
                    simplified_text = " (Simplified)" if "simplified_documents" in st.session_state.last_result and st.session_state.last_result["simplified_documents"] else ""
                    with st.expander(
                        f"📄 Source Documents{simplified_text} ({len(display_documents)} found)",
                        expanded=False,
                    ):
                        for i, doc in enumerate(display_documents, 1):
                            with st.container():
                                # Create a styled header for each document
                                st.markdown(f"### 📋 **Source {i}**")

                                # Display basic info first
                                if isinstance(doc, dict):
                                    if "title" in doc:
                                        st.markdown(
                                            f"**📝 Title:** {doc.get('title', 'No title')}"
                                        )
                                    if "date" in doc:
                                        st.markdown(
                                            f"**📅 Date:** {doc.get('date', 'No date')}"
                                        )

                                # Create tabs for different content views
                                if isinstance(doc, dict) and "text" in doc:
                                    abstract_text = doc.get("text", "")

                                    # Try to split original and simplified versions
                                    if (
                                        "ORIGINAL TEXT:" in abstract_text
                                        and "SIMPLIFIED VERSION:" in abstract_text
                                    ):
                                        parts = abstract_text.split(
                                            "SIMPLIFIED VERSION:"
                                        )
                                        original_text = (
                                            parts[0]
                                            .replace("ORIGINAL TEXT:", "")
                                            .strip()
                                        )
                                        simplified_text = (
                                            parts[1].strip()
                                            if len(parts) > 1
                                            else "No simplified version available"
                                        )

                                        # Create tabs
                                        tab1, tab2, tab3 = st.tabs(
                                            [
                                                "✨ Simplified",
                                                "📄 Original",
                                                "ℹ️ Details",
                                            ]
                                        )

                                        with tab1:
                                            st.text_area(
                                                "AI-Simplified Summary",
                                                simplified_text,
                                                height=300,
                                                disabled=True,
                                                key=f"doc_{i}_simplified",
                                            )

                                        with tab2:
                                            st.text_area(
                                                "Original Abstract",
                                                original_text,
                                                height=300,
                                                disabled=True,
                                                key=f"doc_{i}_original",
                                            )

                                        with tab3:
                                            if "metadata" in doc:
                                                st.json(doc.get("metadata", {}))
                                            else:
                                                st.write(
                                                    "No additional metadata available"
                                                )
                                    else:
                                        # Single content version - show in simplified tab
                                        tab1, tab2 = st.tabs(
                                            ["📄 Content", "ℹ️ Details"]
                                        )

                                        with tab1:
                                            st.text_area(
                                                "Abstract/Content",
                                                abstract_text,
                                                height=300,
                                                disabled=True,
                                                key=f"doc_{i}_content",
                                            )

                                        with tab2:
                                            if "metadata" in doc:
                                                st.json(doc.get("metadata", {}))
                                            else:
                                                st.write(
                                                    "No additional metadata available"
                                                )
                                else:
                                    # Handle simple text document
                                    st.text_area(
                                        f"Document {i} Content",
                                        str(doc),
                                        height=300,
                                        disabled=True,
                                        key=f"doc_text_{i}",
                                    )

                                # Add separator between documents (except for the last one)
                                if i < len(st.session_state.last_result["documents"]):
                                    st.divider()


if __name__ == "__main__":
    main()
