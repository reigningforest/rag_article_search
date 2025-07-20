"""
Streamlit application for the Agentic RAG Article Search System.
Single entry point - uses existing pipeline functions when needed.
"""

import streamlit as st
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

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


def run_data_pipeline():
    """Run the data processing pipeline using existing function."""
    try:
        from scripts.get_data_pipeline import main as run_pipeline_main

        run_pipeline_main()
        return True
    except Exception as e:
        st.error(f"Data pipeline failed: {str(e)}")
        return False


def initialize_rag_system():
    """Initialize RAG components using existing function."""
    try:
        # Import the components directly
        from src.models import load_all_components
        from src.core import build_rag_graph

        config = load_config()

        # Check for required API key
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("GEMINI_API_KEY not found in environment variables.")
            return None

        # Set up device
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load components
        data_dir = config["data_dir"]
        splits, index, gemini_llm, embedder, gemini_model = load_all_components(
            config, data_dir, device, gemini_api_key
        )

        # Build RAG graph
        rag_graph = build_rag_graph(
            splits, index, gemini_llm, embedder, config, gemini_model
        )

        return rag_graph

    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None


def query_rag_system(rag_graph, query):
    """Query the RAG system."""
    try:
        initial_state = {
            "query": query,
            "needs_arxiv": False,
            "rewrites": [],
            "documents": [],
            "response": "",
            "current_step": "initialized",
        }

        result = rag_graph.invoke(initial_state)
        return result

    except Exception as e:
        return {"error": f"Query execution failed: {str(e)}"}


def query_rag_system_with_progress(rag_graph, query, progress_bar, status_text):
    """Query the RAG system with detailed progress tracking."""
    import time

    try:
        # Define the workflow steps
        steps = [
            {"name": "🧠 Analyzing query intent...", "progress": 15},
            {"name": "🤔 Determining response strategy...", "progress": 30},
            {"name": "📝 Generating query variations...", "progress": 45},
            {"name": "🔍 Searching vector database...", "progress": 60},
            {"name": "📄 Processing relevant papers...", "progress": 75},
            {"name": "✨ Generating intelligent response...", "progress": 90},
            {"name": "✅ Finalizing results...", "progress": 100},
        ]

        def update_progress(step_name, progress):
            progress_bar.progress(progress)
            status_text.text(step_name)
            time.sleep(0.3)  # Brief pause to show progress

        # Step 1: Initialize query
        update_progress(steps[0]["name"], steps[0]["progress"])

        initial_state = {
            "query": query,
            "needs_arxiv": False,
            "rewrites": [],
            "documents": [],
            "response": "",
            "current_step": "initialized",
        }

        # Step 2: Start processing
        update_progress(steps[1]["name"], steps[1]["progress"])

        # We'll track the workflow by monitoring state changes
        # This is a simplified approach - in a more advanced implementation,
        # you could modify the nodes to emit progress events

        result = None
        for i, step in enumerate(steps[2:], 2):  # Start from step 2
            if i < len(steps):
                update_progress(step["name"], step["progress"])

            if i == 2:  # After showing "query variations", start the actual graph
                result = rag_graph.invoke(initial_state)
                # If we get here, the graph completed successfully
                break

        # Ensure we reach 100%
        update_progress("✅ Query completed successfully!", 100)
        time.sleep(0.5)  # Brief pause to show completion

        return result

    except Exception as e:
        # Show error in progress tracking
        progress_bar.progress(100)
        status_text.text("❌ Query processing failed")
        return {"error": f"Query execution failed: {str(e)}"}


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
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_graph = initialize_rag_system()

        if st.session_state.rag_graph is None:
            st.error(
                "❌ Failed to initialize RAG system. Check your environment variables."
            )
        else:
            # Query interface
            st.subheader("💬 Ask Questions")

            query = st.text_input(
                "Enter your question:",
                placeholder="What are the recent developments in transformer architectures?",
            )

            col1, col2 = st.columns([1, 4])

            with col1:
                if st.button("🔍 Search", type="primary") and query:
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

            with col2:
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

                # Show additional info in expandable sections
                if (
                    isinstance(st.session_state.last_result, dict)
                    and "documents" in st.session_state.last_result
                    and st.session_state.last_result["documents"]
                ):
                    with st.expander(
                        f"📄 Source Documents ({len(st.session_state.last_result['documents'])} found)",
                        expanded=False,
                    ):
                        for i, doc in enumerate(
                            st.session_state.last_result["documents"], 1
                        ):
                            with st.container():
                                # Create a styled header for each document
                                st.markdown(f"### 📋 **Source {i}**")

                                # Add document content in a styled box
                                with st.container():
                                    # Check if document is a dict with more structure or just text
                                    if isinstance(doc, dict):
                                        # Handle structured document
                                        if "title" in doc:
                                            st.markdown(
                                                f"**📝 Title:** {doc.get('title', 'No title')}"
                                            )
                                        if "content" in doc:
                                            st.markdown("**📄 Content:**")
                                            st.text_area(
                                                f"Document {i} Content",
                                                doc.get("content", ""),
                                                height=150,
                                                disabled=True,
                                                key=f"doc_{i}",
                                            )
                                        if "metadata" in doc:
                                            st.markdown("**ℹ️ Metadata:**")
                                            st.json(doc.get("metadata", {}))
                                    else:
                                        # Handle simple text document
                                        st.markdown("**📄 Content:**")
                                        # Use text area for better formatting of long text
                                        st.text_area(
                                            f"Document {i} Content",
                                            str(doc),
                                            height=150,
                                            disabled=True,
                                            key=f"doc_text_{i}",
                                        )

                                # Add separator between documents (except for the last one)
                                if i < len(st.session_state.last_result["documents"]):
                                    st.divider()

                if (
                    isinstance(st.session_state.last_result, dict)
                    and "rewrites" in st.session_state.last_result
                    and st.session_state.last_result["rewrites"]
                ):
                    with st.expander(
                        f"🔄 Query Rewrites ({len(st.session_state.last_result['rewrites'])} variations)",
                        expanded=False,
                    ):
                        st.markdown("**🎯 Original Query:**")
                        st.info(f"_{st.session_state.last_query}_")

                        st.markdown("**🔄 AI-Generated Variations:**")
                        st.markdown(
                            "*These variations help find more relevant papers by exploring different ways to express your question.*"
                        )

                        # Display rewrites in a more structured way
                        for i, rewrite in enumerate(
                            st.session_state.last_result["rewrites"], 1
                        ):
                            with st.container():
                                col_num, col_rewrite = st.columns([0.1, 0.9])
                                with col_num:
                                    st.markdown(f"**{i}.**")
                                with col_rewrite:
                                    # Use a subtle background for each rewrite
                                    st.markdown(
                                        f"""
                                    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 5px;">
                                        <em>{rewrite}</em>
                                    </div>
                                    """,
                                        unsafe_allow_html=True,
                                    )


if __name__ == "__main__":
    main()
