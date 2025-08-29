"""
Data pipeline management UI components.
"""

import streamlit as st
from pathlib import Path
from scripts.get_data_pipeline import main as run_pipeline_main  # Reuse existing function


def run_data_pipeline():
    """
    Run the data processing pipeline using existing function.
    Reuses scripts.get_data_pipeline.main()
    """
    try:
        run_pipeline_main()  # Use existing function
        return True
    except Exception as e:
        st.error(f"Data pipeline failed: {str(e)}")
        return False


def handle_data_pipeline(config):
    """Handle the data pipeline UI when no data exists."""
    st.warning("⚠️ No processed data found. Please run the data pipeline first.")

    st.subheader("Data Pipeline")
    st.info(
        "This will download ArXiv papers, process them, and create embeddings. "
        "This may take several hours and requires significant storage space."
    )

    if st.button("🚀 Run Data Pipeline", type="primary"):
        # Check for kaggle.json file
        kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json_path.exists():
            st.error(
                "Kaggle authentication required. Please place your kaggle.json file in ~/.kaggle/kaggle.json"
            )
        else:
            with st.spinner("Running data pipeline... This may take several hours."):
                success = run_data_pipeline()
                if success:
                    st.success("✅ Data pipeline completed successfully!")
                    st.rerun()

    # Refresh Data button
    if st.button("🔄 Refresh Data"):
        _render_refresh_warning(config)


def _render_refresh_warning(config):
    """Render the data refresh warning and confirmation."""
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
                with st.spinner("Refreshing data... This may take several hours."):
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
