"""
Reusable UI components for the Streamlit application.
"""

import streamlit as st


def render_sidebar(system_status):
    """Render the system status sidebar."""
    with st.sidebar:
        st.header("System Status")

        st.subheader("Dependencies")
        for dep, status in system_status["dependencies"].items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {dep}")

        st.subheader("Environment Variables")
        for var, status in system_status["env_status"].items():
            icon = "✅" if status else "❌"
            st.write(f"{icon} {var}")

        st.subheader("Data Files")
        if system_status["data_exists"]:
            st.success("✅ Processed data available")
        else:
            st.warning("⚠️ No processed data found")


def render_header():
    """Render the main application header."""
    st.title("🔍 Agentic RAG Article Search System")


def handle_progress_error(progress_bar, status_text, error_msg):
    """Consolidated error handling for progress tracking."""
    progress_bar.progress(100)
    status_text.text("❌ Query processing failed")
    return {"error": f"Query execution failed: {error_msg}"}
