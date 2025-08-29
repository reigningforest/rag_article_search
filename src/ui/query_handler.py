"""
Query processing and progress tracking for the Streamlit UI.
"""

import time
import streamlit as st
from .components import handle_progress_error


def query_rag_system(rag_graph, query):
    """Query the RAG system without progress tracking."""
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
            "selected_for_simplification": [],
            "continue_simplification": False,
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
            if step_name in progress_steps:
                progress, status = progress_steps[step_name]
                progress_bar.progress(progress)
                status_text.text(status)
                time.sleep(0.2)

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
            "progress_callback": update_progress,
            "selected_for_simplification": [],
            "continue_simplification": False,
        }

        # Execute the graph
        result = rag_graph.invoke(initial_state)

        # Show final completion
        final_step = result.get("last_completed_step", "")
        if final_step == "direct_answer":
            status_text.text("🤖 Generated direct response!")
        else:
            status_text.text("🤖 Generated intelligent response!")

        progress_bar.progress(100)
        return result

    except Exception as e:
        return handle_progress_error(progress_bar, status_text, str(e))


def handle_query_interface(config):
    """Handle the query interface and processing."""
    st.success("✅ System ready for queries!")

    # Initialize RAG system if not already done
    if "rag_graph" not in st.session_state:
        from .initialization import initialize_rag_system_with_progress
        st.session_state.rag_graph = initialize_rag_system_with_progress(config)

    if st.session_state.rag_graph is None:
        st.error("❌ Failed to initialize RAG system. Check your environment variables.")
        return

    # Query interface
    st.subheader("💬 Ask Questions")

    # Create a form to handle Enter key press
    with st.form(key="search_form", clear_on_submit=False):
        query = st.text_input(
            "Enter your question:",
            placeholder="What are the recent developments in transformer architectures?",
            key="query_input",
        )
        form_submit = st.form_submit_button("🔍 Search", type="primary")

    # Process search when form is submitted
    if form_submit:
        if not query or query.strip() == "":
            st.error("⚠️ Please enter a proper question before searching!")
        else:
            # Create progress tracking elements
            progress_container = st.empty()

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
