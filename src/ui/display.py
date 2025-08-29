"""
Result display components for the Streamlit UI.
"""

import streamlit as st


def display_results():
    """Display query results if available."""
    if (
        "last_result" not in st.session_state
        or "last_query" not in st.session_state
        or not st.session_state.last_result
    ):
        return

    st.subheader("📝 Response")
    st.write(f"**Question:** {st.session_state.last_query}")

    response = (
        st.session_state.last_result.get("response", "No response generated")
        if isinstance(st.session_state.last_result, dict)
        else "No response generated"
    )
    st.write(response)

    # Show query rewrite first (above source documents)
    _display_query_rewrite()
    _display_source_documents()


def _display_query_rewrite():
    """Display enhanced query perspective if available."""
    if (
        isinstance(st.session_state.last_result, dict)
        and "rewrite" in st.session_state.last_result
        and st.session_state.last_result["rewrite"]
    ):
        with st.expander("🔄 Enhanced Query Perspective", expanded=False):
            st.markdown("**🎯 Original Query:**")
            st.info(f"_{st.session_state.last_query}_")

            st.markdown("**🔄 Enhanced Query Perspective:**")
            st.markdown(
                "*This enhanced perspective helps find more relevant papers by optimizing the search query.*"
            )

            rewrite = st.session_state.last_result["rewrite"]
            st.markdown(f"• {rewrite}")


def _display_source_documents():
    """Display source documents with tabs for original/simplified versions."""
    # Prefer simplified documents over regular documents
    display_documents = st.session_state.last_result.get("simplified_documents", [])
    if not display_documents:
        display_documents = st.session_state.last_result.get("documents", [])

    if not (isinstance(st.session_state.last_result, dict) and display_documents):
        return

    simplified_text = (
        " (Simplified)"
        if "simplified_documents" in st.session_state.last_result
        and st.session_state.last_result["simplified_documents"]
        else ""
    )

    with st.expander(
        f"📄 Source Documents{simplified_text} ({len(display_documents)} found)",
        expanded=False,
    ):
        for i, doc in enumerate(display_documents, 1):
            _render_document(doc, i)
            # Add separator between documents (except for the last one)
            if i < len(display_documents):
                st.divider()


def _render_document(doc, doc_number):
    """Render a single document with metadata and content tabs."""
    with st.container():
        st.markdown(f"### 📋 **Source {doc_number}**")

        # Display basic info first
        if isinstance(doc, dict):
            if "title" in doc:
                st.markdown(f"**📝 Title:** {doc.get('title', 'No title')}")
            if "date" in doc:
                st.markdown(f"**📅 Date:** {doc.get('date', 'No date')}")

        # Create tabs for different content views
        if isinstance(doc, dict) and "text" in doc:
            abstract_text = doc.get("text", "")
            _render_document_tabs(abstract_text, doc, doc_number)
        else:
            # Handle simple text document
            st.text_area(
                f"Document {doc_number} Content",
                str(doc),
                height=300,
                disabled=True,
                key=f"doc_text_{doc_number}",
            )


def _render_document_tabs(abstract_text, doc, doc_number):
    """Render document content in tabs (simplified/original/details)."""
    # Check if document has both original and simplified versions
    if "ORIGINAL TEXT:" in abstract_text and "SIMPLIFIED VERSION:" in abstract_text:
        parts = abstract_text.split("SIMPLIFIED VERSION:")
        original_text = parts[0].replace("ORIGINAL TEXT:", "").strip()
        simplified_text = (
            parts[1].strip() if len(parts) > 1 else "No simplified version available"
        )

        # Create tabs for both versions
        tab1, tab2, tab3 = st.tabs(["✨ Simplified", "📄 Original", "ℹ️ Details"])

        with tab1:
            st.text_area(
                "AI-Simplified Summary",
                simplified_text,
                height=300,
                disabled=True,
                key=f"doc_{doc_number}_simplified",
            )

        with tab2:
            st.text_area(
                "Original Abstract",
                original_text,
                height=300,
                disabled=True,
                key=f"doc_{doc_number}_original",
            )

        with tab3:
            if "metadata" in doc:
                st.json(doc.get("metadata", {}))
            else:
                st.write("No additional metadata available")
    else:
        # Single content version - show in content tab
        tab1, tab2 = st.tabs(["📄 Content", "ℹ️ Details"])

        with tab1:
            st.text_area(
                "Abstract/Content",
                abstract_text,
                height=300,
                disabled=True,
                key=f"doc_{doc_number}_content",
            )

        with tab2:
            if "metadata" in doc:
                st.json(doc.get("metadata", {}))
            else:
                st.write("No additional metadata available")
