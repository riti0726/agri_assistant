import streamlit as st
from utils.ui_utils import apply_page_style, page_header, web_search_box
from utils.rag_utils import list_ingested_sources, get_document_count, get_vectorstore

st.set_page_config(page_title="Research Library | Agri Research AI", layout="wide")
apply_page_style()
page_header("Research Library", "Manage, search, and inspect agricultural research literature stored in ChromaDB.")
web_search_box("research_library")

sources = list_ingested_sources()
chunk_count = get_document_count()

col1, col2, col3 = st.columns(3)
col1.metric("Ingested Documents", len(sources))
col2.metric("Total Indexed Chunks", chunk_count)
col3.metric("Vector Storage Status", "Active" if chunk_count > 0 else "Empty")

st.divider()

if not sources:
    st.info("Your research library is currently empty. Upload research papers to begin semantic indexing.")
    st.page_link("pages/1_Research_Assistant.py", label="Upload Papers in Research Assistant")
else:
    # ---------------------------------------------------------------------------
    # Interactive Search & Keyword Quick-Filters
    # ---------------------------------------------------------------------------
    st.markdown("#### Interactive Keyword Search")
    st.caption("Click a common agronomic keyword to quickly explore matching excerpts:")

    quick_tags = ["Yield", "Nitrogen", "Irrigation", "Drought", "Pest", "Phosphorus", "Soil organic matter"]
    t_cols = st.columns(len(quick_tags))
    for idx, tag in enumerate(quick_tags):
        with t_cols[idx]:
            if st.button(tag, key=f"lib_tag_{idx}", use_container_width=True):
                st.session_state["lib_search_term"] = tag

    search_term = st.text_input(
        "Search library by semantic keyword or phrase", 
        value=st.session_state.get("lib_search_term", ""),
        placeholder="e.g. Grain protein concentration or water use efficiency",
    )

    k_val = st.slider("Max search results to retrieve", min_value=3, max_value=12, value=5)

    if search_term.strip():
        with st.spinner("Executing similarity search across ChromaDB embeddings..."):
            try:
                vectorstore = get_vectorstore()
                results = vectorstore.similarity_search(search_term, k=k_val)
                if results:
                    st.write(f"**Top {len(results)} matches for “{search_term}”:**")
                    for doc in results:
                        with st.container(border=True):
                            src_name = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
                            pg_num = doc.metadata.get("page", "?") if doc.metadata else "?"
                            st.markdown(f"**Document:** `{src_name}` — **Page:** `{pg_num}`")
                            st.write(doc.page_content)
                else:
                    st.warning("No matches found for that query.")
            except Exception as e:
                st.error(f"Search failed: {e}")
    else:
        st.write("#### Indexed Papers Overview")
        for s in sources:
            with st.container(border=True):
                col_n, col_act = st.columns([4, 1])
                with col_n:
                    st.markdown(f"**{s}**")
                    st.caption("Indexed and active in persistent ChromaDB vector collection.")
                with col_act:
                    if st.button("Search this file", key=f"btn_search_{s}"):
                        st.session_state["lib_search_term"] = s.replace(".pdf", "")
                        st.rerun()
