import streamlit as st
from utils.ui_utils import apply_page_style, page_header, web_search_box
from utils.rag_utils import (
    process_uploaded_pdfs,
    ask_question,
    get_document_count,
    list_ingested_sources,
)

st.set_page_config(page_title="Research Assistant | Agri Research AI", layout="wide")
apply_page_style()
page_header("Research Assistant", "Upload agricultural research papers and ask AI-generated questions grounded in your documents.")
web_search_box("research_assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

left, right = st.columns([1, 2])

# ---------------------------------------------------------------------------
# Left column: upload + index
# ---------------------------------------------------------------------------
with left:
    st.subheader("Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Upload one or more research PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process Research Papers", type="primary", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Extracting text, chunking, and embedding into ChromaDB..."):
            try:
                summary = process_uploaded_pdfs(uploaded_files)
                for fname, count in summary.items():
                    st.success(f"{fname}: {count} chunks added")
            except Exception as e:
                st.error(f"Failed to process PDFs: {e}")

    st.divider()
    st.metric("Indexed Chunks in ChromaDB", get_document_count())

    sources = list_ingested_sources()
    if sources:
        st.write("**Currently Indexed Papers:**")
        for s in sources:
            st.markdown(f"- `{s}`")
    else:
        st.caption("No papers loaded yet.")

# ---------------------------------------------------------------------------
# Right column: Q&A + Interactive Quick Prompts
# ---------------------------------------------------------------------------
with right:
    st.subheader("Ask a Research Question")

    if get_document_count() == 0:
        st.warning("No papers indexed yet. Upload and process a PDF on the left first.")

    # Interactive sample chips
    st.markdown("**Quick Sample Queries:**")
    quick_samples = [
        "What specific fertilizer treatments yielded highest crop gains?",
        "What disease resistance mechanisms or genes were identified?",
        "What irrigation intervals and water use efficiencies are reported?",
    ]
    s_cols = st.columns(len(quick_samples))
    for idx, s_text in enumerate(quick_samples):
        with s_cols[idx]:
            if st.button(s_text, key=f"rag_sample_{idx}", use_container_width=True):
                st.session_state["rag_question_input"] = s_text

    question = st.text_area(
        "Your question",
        value=st.session_state.get("rag_question_input", ""),
        placeholder="e.g. What nitrogen application rates improved wheat yield in the study?",
        height=95,
    )

    col_slider, col_actions = st.columns([2, 1])
    with col_slider:
        top_k = st.slider("Number of relevant chunks to retrieve", min_value=2, max_value=8, value=4)
    with col_actions:
        st.write("")
        st.write("")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            ask_clicked = st.button("Ask AI", type="primary", disabled=not question.strip(), use_container_width=True)
        with col_b2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state["rag_question_input"] = ""
                st.rerun()

    if ask_clicked:
        with st.spinner("Searching ChromaDB vectors and synthesizing answer with Llama 3.3 70B..."):
            try:
                result = ask_question(question, k=top_k)
                st.session_state.chat_history.append(
                    {"question": question, "answer": result["answer"], "sources": result["sources"]}
                )
                st.session_state.query_count += 1
            except EnvironmentError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    st.divider()

    for entry in reversed(st.session_state.chat_history):
        with st.container(border=True):
            st.markdown(f"**Inquiry:** {entry['question']}")
            st.markdown("##### Grounded Findings:")
            st.write(entry["answer"])
            if entry.get("sources"):
                with st.expander(f"Context Excerpts & Citations ({len(entry['sources'])})"):
                    for src in entry["sources"]:
                        st.markdown(f"**{src['source']}** — Page {src['page']}")
                        st.caption(src["text"] + "...")
