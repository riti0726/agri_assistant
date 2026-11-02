import pandas as pd
import plotly.express as px
import streamlit as st
from utils.ui_utils import apply_page_style, page_header, web_search_box
from utils.rag_utils import get_vectorstore, get_document_count

st.set_page_config(page_title="Research Analytics | Agri Research AI", layout="wide")
apply_page_style()
page_header("Research Analytics", "Quantitative statistics, chunk distributions, and page coverage across your indexed research library.")
web_search_box("research_analytics")

total_chunks = get_document_count()

if total_chunks == 0:
    st.info("No research papers indexed yet. Below is a preview of the interactive analytics generated once your papers are processed:")
    
    # Interactive mock demonstration
    mock_data = pd.DataFrame([
        {"source": "wheat_drought_resilience_2024.pdf", "page": 1, "chunks": 24},
        {"source": "wheat_drought_resilience_2024.pdf", "page": 8, "chunks": 24},
        {"source": "rice_nitrogen_efficiency_study.pdf", "page": 3, "chunks": 38},
        {"source": "rice_nitrogen_efficiency_study.pdf", "page": 12, "chunks": 38},
        {"source": "soil_microbiome_conservation_ag.pdf", "page": 5, "chunks": 19},
    ])
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fig_mock = px.bar(
            mock_data.drop_duplicates(subset=["source"]), 
            x="source", y="chunks", color="source", 
            title="Sample Preview: Chunks per Research Paper"
        )
        fig_mock.update_layout(showlegend=False)
        st.plotly_chart(fig_mock, use_container_width=True)
    with col_m2:
        st.markdown("#### Ready to analyze your own PDFs?")
        st.write("Upload your agricultural publications, dissertations, and extension bulletins to generate live analytics.")
        st.page_link("pages/1_Research_Assistant.py", label="Upload Research Papers in Research Assistant")
else:
    vectorstore = get_vectorstore()
    data = vectorstore._collection.get(include=["metadatas", "documents"])
    metadatas = data.get("metadatas", []) or []
    documents = data.get("documents", []) or []

    # Clean DataFrame construction
    rows = []
    for idx, m in enumerate(metadatas):
        row = m.copy() if isinstance(m, dict) else {}
        if idx < len(documents):
            row["doc_len"] = len(documents[idx]) if documents[idx] else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "page" not in df.columns:
        df["page"] = 1
    else:
        df["page"] = pd.to_numeric(df["page"], errors="coerce").fillna(1).astype(int)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Indexed Research Papers", df["source"].nunique())
    col2.metric("Total Extracted Chunks", len(df))
    col3.metric("Avg Chunks / Paper", round(len(df) / max(df["source"].nunique(), 1), 1))
    avg_chars = round(df["doc_len"].mean(), 0) if "doc_len" in df.columns else 1000
    col4.metric("Avg Chunk Character Length", f"{int(avg_chars)} chars")

    st.divider()

    # Interactive paper filter
    all_sources = ["All Papers"] + sorted(df["source"].unique().tolist())
    selected_filter = st.selectbox("Interactive Filter: Select specific paper", all_sources)
    filtered_df = df if selected_filter == "All Papers" else df[df["source"] == selected_filter]

    st.write("")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.subheader("Chunk Volume per Research Document")
        chunk_counts = filtered_df.groupby("source").size().reset_index(name="chunks")
        fig1 = px.bar(
            chunk_counts, 
            x="source", 
            y="chunks", 
            color="source", 
            title="Extracted chunks per research paper",
            labels={"chunks": "Number of Chunks", "source": "Paper Filename"}
        )
        fig1.update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(fig1, use_container_width=True)

    with col_c2:
        st.subheader("Page Coverage & Reference Spread")
        fig2 = px.box(
            filtered_df, 
            x="source", 
            y="page", 
            color="source",
            title="Distribution of page citations across documents",
            labels={"page": "Page Number", "source": "Paper Filename"}
        )
        fig2.update_layout(showlegend=False, xaxis_tickangle=-25)
        st.plotly_chart(fig2, use_container_width=True)

    if "doc_len" in filtered_df.columns:
        st.write("")
        st.subheader("Chunk Size Distribution (Character Count)")
        fig3 = px.histogram(
            filtered_df,
            x="doc_len",
            nbins=25,
            title="Histogram of chunk text length across indexed collection",
            labels={"doc_len": "Chunk Length (characters)"},
            color_discrete_sequence=["#4CAF50"]
        )
        st.plotly_chart(fig3, use_container_width=True)
