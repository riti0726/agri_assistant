import os
from datetime import datetime
import streamlit as st
from utils.ui_utils import apply_page_style, page_header
from utils.rag_utils import get_document_count, list_ingested_sources, ask_question

st.set_page_config(
    page_title="Agri Research AI - Dashboard",
    layout="wide",
)

apply_page_style()

# ---------------------------------------------------------------------------
# Sidebar (clean menu without icons in front)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Agri Research AI")
    st.caption("Agricultural Intelligence & Research Platform")
    st.divider()
    
    st.markdown("**Quick Navigation**")
    st.caption("Use the menu above to switch modules.")
    
    st.divider()
    st.markdown("**Platform Status**")
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    if groq_configured:
        st.success("Groq API: Connected")
    else:
        st.error("Groq API: Missing Key")
    
    doc_count = get_document_count()
    st.info(f"Vector Store: {doc_count} chunks")

page_header(
    "Agri Research AI",
    "AI-powered agricultural research platform — RAG · Llama 3.3 70B · ChromaDB · Vision AI",
)

# ---------------------------------------------------------------------------
# Top-level metrics with interactive Refresh
# ---------------------------------------------------------------------------
sources = list_ingested_sources()
chunk_count = get_document_count()

if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "report_count" not in st.session_state:
    st.session_state.report_count = 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Research Papers", len(sources), delta=f"{len(sources)} active")
with col2:
    st.metric("Indexed Chunks", chunk_count, delta=f"{chunk_count} vectors")
with col3:
    st.metric("Research Queries", st.session_state.query_count)
with col4:
    st.metric("Reports Generated", st.session_state.report_count)

st.write("")

# ---------------------------------------------------------------------------
# Interactive Quick Research Console
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("#### Interactive Research Console")
    st.caption("Ask questions directly against your indexed papers and knowledge base from the dashboard.")
    
    # Quick prompt chips
    sample_queries = [
        "What are optimal nitrogen application rates for wheat?",
        "How does drought stress affect maize yield?",
        "Management practices for leaf blight in rice",
        "Methods to improve soil organic carbon in semi-arid zones"
    ]
    
    st.markdown("**Quick Prompts (click to load):**")
    chip_cols = st.columns(len(sample_queries))
    for idx, prompt_text in enumerate(sample_queries):
        with chip_cols[idx]:
            if st.button(prompt_text, key=f"quick_prompt_{idx}", use_container_width=True):
                st.session_state["dashboard_query_input"] = prompt_text
    
    user_query = st.text_input(
        "Enter your agricultural research question",
        value=st.session_state.get("dashboard_query_input", ""),
        placeholder="e.g. What fertilizer management strategies maximize crop yield while minimizing runoff?",
        key="dash_search_box",
    )
    
    col_ask, col_clear, col_status = st.columns([1, 1, 4])
    with col_ask:
        ask_btn = st.button("Ask AI", type="primary", use_container_width=True, disabled=not user_query.strip())
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state["dashboard_query_input"] = ""
            st.rerun()
            
    if ask_btn and user_query.strip():
        with st.spinner("Analyzing papers and generating grounded response with Llama 3.3 70B..."):
            try:
                res = ask_question(user_query.strip())
                st.session_state.query_count += 1
                with st.container(border=True):
                    st.markdown("##### Research Findings")
                    st.write(res["answer"])
                    if res.get("sources"):
                        with st.expander(f"Grounded Sources ({len(res['sources'])})"):
                            for s in res["sources"]:
                                st.markdown(f"- **{s['source']}** (Page {s['page']})")
                                st.caption(s["text"][:300] + "...")
            except Exception as e:
                st.error(f"Query error: {e}")

st.write("")

# ---------------------------------------------------------------------------
# Interactive Navigation Menu & Module Cards (No icons in front of menu items)
# ---------------------------------------------------------------------------
st.markdown("#### Dashboard Menu & Modules")
st.caption("Select a category or launch any research module below:")

modules = [
    {
        "category": "AI Assistants",
        "name": "Research Assistant",
        "desc": "Upload research papers and ask AI-powered questions grounded in your documents.",
        "path": "pages/1_Research_Assistant.py",
        "badge": "Core RAG",
    },
    {
        "category": "AI Assistants",
        "name": "Agent Assistant",
        "desc": "Autonomous agent chaining ChromaDB retrieval, live web search, and reasoning.",
        "path": "pages/9_Agent_Assistant.py",
        "badge": "LangGraph Agent",
    },
    {
        "category": "Crop & Soil Intel",
        "name": "Crop Research",
        "desc": "Crop-specific agronomic practices, yield optimization, and pest management.",
        "path": "pages/2_Crop_Research.py",
        "badge": "Agronomy",
    },
    {
        "category": "Crop & Soil Intel",
        "name": "Soil Analysis",
        "desc": "Interactive soil suitability assessment across pH, N, P, and K parameters.",
        "path": "pages/4_Soil_Analysis.py",
        "badge": "Interactive",
    },
    {
        "category": "Crop & Soil Intel",
        "name": "Disease Detection",
        "desc": "Leaf and crop image disease identification using MobileNetV2 computer vision.",
        "path": "pages/5_Disease_Detection.py",
        "badge": "Vision AI",
    },
    {
        "category": "Crop & Soil Intel",
        "name": "Weather & Climate",
        "desc": "Weather-aware agricultural insights and microclimate crop recommendations.",
        "path": "pages/6_Weather_Climate.py",
        "badge": "Climate",
    },
    {
        "category": "Data & Reports",
        "name": "Research Library",
        "desc": "Manage, search, and inspect all ingested PDF documents in ChromaDB.",
        "path": "pages/3_Research_Library.py",
        "badge": "Vector Store",
    },
    {
        "category": "Data & Reports",
        "name": "Research Analytics",
        "desc": "Statistical distribution, chunk metrics, and page coverage analytics.",
        "path": "pages/7_Research_Analytics.py",
        "badge": "Analytics",
    },
    {
        "category": "Data & Reports",
        "name": "Report Generator",
        "desc": "Automated AI synthesis of comprehensive literature review and executive reports.",
        "path": "pages/8_Report_Generator.py",
        "badge": "Synthesis",
    },
]

# Interactive Filter Tabs
tab_all, tab_ai, tab_crop, tab_data = st.tabs([
    f"All Modules ({len(modules)})",
    "AI Assistants (2)",
    "Crop & Soil Intel (4)",
    "Data & Reports (3)",
])

tab_filters = {
    0: None,
    1: "AI Assistants",
    2: "Crop & Soil Intel",
    3: "Data & Reports",
}

for tab_idx, tab_obj in enumerate([tab_all, tab_ai, tab_crop, tab_data]):
    with tab_obj:
        selected_category = tab_filters[tab_idx]
        filtered_modules = [
            m for m in modules if selected_category is None or m["category"] == selected_category
        ]
        
        # Grid display without any icons in front of titles or links
        cols = st.columns(3)
        for i, mod in enumerate(filtered_modules):
            with cols[i % 3]:
                with st.container(border=True):
                    # Clean title without icon in front
                    col_t, col_b = st.columns([3, 1])
                    with col_t:
                        st.markdown(f"### {mod['name']}")
                    with col_b:
                        st.markdown(f"<span class='badge-pill badge-ready'>{mod['badge']}</span>", unsafe_allow_html=True)
                    st.caption(mod["desc"])
                    st.write("")
                    # Clean page link with NO icon in front
                    st.page_link(mod["path"], label=f"Open {mod['name']}")

st.write("")

# ---------------------------------------------------------------------------
# Interactive Workflow Inspector
# ---------------------------------------------------------------------------
with st.expander("Interactive AI Pipeline & Architecture Inspector"):
    st.markdown("Click any stage below to inspect how data moves through the Agri Research AI pipeline:")
    
    stages = [
        ("PDF Ingestion", "Uploads are saved to `data/uploaded_pdfs/` and parsed using PyPDFLoader."),
        ("Text Chunking", "Documents split with RecursiveCharacterTextSplitter (1000 chars, 200 overlap)."),
        ("Vector Embeddings", "Transformed into 384-dimensional dense vectors with `all-MiniLM-L6-v2`."),
        ("ChromaDB Index", "Vectors stored with metadata (source, page) in persistent SQLite/Chroma collection."),
        ("Semantic Search", "Cosine similarity retrieval fetches top-k most relevant research excerpts."),
        ("Llama 3.3 70B", "Groq-hosted Llama 3.3 70B generates grounded answers strictly cited from context."),
    ]
    
    stage_cols = st.columns(len(stages))
    for s_idx, (s_name, s_desc) in enumerate(stages):
        with stage_cols[s_idx]:
            if st.button(s_name, key=f"stage_btn_{s_idx}", use_container_width=True):
                st.session_state["selected_stage_desc"] = (s_name, s_desc)
                
    if "selected_stage_desc" in st.session_state:
        name, desc = st.session_state["selected_stage_desc"]
        st.info(f"**{name}**: {desc}")

st.write("")

# ---------------------------------------------------------------------------
# Indexed Papers Library Preview & Quick Actions
# ---------------------------------------------------------------------------
st.markdown("#### Indexed Document Status")
if sources:
    with st.container(border=True):
        st.markdown(f"**Currently Indexed Research Papers ({len(sources)}):**")
        p_cols = st.columns(3)
        for idx, src in enumerate(sources):
            with p_cols[idx % 3]:
                st.markdown(f"- `{src}`")
        st.write("")
        col_lib, col_upload = st.columns([1, 4])
        with col_lib:
            st.page_link("pages/3_Research_Library.py", label="Explore in Research Library")
else:
    st.info("No research papers indexed yet. Upload your first PDF in **Research Assistant** to activate document grounding.")
    st.page_link("pages/1_Research_Assistant.py", label="Go to Research Assistant")
