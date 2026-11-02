"""
ui_utils.py
-----------
Small shared helpers for consistent page config and styling across
every page in the Streamlit multipage app.
"""

import streamlit as st


def apply_page_style():
    st.markdown(
        """
        <style>
        /* --- Remove icons in front of sidebar navigation menu items --- */
        [data-testid="stSidebarNavLinkIcon"] {
            display: none !important;
        }
        [data-testid="stSidebarNav"] a svg {
            display: none !important;
        }
        [data-testid="stSidebarNav"] a span:first-child:not(:last-child) {
            display: none !important;
        }

        /* --- Global & Typography Polish --- */
        .metric-card {
            background: linear-gradient(135deg, #f8fbf8 0%, #edf7ee 100%);
            border: 1px solid #d0e7d2;
            border-radius: 12px;
            padding: 1.1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            transition: all 0.25s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(46, 125, 50, 0.12);
            border-color: #81c784;
        }

        /* --- Interactive Module Cards --- */
        .module-card {
            background: #ffffff;
            border: 1px solid #e2ece2;
            border-radius: 12px;
            padding: 1.25rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        }
        .module-card:hover {
            transform: translateY(-3px);
            border-color: #4caf50;
            box-shadow: 0 10px 24px rgba(46, 125, 50, 0.12);
        }

        /* --- Status & Tag Badges --- */
        .badge-pill {
            display: inline-block;
            padding: 3px 10px;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 20px;
            letter-spacing: 0.3px;
            text-transform: uppercase;
        }
        .badge-ready {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #c8e6c9;
        }
        .badge-agent {
            background-color: #ede7f6;
            color: #512da8;
            border: 1px solid #d1c4e9;
        }
        .badge-vision {
            background-color: #e0f2f1;
            color: #00695c;
            border: 1px solid #b2dfdb;
        }

        /* --- Buttons & Inputs Polish --- */
        .stButton>button {
            border-radius: 9px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }

        /* --- Sidebar nav panel --- */
        [data-testid="stSidebarNav"] {
            padding-top: 0.5rem;
        }
        [data-testid="stSidebarNav"] ul {
            padding: 0 0.5rem;
        }
        [data-testid="stSidebarNav"] li {
            margin-bottom: 5px;
        }
        [data-testid="stSidebarNav"] a {
            border-radius: 10px;
            padding: 9px 16px;
            font-size: 0.93rem;
            font-weight: 500;
            color: #2c3e50;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
        }
        [data-testid="stSidebarNav"] a:hover {
            background-color: rgba(76, 175, 80, 0.12);
            color: #2e7d32;
            transform: translateX(3px);
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: linear-gradient(90deg, rgba(76, 175, 80, 0.22) 0%, rgba(76, 175, 80, 0.08) 100%);
            border-left: 3.5px solid #2e7d32;
            color: #1b5e20;
            font-weight: 700;
        }

        /* --- Interactive Workflow Steps --- */
        .workflow-step {
            background: #ffffff;
            border: 1px solid #d8ebd8;
            border-radius: 10px;
            padding: 12px 14px;
            text-align: center;
            font-size: 0.85rem;
            font-weight: 600;
            color: #2e7d32;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: all 0.2s ease;
        }
        .workflow-step:hover {
            background: #f1f8f1;
            transform: scale(1.02);
            border-color: #4caf50;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.divider()
def web_search_box(key_prefix: str, default_query: str = ""):
    """
    Drop-in 'Direct Web Search' box. Call with a unique key_prefix per
    page (e.g. the page name) so widget keys don't collide.
    """
    from utils.rag_utils import ask_with_web_search

    with st.expander("🌐 Direct Web Search (live internet, not your papers)"):
        query = st.text_input(
            "Search query", value=default_query, key=f"{key_prefix}_web_query"
        )
        if st.button("Search Web", key=f"{key_prefix}_web_btn", disabled=not query.strip()):
            with st.spinner("Searching the web and summarizing with Llama 3.3 70B..."):
                try:
                    result = ask_with_web_search(query)
                    st.write(result["answer"])
                    if result["results"]:
                        with st.expander("Sources"):
                            for r in result["results"]:
                                st.markdown(f"- [{r.get('title', 'link')}]({r.get('href', '#')})")
                except Exception as e:
                    st.error(f"Web search failed: {e}")    