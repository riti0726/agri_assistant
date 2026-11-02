from datetime import date
import streamlit as st
from utils.ui_utils import apply_page_style, page_header, web_search_box
from utils.rag_utils import ask_question, list_ingested_sources

st.set_page_config(page_title="Report Generator | Agri Research AI", layout="wide")
apply_page_style()
page_header("Report Generator", "Synthesize AI-written literature reviews and executive summaries from indexed research papers.")
web_search_box("report_generator")

if "report_count" not in st.session_state:
    st.session_state.report_count = 0
if "last_report_type" not in st.session_state:
    st.session_state.last_report_type = "Executive Summary"

sources = list_ingested_sources()

if not sources:
    st.info("No research papers are indexed in ChromaDB yet. Upload papers in **Research Assistant** first.")
    st.page_link("pages/1_Research_Assistant.py", label="Open Research Assistant to Upload Papers")
else:
    # ---------------------------------------------------------------------------
    # Interactive Report Templates
    # ---------------------------------------------------------------------------
    st.markdown("#### Interactive Report Presets")
    st.caption("Select a standard agronomic inquiry template to auto-populate the generator:")

    templates = [
        ("Drought Stress Impact", "Mechanisms of drought tolerance, physiological impacts, and water efficiency practices.", "Literature Review"),
        ("Nutrient Management", "Nitrogen and phosphorus utilization efficiency, soil health, and yield correlation.", "Executive Summary"),
        ("Pest & Disease Resistance", "Host plant resistance genetics, integrated pest management, and biological control.", "Key Findings Brief"),
        ("Conservation Tillage", "Soil organic matter retention, microbial activity, and erosion prevention under zero-till.", "Literature Review"),
    ]

    t_cols = st.columns(len(templates))
    for t_idx, (t_title, t_desc, t_type) in enumerate(templates):
        with t_cols[t_idx]:
            with st.container(border=True):
                st.markdown(f"**{t_title}**")
                st.caption(t_desc)
                if st.button("Use Template", key=f"tpl_btn_{t_idx}", use_container_width=True):
                    st.session_state["report_topic_input"] = f"{t_title}: {t_desc}"
                    st.session_state["report_type_input"] = t_type

    st.write("")
    st.subheader("Report Configuration")
    
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        topic = st.text_input(
            "Report topic / focus", 
            value=st.session_state.get("report_topic_input", ""),
            placeholder="e.g. Assessment of biochar application on nitrogen retention in sandy loam soils",
        )
    with col_c2:
        report_type_options = ["Executive Summary", "Literature Review", "Key Findings Brief", "Policy & Field Recommendation"]
        curr_type = st.session_state.get("report_type_input", "Executive Summary")
        type_idx = report_type_options.index(curr_type) if curr_type in report_type_options else 0
        report_type = st.selectbox("Report type", report_type_options, index=type_idx)

    selected_sources = st.multiselect(
        "Filter to specific papers (optional - leave blank to synthesize across entire library)", 
        sources
    )

    if st.button("Generate Agricultural Research Report", type="primary", disabled=not topic.strip(), use_container_width=True):
        scope = f" Focus specifically on evidence from these papers: {', '.join(selected_sources)}." if selected_sources else ""
        prompt = (
            f"Write a comprehensive, professional {report_type.lower()} on the topic '{topic}' "
            f"grounded in the provided agricultural research context.{scope} "
            f"Structure the report with an Executive Overview, Key Experimental Findings, "
            f"Agronomic Implications, and Recommendations for Farmers and Researchers. "
            f"Cite specific findings where available."
        )
        with st.spinner("Synthesizing multi-document report with Llama 3.3 70B..."):
            try:
                result = ask_question(prompt, k=6)
                st.session_state.report_count += 1
                st.session_state["last_report"] = result["answer"]
                st.session_state["last_report_sources"] = result["sources"]
                st.session_state["last_report_type"] = report_type
                st.session_state["last_report_topic"] = topic
            except EnvironmentError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Report generation failed: {e}")

    # Display generated report
    if st.session_state.get("last_report"):
        st.divider()
        rep_type_title = st.session_state.get("last_report_type", "Research Report")
        rep_topic_title = st.session_state.get("last_report_topic", topic)
        
        st.markdown(f"### {rep_type_title}: {rep_topic_title}")
        st.caption(f"Synthesized on {date.today().strftime('%B %d, %Y')} · Powered by Llama 3.3 70B")
        
        with st.container(border=True):
            st.markdown(st.session_state["last_report"])

        col_dl, col_space = st.columns([1, 3])
        with col_dl:
            st.download_button(
                "Download Report (.md)",
                data=st.session_state["last_report"],
                file_name=f"agri_research_report_{date.today().isoformat()}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        sources_used = st.session_state.get("last_report_sources", [])
        if sources_used:
            with st.expander(f"Grounded Context & Citations ({len(sources_used)})"):
                for src in sources_used:
                    st.markdown(f"- **{src['source']}** (Page {src['page']})")
                    st.caption(src["text"][:350] + "...")
