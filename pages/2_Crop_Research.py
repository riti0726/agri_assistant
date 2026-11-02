import streamlit as st
from utils.ui_utils import apply_page_style, page_header, web_search_box
from utils.rag_utils import ask_question, ask_llm_directly, get_document_count

st.set_page_config(page_title="Crop Research | Agri Research AI", layout="wide")
apply_page_style()
page_header("Crop Research", "Crop-specific agronomic practices, physiology, and yield optimization grounded in your literature.")
web_search_box("crop_research")

CROPS = ["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Sugarcane", "Pulses", "Other"]

# Crop specific quick inquiry starters
CROP_SAMPLE_PROMPTS = {
    "Wheat": [
        "What are the best irrigation schedules to avoid terminal heat stress during grain filling?",
        "How do balanced N-P-K-S ratios affect gluten quality and grain yield?",
        "Which conservation agriculture practices best reduce crown rot and powdery mildew?"
    ],
    "Rice": [
        "What are water saving benefits of Alternate Wetting and Drying (AWD) compared to flooded paddies?",
        "Management strategies to mitigate methane emissions without sacrificing paddy yield.",
        "Optimal zinc and nitrogen application timings for basmati varieties."
    ],
    "Maize": [
        "What are critical nitrogen uptake periods between V6 and R1 growth stages?",
        "Practices to narrow the anthesis-silking interval during mid-season drought.",
        "Integrated pest management tactics for Fall Armyworm (Spodoptera frugiperda)."
    ],
    "Soybean": [
        "Does Bradyrhizobium seed inoculation reduce the need for starter nitrogen in nodulating beans?",
        "Management of sudden death syndrome and soybean cyst nematode.",
        "Optimal plant population density for high-yield narrow-row soybean."
    ],
    "Cotton": [
        "Defoliation timing and boll opening management practices.",
        "Management of pink bollworm resistance in transgenic Bt cotton.",
        "Potassium deficiency symptoms and foliar feeding during peak squaring."
    ],
}

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Target Crop & Focus")
    crop = st.selectbox("Select a crop", CROPS)
    if crop == "Other":
        crop = st.text_input("Enter crop name", value="Barley")

    focus_area = st.radio(
        "Agronomic Research Focus",
        ["General practices", "Yield optimization", "Pest & disease resistance", "Irrigation & water use", "Soil requirements"],
    )

    doc_cnt = get_document_count()
    if doc_cnt > 0:
        st.success(f"{doc_cnt} research chunks indexed in ChromaDB.")
    else:
        st.info("No documents currently indexed. General knowledge fallback is available.")

with col2:
    st.subheader(f"Inquire on {crop}")
    
    # Interactive quick prompt chips for selected crop
    prompts_for_crop = CROP_SAMPLE_PROMPTS.get(crop, [
        f"What are key yield limiting factors for {crop}?",
        f"What are optimum planting dates and seedling rates for {crop}?",
    ])
    
    st.markdown("**Suggested Research Inquiries:**")
    prompt_cols = st.columns(len(prompts_for_crop))
    for p_i, p_txt in enumerate(prompts_for_crop):
        with prompt_cols[p_i]:
            if st.button(p_txt, key=f"cp_prompt_{p_i}", use_container_width=True):
                st.session_state["crop_q_input"] = p_txt

    question = st.text_area(
        "Research Question",
        value=st.session_state.get("crop_q_input", ""),
        placeholder=f"e.g. What are the best {focus_area.lower()} practices for {crop}?",
        height=95,
    )

    col_toggle, col_btn = st.columns([2, 1])
    with col_toggle:
        use_general_knowledge = st.checkbox(
            "Answer from general agronomic knowledge (bypasses ChromaDB)",
            value=(doc_cnt == 0),
        )
    with col_btn:
        run_btn = st.button("Get Research Insight", type="primary", disabled=not (crop and question.strip()), use_container_width=True)

    if run_btn:
        scoped_question = f"Regarding {crop} and {focus_area.lower()}: {question}"
        with st.spinner("Synthesizing agronomic research..."):
            try:
                if use_general_knowledge:
                    answer = ask_llm_directly(scoped_question)
                    st.write(answer)
                    st.caption("Synthesized from Llama 3.3 70B general knowledge base.")
                else:
                    result = ask_question(scoped_question)
                    st.write(result["answer"])
                    if result.get("sources"):
                        with st.expander(f"Grounded Literature Sources ({len(result['sources'])})"):
                            for src in result["sources"]:
                                st.markdown(f"- **{src['source']}** (Page {src['page']})")
                                st.caption(src["text"][:300] + "...")
            except EnvironmentError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Something went wrong: {e}")