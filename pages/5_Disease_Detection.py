"""
5_🦠_Disease_Detection.py
----------------------------
Crop/leaf image upload for disease identification.

Uses a pretrained Hugging Face image-classification model
(utils/vision_utils.py) to automatically predict the likely disease from
an uploaded leaf image, then optionally hands that prediction (plus any
manually described symptoms) to the RAG pipeline for research-based
management guidance.
"""

import streamlit as st
from utils.ui_utils import apply_page_style, page_header, web_search_box
from utils.rag_utils import ask_question
from utils.vision_utils import classify_leaf_image

st.set_page_config(page_title="Disease Detection | Agri Research AI", layout="wide")
apply_page_style()
page_header("Disease Detection", "Upload a crop or leaf image to identify possible diseases.")
web_search_box("disease_detection")

uploaded_image = st.file_uploader("Upload a leaf/crop image", type=["jpg", "jpeg", "png"])

detected_label = None

if uploaded_image:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

    with col2:
        st.subheader("Analysis")

        with st.spinner("Running disease classification model..."):
            try:
                predictions = classify_leaf_image(uploaded_image)
            except Exception as e:
                predictions = []
                st.error(f"Classification failed: {e}")

        if predictions:
            st.markdown("**Model predictions:**")
            for p in predictions:
                st.write(f"- {p['label'].replace('_', ' ')} — {p['score'] * 100:.1f}% confidence")
            detected_label = predictions[0]["label"].replace("_", " ")
            st.success(f"Most likely: **{detected_label}**")

        st.divider()

        symptom_desc = st.text_area(
            "Describe the visible symptoms (spots, discoloration, wilting, etc.) "
            "— optional if a model prediction was found above",
            placeholder="e.g. Yellow-brown circular spots with dark borders on lower leaves",
        )
        crop_name = st.text_input("Crop", placeholder="e.g. Tomato")

        if st.button(
            "Get Research-Based Guidance",
            type="primary",
            disabled=not (symptom_desc.strip() or detected_label),
        ):
            if detected_label:
                q = (
                    f"The image was classified as '{detected_label}' for "
                    f"{crop_name or 'this crop'}. Confirm this diagnosis and "
                    f"suggest management practices."
                )
                if symptom_desc.strip():
                    q += f" Additional observed symptoms: {symptom_desc}"
            else:
                q = (
                    f"For {crop_name or 'this crop'}, what disease could cause "
                    f"these symptoms: {symptom_desc}? Suggest likely causes and "
                    f"management practices."
                )

            with st.spinner("Searching indexed research..."):
                try:
                    result = ask_question(q)
                    st.write(result["answer"])
                    if result["sources"]:
                        with st.expander("Sources"):
                            for src in result["sources"]:
                                st.markdown(f"**{src['source']}** — page {src['page']}")
                except EnvironmentError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
else:
    st.info("Upload an image above, or explore the interactive symptom diagnostics below.")

st.divider()
st.subheader("Interactive Symptom Simulator & Quick Diagnosis")
st.caption("Don't have an image on hand? Click any sample symptom pattern to simulate research guidance:")

sample_diseases = [
    {
        "crop": "Tomato",
        "disease": "Early Blight (Alternaria solani)",
        "symptoms": "Concentric dark brown rings ('target' spots) on lower leaves, yellow halo surrounding lesions, premature leaf drop.",
    },
    {
        "crop": "Rice",
        "disease": "Bacterial Leaf Blight (Xanthomonas oryzae)",
        "symptoms": "Water-soaked to yellowish-white stripes along leaf margins, wilting seedlings ('kresek' phase).",
    },
    {
        "crop": "Wheat",
        "disease": "Stripe Rust (Puccinia striiformis)",
        "symptoms": "Yellow-orange pustules arranged in linear stripes on leaf blades, reduced grain fill and shriveled kernels.",
    },
    {
        "crop": "Maize",
        "disease": "Northern Corn Leaf Blight (Exserohilum turcicum)",
        "symptoms": "Long, elliptical grayish-green or tan lesions measuring 1 to 6 inches on leaves.",
    },
]

sim_cols = st.columns(len(sample_diseases))
for i, item in enumerate(sample_diseases):
    with sim_cols[i]:
        with st.container(border=True):
            st.markdown(f"**{item['crop']}**")
            st.caption(f"*{item['disease']}*")
            if st.button("Simulate Diagnosis", key=f"sim_dis_{i}", use_container_width=True):
                st.session_state["active_sim"] = item

if "active_sim" in st.session_state:
    active = st.session_state["active_sim"]
    with st.container(border=True):
        st.markdown(f"### Research Guidance for {active['crop']}: {active['disease']}")
        st.markdown(f"**Simulated Symptoms:** {active['symptoms']}")
        
        sim_q = f"For {active['crop']}, regarding {active['disease']} with symptoms: {active['symptoms']}, what are the recommended cultural, biological, and chemical management strategies according to agricultural research?"
        with st.spinner("Retrieving agronomic research insights..."):
            try:
                res = ask_question(sim_q)
                st.write(res["answer"])
                if res.get("sources"):
                    with st.expander("Cited Sources"):
                        for src in res["sources"]:
                            st.markdown(f"- **{src['source']}** (Page {src['page']})")
            except Exception as e:
                st.error(f"Guidance retrieval failed: {e}")