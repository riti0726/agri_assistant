import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.ui_utils import apply_page_style, page_header, web_search_box
from utils.rag_utils import ask_question

st.set_page_config(page_title="Soil Analysis | Agri Research AI", layout="wide")
apply_page_style()
page_header("Soil Analysis", "Enter soil test parameters or load presets to get a crop suitability assessment.")
web_search_box("soil_analysis")

# ---------------------------------------------------------------------------
# Interactive Presets
# ---------------------------------------------------------------------------
st.markdown("#### Interactive Soil Test Presets")
st.caption("Click a preset to instantly load realistic regional soil test parameters:")

presets = {
    "Fertile Alluvial Loam": {"ph": 6.8, "n": 140, "p": 45, "k": 120, "crop": "Wheat"},
    "Acidic Red Soil": {"ph": 5.2, "n": 90, "p": 25, "k": 70, "crop": "Rice"},
    "Alkaline Black Soil": {"ph": 8.1, "n": 110, "p": 35, "k": 130, "crop": "Cotton"},
    "Degraded Sandy Soil": {"ph": 6.2, "n": 50, "p": 15, "k": 40, "crop": "Maize"},
}

p_cols = st.columns(len(presets))
for idx, (p_name, p_vals) in enumerate(presets.items()):
    with p_cols[idx]:
        if st.button(p_name, key=f"preset_btn_{idx}", use_container_width=True):
            st.session_state["soil_ph"] = p_vals["ph"]
            st.session_state["soil_n"] = p_vals["n"]
            st.session_state["soil_p"] = p_vals["p"]
            st.session_state["soil_k"] = p_vals["k"]
            st.session_state["soil_crop"] = p_vals["crop"]

col1, col2 = st.columns([1, 1])

CROPS = ["Wheat", "Rice", "Maize", "Soybean", "Cotton"]
CROP_IDEAL = {
    "Wheat":   {"ph": (6.0, 7.5), "n": (100, 150), "p": (30, 60),  "k": (80, 150)},
    "Rice":    {"ph": (5.5, 6.5), "n": (100, 160), "p": (30, 60),  "k": (80, 150)},
    "Maize":   {"ph": (5.8, 7.0), "n": (120, 180), "p": (40, 70),  "k": (80, 160)},
    "Soybean": {"ph": (6.0, 7.0), "n": (20, 40),   "p": (40, 60),  "k": (80, 140)},
    "Cotton":  {"ph": (6.0, 7.5), "n": (100, 150), "p": (30, 60),  "k": (60, 120)},
}

with col1:
    st.subheader("Soil Parameters")
    
    current_crop = st.session_state.get("soil_crop", "Wheat")
    crop_idx = CROPS.index(current_crop) if current_crop in CROPS else 0
    crop = st.selectbox("Crop to assess suitability for", CROPS, index=crop_idx)
    
    ph = st.slider(
        "Soil pH", 
        3.5, 9.5, 
        value=float(st.session_state.get("soil_ph", 6.5)), 
        step=0.1
    )
    nitrogen = st.number_input(
        "Nitrogen (N) — kg/ha", 
        min_value=0, max_value=500, 
        value=int(st.session_state.get("soil_n", 120))
    )
    phosphorus = st.number_input(
        "Phosphorus (P) — kg/ha", 
        min_value=0, max_value=300, 
        value=int(st.session_state.get("soil_p", 40))
    )
    potassium = st.number_input(
        "Potassium (K) — kg/ha", 
        min_value=0, max_value=400, 
        value=int(st.session_state.get("soil_k", 100))
    )

def in_range(value, bounds):
    return bounds[0] <= value <= bounds[1]

with col2:
    st.subheader("Suitability & Diagnostics")
    ideal = CROP_IDEAL[crop]

    checks = {
        "pH": (in_range(ph, ideal["ph"]), f"Ideal: {ideal['ph'][0]}–{ideal['ph'][1]}", ph, ideal["ph"]),
        "Nitrogen": (in_range(nitrogen, ideal["n"]), f"Ideal: {ideal['n'][0]}–{ideal['n'][1]} kg/ha", nitrogen, ideal["n"]),
        "Phosphorus": (in_range(phosphorus, ideal["p"]), f"Ideal: {ideal['p'][0]}–{ideal['p'][1]} kg/ha", phosphorus, ideal["p"]),
        "Potassium": (in_range(potassium, ideal["k"]), f"Ideal: {ideal['k'][0]}–{ideal['k'][1]} kg/ha", potassium, ideal["k"]),
    }

    passed = sum(1 for ok, _, _, _ in checks.values() if ok)
    score = round(passed / len(checks) * 100)

    score_col, metric_col = st.columns(2)
    with score_col:
        st.metric(f"Suitability Score for {crop}", f"{score}%")
    with metric_col:
        if score >= 75:
            st.success("High Suitability - Optimal conditions")
        elif score >= 50:
            st.warning("Moderate Suitability - Needs balancing")
        else:
            st.error("Low Suitability - Substantial amendment required")

    st.progress(score / 100)

    for param, (ok, note, val, bounds) in checks.items():
        status_icon = "Optimal" if ok else ("Deficit" if val < bounds[0] else "Surplus")
        badge_style = "color: #2e7d32; font-weight:600;" if ok else "color: #c62828; font-weight:600;"
        st.markdown(f"**{param}**: Current = `{val}`, {note} &nbsp; <span style='{badge_style}'>[{status_icon}]</span>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Interactive Comparison Chart & AI Amendment Advisor
# ---------------------------------------------------------------------------
st.divider()
c_chart, c_advisor = st.columns([1, 1])

with c_chart:
    st.subheader("Nutrient Balance vs. Optimal Target")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Nitrogen", "Phosphorus", "Potassium"],
        y=[nitrogen, phosphorus, potassium],
        name="Current Test (kg/ha)",
        marker_color="#4CAF50"
    ))
    fig.add_trace(go.Bar(
        x=["Nitrogen", "Phosphorus", "Potassium"],
        y=[sum(ideal["n"])/2, sum(ideal["p"])/2, sum(ideal["k"])/2],
        name=f"Optimal Target for {crop}",
        marker_color="#81C784"
    ))
    fig.update_layout(
        barmode='group',
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with c_advisor:
    st.subheader("Interactive Research-Grounded Amendment Plan")
    st.caption("Generate an actionable fertilization and soil reclamation regimen with Llama 3.3 70B:")
    
    if st.button("Generate AI Amendment Regimen", type="primary", use_container_width=True):
        deficits = []
        if ph < ideal["ph"][0]:
            deficits.append(f"pH is acidic ({ph} vs ideal {ideal['ph'][0]}–{ideal['ph'][1]})")
        elif ph > ideal["ph"][1]:
            deficits.append(f"pH is alkaline ({ph} vs ideal {ideal['ph'][0]}–{ideal['ph'][1]})")
        if nitrogen < ideal["n"][0]:
            deficits.append(f"Nitrogen deficiency ({nitrogen} kg/ha vs ideal {ideal['n'][0]}–{ideal['n'][1]})")
        if phosphorus < ideal["p"][0]:
            deficits.append(f"Phosphorus deficiency ({phosphorus} kg/ha vs ideal {ideal['p'][0]}–{ideal['p'][1]})")
        if potassium < ideal["k"][0]:
            deficits.append(f"Potassium deficiency ({potassium} kg/ha vs ideal {ideal['k'][0]}–{ideal['k'][1]})")

        query = (
            f"Provide practical soil amendment and fertilizer recommendations for {crop}. "
            f"Current parameters: pH {ph}, N {nitrogen} kg/ha, P {phosphorus} kg/ha, K {potassium} kg/ha. "
            f"Issues detected: {', '.join(deficits) if deficits else 'Parameters are balanced'}. "
            f"Specify organic and mineral options (e.g. lime, gypsum, compost, NPK blends) and application timing."
        )
        with st.spinner("Synthesizing agronomic recommendations..."):
            try:
                res = ask_question(query)
                st.write(res["answer"])
            except Exception as e:
                st.error(f"Advisor error: {e}")

st.caption("Guidance only — always cross-reference with certified agricultural extension or soil testing laboratories.")
