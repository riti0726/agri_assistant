import streamlit as st
from utils.ui_utils import apply_page_style, page_header, web_search_box
from utils.rag_utils import ask_question

st.set_page_config(page_title="Weather & Climate | Agri Research AI", layout="wide")
apply_page_style()
page_header("Weather & Climate", "Weather-aware agricultural research insights and climate stress adaptations.")
web_search_box("weather_climate")

# ---------------------------------------------------------------------------
# Interactive Climate Scenario Presets
# ---------------------------------------------------------------------------
st.markdown("#### Interactive Climate Scenario Presets")
st.caption("Click any climate event scenario to instantly simulate agronomic recommendations:")

scenarios = {
    "Severe Heatwave & Drought": {"temp": 41, "rain": 0, "hum": 25, "region": "Semi-Arid Central Plains", "crop": "Wheat"},
    "Heavy Monsoon Flood Risk": {"temp": 27, "rain": 220, "hum": 94, "region": "Coastal River Basin", "crop": "Rice"},
    "Optimal Temperate Conditions": {"temp": 23, "rain": 35, "hum": 62, "region": "Northern Plateau", "crop": "Maize"},
    "Cold Snap & Early Frost": {"temp": 3, "rain": 5, "hum": 78, "region": "Highland Valley", "crop": "Pulses"},
}

sc_cols = st.columns(len(scenarios))
for s_i, (s_name, s_vals) in enumerate(scenarios.items()):
    with sc_cols[s_i]:
        if st.button(s_name, key=f"sc_btn_{s_i}", use_container_width=True):
            st.session_state["weather_temp"] = s_vals["temp"]
            st.session_state["weather_rain"] = s_vals["rain"]
            st.session_state["weather_hum"] = s_vals["hum"]
            st.session_state["weather_region"] = s_vals["region"]
            st.session_state["weather_crop"] = s_vals["crop"]

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input(
        "Temperature (°C)", 
        -10, 55, 
        value=int(st.session_state.get("weather_temp", 28))
    )
with col2:
    rainfall = st.number_input(
        "Recent rainfall (mm, last 7 days)", 
        0, 500, 
        value=int(st.session_state.get("weather_rain", 20))
    )
with col3:
    humidity = st.slider(
        "Humidity (%)", 
        0, 100, 
        value=int(st.session_state.get("weather_hum", 60))
    )

col_r, col_c = st.columns(2)
with col_r:
    region = st.text_input(
        "Region / location", 
        value=st.session_state.get("weather_region", "Madhya Pradesh, India"),
        placeholder="e.g. Punjab, India or Midwest, USA",
    )
with col_c:
    crop = st.text_input(
        "Crop of interest", 
        value=st.session_state.get("weather_crop", "Soybean"),
        placeholder="e.g. Soybean, Wheat, Maize",
    )

if st.button("Generate Weather-Aware Agronomic Advisory", type="primary", use_container_width=True):
    q = (
        f"Given weather conditions: Temperature {temp}°C, {rainfall}mm cumulative rainfall in the last 7 days, "
        f"and {humidity}% relative humidity in {region}, what does peer-reviewed agricultural research recommend "
        f"for managing {crop} right now? Address: (1) irrigation adjustment, (2) pathogen/pest outbreak risks under this humidity/temperature, "
        f"and (3) protective field operations."
    )
    with st.spinner("Evaluating weather stresses against indexed agricultural literature..."):
        try:
            result = ask_question(q)
            st.write(result["answer"])
            if result.get("sources"):
                with st.expander(f"Supporting Citations ({len(result['sources'])})"):
                    for s in result["sources"]:
                        st.markdown(f"- **{s['source']}** (Page {s['page']})")
        except EnvironmentError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()
st.caption("Live weather API integration (OpenWeather) planned for automatic regional syncing.")
