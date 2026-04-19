import streamlit as st
from advisor import get_farm_advisory
import pandas as pd
import joblib

# --- Page Config ---
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# --- Load Model Files ---
model = joblib.load("models/model.pkl")
feature_columns = joblib.load("models/features.pkl")
model_name = joblib.load("models/model_name.pkl")
metrics = joblib.load("models/metrics.pkl")
areas = joblib.load("models/areas.pkl")
items = joblib.load("models/items.pkl")

# --- Header ---
st.title("🌾 Crop Yield Prediction System")
st.markdown("Fill in your farm details below to get an estimated crop yield.")
st.divider()

# --- Input Section ---
st.subheader("📋 Farm Details")

col1, col2 = st.columns(2)

with col1:
    area = st.selectbox("🌍 Country / Region", options=areas)
    item = st.selectbox("🌱 Crop Type", options=items)
    year = st.number_input("📅 Year", min_value=1990, max_value=2030, value=2020)

with col2:
    rainfall = st.number_input("🌧️ Average Rainfall (mm/year)", min_value=0.0, value=1000.0)
    pesticides = st.number_input("🧪 Pesticides Used (tonnes)", min_value=0.0, value=100.0)
    temperature = st.number_input("🌡️ Average Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)

st.divider()

# --- Predict Button ---
if st.button("🔍 Predict Yield", use_container_width=True):

    # --- Input Validation ---
    if rainfall == 0:
        st.warning("⚠️ Rainfall is set to 0. Are you sure? This may affect prediction accuracy.")
    if temperature <= 0:
        st.warning("⚠️ Temperature is very low. Please double-check your value.")

    # --- Build Input ---
    input_dict = {
        "Area": area,
        "Item": item,
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temperature
    }
    input_df = pd.DataFrame([input_dict])

    # --- Feature Engineering ---
    input_df["rain_temp_ratio"] = (
        input_df["average_rain_fall_mm_per_year"] / (input_df["avg_temp"] + 1)
    )

    # --- One-hot Encoding + Align ---
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # --- Prediction ---
    prediction = model.predict(input_df)[0]
    prediction_kg = prediction / 10  # convert hg/ha to kg/ha

    # --- Yield Category ---
    if prediction_kg < 1500:
        category = "🔴 Low Yield"
        category_color = "red"
    elif prediction_kg < 3500:
        category = "🟡 Medium Yield"
        category_color = "orange"
    elif prediction_kg < 6000:
        category = "🟢 High Yield"
        category_color = "green"
    else:
        category = "🏆 Exceptional Yield"
        category_color = "blue"

    # --- Results Section ---
    st.subheader("📊 Prediction Results")
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric(label="Predicted Yield", value=f"{prediction_kg:,.0f} kg/ha")
    with res_col2:
        st.metric(label="Yield Category", value=category)
    with res_col3:
        st.metric(label="Model Used", value=model_name)

    st.divider()

    # --- AI Advisory Section ---
    st.subheader("🤖 AI Farm Advisory Report")

    with st.spinner("Generating your personalized farm advisory..."):
        try:
            advisory = get_farm_advisory(
                area, item, year, rainfall, pesticides, temperature, prediction_kg
            )

            st.markdown("### 📋 Crop & Field Summary")
            st.info(advisory["crop_summary"])

            st.markdown("### 📊 Yield Interpretation")
            st.success(advisory["yield_interpretation"])

            st.markdown("### ⚠️ Risk Factors")
            for risk in advisory["risk_factors"]:
                if risk["severity"] == "High":
                    st.error(f"🔴 **{risk['factor']}** — {risk['explanation']}")
                elif risk["severity"] == "Medium":
                    st.warning(f"🟡 **{risk['factor']}** — {risk['explanation']}")
                else:
                    st.info(f"🟢 **{risk['factor']}** — {risk['explanation']}")

            st.markdown("### ✅ Recommended Actions")
            for i, action in enumerate(advisory["recommended_actions"], 1):
                st.markdown(f"**{i}. {action['action']}**")
                st.caption(action["reason"])

            st.divider()
            st.caption(f"⚠️ {advisory['disclaimer']}")
            


        except Exception as e:
            st.error(f"Could not generate advisory. Error: {e}")