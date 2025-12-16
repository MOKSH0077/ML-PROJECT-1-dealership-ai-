# -*- coding: utf-8 -*-





# streamlit_app.py
%%writefile dealership_Ai.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# ===============================
# Load Pickle Files
# ===============================
model = joblib.load("used_car_model.pkl")
ct = joblib.load("column_transformer.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
feature_names = list(feature_names)


# ===============================
# Vision Pro UI CSS
# ===============================
st.markdown("""
<style>

body {
    background: #000000;
}

.main {
    background: linear-gradient(145deg, #0a0a0a, #121212);
    padding: 0 !important;
}

.glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(25px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 0px 40px rgba(0, 140, 255,0.1);
    border-radius: 22px;
    padding: 25px;
    margin-top: 20px;
}

.title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 22px;
    font-weight: 600;
    color: #cccccc;
    margin-bottom: 10px;
}

* {
    transition: 0.3s ease-in-out;
}

.divider {
    height: 2px;
    margin: 25px 0;
    background: linear-gradient(90deg, transparent, #007aff, transparent);
}

</style>
""", unsafe_allow_html=True)


# ===============================
# Title
# ===============================
st.markdown("<h1 class='title'>🔮 AI Used Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# ===============================
# INPUT PANEL
# ===============================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>🔧 Input Car Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", ["Audi", "BMW", "Honda", "Hyundai", "Toyota"])
    model_name = st.selectbox(
        "Model",
        ["A3", "A4", "3 Series", "5 Series", "City", "Civic", "Creta", "Fortuner", "Corolla", "i20"]
    )
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])

with col2:
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    owner_label = st.selectbox(
        "Ownership",
        ["First Owner", "Second Owner", "Third Owner"]
    )

    owner_map = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3
    }
    owner = owner_map[owner_label]

    year = st.slider("Manufacturing Year", 2005, 2024, 2018)

mileage = st.slider("Mileage (km driven)", 0, 300000, 45000, step=1000)
engine_size = st.slider("Engine Size (CC)", 800, 5000, 1500)
horsepower = st.slider("Horsepower (HP)", 50, 500, 120)

st.markdown("</div>", unsafe_allow_html=True)


# ===============================
# PREDICTION PANEL
# ===============================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>📊 Prediction Panel</div>", unsafe_allow_html=True)

if st.button("🚗 Predict Car Price", use_container_width=True):
    try:

        df = pd.DataFrame([[brand, model_name, year, mileage, fuel,
                            transmission, engine_size, horsepower, owner]],
                          columns=["Brand", "Model", "Year", "Mileage",
                                   "FuelType", "Transmission", "EngineSize",
                                   "Horsepower", "Owner"])


        encoded = ct.transform(df)







        final_df = pd.DataFrame(encoded, columns=feature_names)

        # Prediction
        prediction = model.predict(final_df)[0]
        lower = prediction * 0.92
        upper = prediction * 1.12

        st.success(f"### 💰 Estimated Price: **₹ {prediction:,.0f}**")
        st.write(f"Lower Range: ₹ {lower:,.0f}")
        st.write(f"Upper Range: ₹ {upper:,.0f}")

        # ===============================
        # PRICE RANGE GRAPH
        # ===============================
        x = ["Lower", "Predicted", "Upper"]
        y = [lower, prediction, upper]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(width=4),
            fill="tozeroy",
            marker=dict(size=12)
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            font=dict(color="white"),
            xaxis_title="Price Category",
            yaxis_title="Price (₹)"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)













