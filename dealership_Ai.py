import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# Load Pickle Files

# LOADING THE TRAINED MODELS.
model = joblib.load("car_price_model.pkl")
ct = joblib.load("column_transformer.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


# Vision Pro UI CSS

st.markdown("""
<style>

body {
    background: #000000;
}

/* page bg */
.main {
    background: linear-gradient(145deg, #0a0a0a, #121212);
    padding: 0 !important;
}

/* Vision Pro glass container */
.glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(25px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 0px 40px rgba(0, 140, 255, 0.1);
    border-radius: 22px;
    padding: 25px;
    margin-top: 20px;
}

/* neon heading */
.title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* sub */
.subtitle {
    font-size: 22px;
    font-weight: 600;
    color: #cccccc;
    margin-bottom: 10px;
}

/* smooth animations */
* {
    transition: 0.3s ease-in-out;
}

/* glowing divider */
.divider {
    height: 2px;
    margin: 25px 0;
    background: linear-gradient(90deg, transparent, #007aff, transparent);
}

/* slider glow */
.css-1a2ok8l {
    accent-color: #00aaff !important;
}

</style>
""", unsafe_allow_html=True)


# Title

st.markdown("<h1 class='title'>🔮 Ai UsedCar Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# INPUT PANEL

st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>🔧 Input Car Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fuel = st.selectbox("Fuel Type", ["gas", "diesel"])
    body = st.selectbox("Car Body", ["sedan", "hatchback", "suv", "wagon", "convertible"])
    drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])

with col2:
    brand = st.selectbox("Brand", ["Toyota", "BMW", "Audi", "Honda", "Hyundai", "Kia", "Mercedes"])

enginesize = st.slider("Engine Size (CC)", 1, 500, 150)
horsepower = st.slider("Horsepower", 1, 500, 120)

st.markdown("</div>", unsafe_allow_html=True)

# PREDICTION

# WE ALREADY LOAD ALL SAVED PKL FILES SO THAT WE CAN USE THEM IN THIS SCRIPT.
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>📊 Prediction Panel</div>", unsafe_allow_html=True)

if st.button("🚗 Predict Car Price", use_container_width=True):# IF USER PRESS BUTTON
    try:
        df = pd.DataFrame([[fuel, body, drivewheel, enginesize, horsepower, brand]],
                          columns=["fueltype", "carbody", "drivewheel", "enginesize", "horsepower", "Brand"])
        # WE FIRST MANNUALY CREATE DATAFRAME OF FEATURES OF OUR MODEL BUT IF THERE ARE MANY FEATURES WE NOT DO IT MANUALLY.
        encoded = ct.transform(df)
        # NOW WE GET THE ENCODED DATAFRAME THAT CONTAINS OHE FEATURES AND 2 NUMERIC FEATURES.

        try:
            encoded[:, -2:] = scaler.transform(encoded[:, -2:])
            # NOW AFTER GETTING ENCODED DATAFRAME THAT HAVE OHE AND 2 NUMERIC FEATURES , SO NOW WE DO FEATURE SCALLING ON THESE 2 NUMERIC FEATURES.
        except:
            pass

        final_df = pd.DataFrame(encoded, columns=feature_names)
        # NOW WE GET THE OUR FINAL DATAFRAME THAT CONTAIN OHE AND SCALED FEATURES AND ALSO WE HAVE OUR FEATURE NAMES IN TRAINING ORDER.
        # WE HAVE DATAFRAME OF 1 ROW AND MANY COLUMNS.
        prediction = model.predict(final_df)[0]
        # WE USE[0] AS sklearn predictions always return array/list form—even for one row.
        # model.predict() pandas DataFrame return nahi karta.
        # Woh NumPy array return karta hai.

        lower = prediction * 0.92
        upper = prediction * 1.12

        st.success(f"### 💰 Estimated Price: **$ {prediction:,.0f}**")
        st.write(f"Lower Range: $ {lower:,.0f}")
        st.write(f"Upper Range: $ {upper:,.0f}")

     
        # TRENDY GRAPH (NEON CURVE + GRADIENT)
        
        x = ["Lower Price", "Predicted Price", "Upper Price"]
        y = [lower, prediction, upper]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(width=4),
            fill="tozeroy",
            fillcolor="rgba(0,122,255,0.25)",
            marker=dict(size=12)
        ))

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=350,
            font=dict(color="white"),
            xaxis_title="Price Category",
            yaxis_title="Price (₹)",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
