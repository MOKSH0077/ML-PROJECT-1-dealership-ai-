UsedCar Price Prediction App

This project is a Machine Learning–based web application that predicts car prices using input features such as fuel type, car body, drive wheel, engine size, horsepower, and brand. The model is deployed using Streamlit and is powered by preprocessing pipelines stored as .pkl files for consistent results during inference.

Features
Predicts car prices based on key technical and categorical attributes.
Uses trained ML model along with saved encoders and scalers.
Clean and interactive UI built with Streamlit.

Modular project structure with separate model and app files.

Project Structure
├── dealership_ai.py        # Streamlit application script
├── model.pkl               # Trained ML model
├── scaler.pkl              # Feature scaler
├── transformer.pkl         # OneHotEncoder / ColumnTransformer
├── feature_names.pkl       # Feature names used during training

How It Works

User inputs are collected through the Streamlit interface.
Inputs are converted into a dataframe matching the training structure.
Categorical features are encoded using the saved transformer.
Numerical features are scaled using the saved scaler.
The processed data is passed into the trained model for prediction.

Installation
pip install -r requirements.txt

Run Locally
streamlit run dealership_ai.py

Use Case
Ideal for car dealerships, data science learners, and ML deployment practice.

Contribution

Pull requests and suggestions are welcome.
