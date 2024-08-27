import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Load the saved model and scaler
model = tf.keras.models.load_model('support2_model.h5', compile=False)
scaler = joblib.load('support2_scaler.pkl')

# Streamlit UI
st.title("Death Prediction System")

# Sidebar for input features
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=45)
blood_glucose_level = st.sidebar.number_input("Blood Glucose Level (mg/dL)", min_value=0, max_value=500, value=135)
total_cost = st.sidebar.number_input("Total Cost ($)", min_value=0.0, max_value=10000.0, value=144.73)
disease_group = st.sidebar.selectbox("Disease Group (0-7)", options=[0, 1, 2, 3, 4, 5, 6, 7], index=6)
blood_ph_level = st.sidebar.number_input("Blood pH Level", min_value=0.0, max_value=14.0, value=4.5)
disease_classification = st.sidebar.selectbox("Disease Classification (0-3)", options=[0, 1, 2, 3], index=1)

# Prepare the input data
new_data = pd.DataFrame({
    'Age': [age],
    'Blood_Glucose_Level': [blood_glucose_level],
    'Total_Cost': [total_cost],
    'Disease_Group': [disease_group],
    'Blood_pH_Level': [blood_ph_level],
    'Disease_Classification': [disease_classification]
})

# Scale the numerical features
new_data[['Age', 'Blood_Glucose_Level', 'Total_Cost', 'Blood_pH_Level']] = scaler.transform(
    new_data[['Age', 'Blood_Glucose_Level', 'Total_Cost', 'Blood_pH_Level']]
)

# Make prediction
predictions = model.predict(new_data)
predictions = (predictions > 0.5).astype("int32").flatten()
prediction_label = 'Death' if predictions[0] == 1 else 'Not Death'
st.write(f"Prediction: **{prediction_label}**")

# Customize the UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #e5e5e5;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    .stButton>button {
        background-color: #0099ff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
