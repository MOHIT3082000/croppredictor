import streamlit as st
import numpy as np
import pickle  # Assuming you're loading a trained model

# Load trained model (ensure 'model.pkl' exists)
with open("model_compressed.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Crop Prediction Model")

# User input fields
N = st.number_input("Nitrogen", min_value=0.0, format="%.2f")
P = st.number_input("Phosphorus", min_value=0.0, format="%.2f")
K = st.number_input("Potassium", min_value=0.0, format="%.2f")
temperature = st.number_input("Temperature", format="%.2f")
humidity = st.number_input("Humidity", format="%.2f")
ph = st.number_input("pH Level", format="%.2f")
rainfall = st.number_input("Rainfall", format="%.2f")

# Prediction button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_data)

    # Display prediction
    st.write("Predicted Label:", prediction[0])
