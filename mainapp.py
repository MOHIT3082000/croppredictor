import streamlit as st
import pickle
import numpy as np

# Load the model
model_path = 'model_compressed.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit interface
st.title('Crop Yield Prediction')

# Input parameters
N = st.number_input('Nitrogen content (N)', min_value=0, max_value=300, value=50)
P = st.number_input('Phosphorus content (P)', min_value=0, max_value=300, value=50)
K = st.number_input('Potassium content (K)', min_value=0, max_value=300, value=50)
temperature = st.number_input('Temperature (°C)', min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input('pH value', min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=100.0)

# Prediction button
if st.button('Predict'):
    # Make prediction
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)

    st.write('Predicted Label:', prediction[0])

if __name__ == '__main__':
    st.run()
