import streamlit as st
import requests

# Title for the app
st.title("Iris Species Prediction App")

# Input fields for the features
sepal_length = st.number_input("Sepal Length", value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width", value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length", value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width", value=0.0, format="%.2f")

# Button to trigger prediction
if st.button("Predict"):
    # Prepare input data
    data = {"data": [[sepal_length, sepal_width, petal_length, petal_width]]}

    try:
        # Send POST request to Flask API
        response = requests.post("http://127.0.0.1:5000/predict", json=data)

        if response.status_code == 200:
            # Display the prediction result
            species = response.json()["species"][0]  # Get the first species
            st.success(f"Predicted Species: {species}")
        else:
            # Handle API errors
            st.error(f"Error from API: {response.text}")

    except requests.exceptions.RequestException as e:
        # Handle connection issues
        st.error(f"Failed to connect to API: {e}")
