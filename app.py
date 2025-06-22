
import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("g3_xgb_model.pkl")

# App title
st.title("🎓 Student Grade Predictor (G3)")

# Input fields
g1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
g2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
studytime = st.selectbox("Study Time (hours)", [1, 2, 3, 4])
absences = st.number_input("Number of Absences", 0, 100, 4)

# Predict button
if st.button("Predict Final Grade (G3)"):
    # Prepare input
    features = np.array([[g1, g2, studytime, absences]])
    prediction = model.predict(features)[0]
    
    st.success(f"📚 Predicted Final Grade (G3): {prediction:.2f}")
