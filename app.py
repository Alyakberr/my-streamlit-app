import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb

# Load the model
model = joblib.load("g3_xgb_model.pkl")

# App title
st.title("🎓 Student Grade Predictor (G3)")

# Input fields
g1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
g2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
studytime = st.selectbox("Study Time (1 = <2hrs, 4 = >10hrs)", [1, 2, 3, 4])
absences = st.number_input("Number of Absences", 0, 100, 4)

# Predict button
if st.button("Predict Final Grade (G3)"):
    input_df = pd.DataFrame({
        "G1": [g1],
        "G2": [g2],
        "studytime": [studytime],
        "absences": [absences]
    })

    try:
        # 🧪 Debug: print what the model expects
        st.write("Expected columns:", model.feature_names_in_)

        # Prediction
        prediction = model.predict(input_df)[0]
        st.success(f"📚 Predicted Final Grade (G3): {prediction:.2f}")
    except Exception as e:
        st.error("❌ Prediction failed. Error:")
        st.code(str(e))
