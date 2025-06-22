import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("g3_xgb_model.pkl")

# Title
st.title("🎓 Student Grade Predictor (G3)")

# Inputs from user
g1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
g2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
studytime = st.selectbox("Study Time (1 = <2hrs, 4 = >10hrs)", [1, 2, 3, 4])

# Predict button
if st.button("Predict Final Grade (G3)"):
    # Create full feature set with default values
    input_data = {
        "G1": [g1],
        "G2": [g2],
        "studytime": [studytime],
        "age": [17],
        "Medu": [2],
        "Fedu": [2],
        "famrel": [3],
        "freetime": [3],
        "goout": [3],
        "Walc": [1],
        "health": [3],
        "school_MS": [1],
        "sex_M": [1],
        "address_U": [1],
        "famsize_LE3": [0],
        "Pstatus_T": [1],
        "Mjob_health": [0],
        "Mjob_other": [1],
        "Mjob_services": [0],
        "Mjob_teacher": [0],
        "Fjob_health": [0],
        "Fjob_other": [1],
        "Fjob_services": [0],
        "Fjob_teacher": [0],
        "reason_home": [0],
        "reason_other": [0],
        "reason_reputation": [1],
        "guardian_mother": [1],
        "guardian_other": [0],
        "schoolsup_yes": [0],
        "famsup_yes": [0],
        "paid_yes": [0],
        "activities_yes": [0],
        "nursery_yes": [1],
        "higher_yes": [1],
        "internet_yes": [1],
        "romantic_yes": [0],
    }

    input_df = pd.DataFrame(input_data)

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"📚 Predicted Final Grade (G3): {prediction:.2f}")
    except Exception as e:
        st.error("❌ Prediction failed. See error details below:")
        st.code(repr(e))  # Shows full error type + message
        st.write("📊 Input DataFrame shape:", input_df.shape)
        st.write("📋 Input DataFrame preview:")
        st.dataframe(input_df)
