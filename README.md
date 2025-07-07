# 🎓 Predicting Student Performance with Real-World Data

This machine learning project aims to predict students' final grades (G3) using a real-world dataset collected from Portuguese secondary schools. The model helps identify key factors influencing academic performance.

## 📌 Project Overview

- **Goal**: Predict students' final math grades (G3) based on various features like previous grades, study time, absences, and family background.
- **Dataset**: `student-mat.csv` from the UCI Machine Learning Repository.
- **Target Variable**: G3 (Final Grade)

## 🧠 What I Did

- 🧹 Cleaned and preprocessed the data  
- 📊 Performed exploratory data analysis (EDA)  
- ⚙️ Engineered useful features  
- 🔍 Trained and evaluated multiple ML models  
- 🏆 Tuned a boosted regression model (XGBoost) for best performance

## 📈 Model Performance

- **Best Model**: XGBoost Regressor  
- **Metrics (Cross-Validated)**:
  - Mean Absolute Error (MAE): *~1.25*  
  - Root Mean Squared Error (RMSE): *~2.05*  
  - R² Score: *~0.79*

## 🛠️ Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib & Seaborn

## 📂 Structure

```bash
├── data/
│   └── student-mat.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── model/
│   └── trained_model.pkl
├── app/ (optional, if deployed)
│   └── streamlit_app.py
├── README.md
