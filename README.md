# 🎓 EduScope: Student Dropout Predictor

A machine learning-powered web app to predict the risk of student dropout based on academic and behavioral data.

## 🔍 Features

- Predict for individual students using a simple form
- Bulk prediction for 100+ students via CSV upload
- Download annotated prediction results

## 🚀 Tech Stack

- Streamlit
- Scikit-Learn
- Pandas, NumPy
- Trained using Random Forest

## 🧠 Model Input Features

- Attendance %
- Average Grade
- Assignment Completion %
- Participation Score (0–10)
- Internet Access (Yes/No)
- Parental Education (High School, Bachelor, Master, PhD)

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
