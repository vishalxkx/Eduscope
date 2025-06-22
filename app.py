# ========================================
# Streamlit App: EduScope Student Dropout Predictor (Single & Bulk Prediction)
# ========================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="EduScope - Dropout Predictor", layout="centered")
st.title("🎓 EduScope: Predicting Student Dropouts")

st.markdown("""
This tool predicts whether a student is likely to drop out based on their academic and behavioral data.
You can either enter a single student's data or upload a CSV file with multiple students.
""")

# Load model and scaler
try:
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")

    mode = st.radio("Select Prediction Mode", ["Single Student", "Bulk Upload (CSV)"])

    if mode == "Single Student":
        attendance = st.slider("Attendance (%)", 50, 100, 85, key="attendance")
        grade = st.slider("Average Grade", 40, 100, 75, key="grade")
        assignments = st.slider("Assignment Completion (%)", 50, 100, 80, key="assignments")
        participation = st.slider("Participation Score (0-10)", 0, 10, 5, key="participation")
        internet = st.selectbox("Internet Access at Home", ["Yes", "No"], key="internet")
        parent_edu = st.selectbox("Parental Education Level", ["Bachelor", "High School", "Master", "PhD"], key="parent_edu")

        internet = 1 if internet == "Yes" else 0
        parent_edu_dict = {
            "Bachelor": [0, 0, 0],
            "High School": [1, 0, 0],
            "Master": [0, 1, 0],
            "PhD": [0, 0, 1]
        }
        parent_edu_encoded = parent_edu_dict[parent_edu]

        features = np.array([[attendance, grade, assignments, participation, internet] + parent_edu_encoded])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        if st.button("Predict", key="predict_button"):
            if prediction == 0:
                st.success("✅ The student is **NOT likely** to drop out.")
            else:
                st.error("⚠️ The student is **AT RISK** of dropping out.")

    else:
        st.info("Upload a CSV file with columns: attendance_percent, avg_grade, assignment_completion, participation_score, internet_access, parental_education")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Preprocess bulk input
                df["internet_access"] = df["internet_access"].map({"Yes": 1, "No": 0})
                df = pd.get_dummies(df, columns=["parental_education"], drop_first=False)

                for col in ["parental_education_High School", "parental_education_Master", "parental_education_PhD"]:
                    if col not in df.columns:
                        df[col] = 0

                # Reorder columns
                final_cols = ["attendance_percent", "avg_grade", "assignment_completion", "participation_score", "internet_access",
                              "parental_education_High School", "parental_education_Master", "parental_education_PhD"]
                df = df[final_cols]

                X_scaled = scaler.transform(df)
                predictions = model.predict(X_scaled)
                df["Dropout Prediction"] = ["No" if p == 0 else "Yes" for p in predictions]

                st.success("Predictions complete!")
                st.dataframe(df)

                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predictions as CSV", data=csv_download, file_name="predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error processing file: {e}")

except FileNotFoundError:
    st.warning("Model or Scaler not found. Please ensure 'rf_model.pkl' and 'scaler.pkl' are in the same folder.")
