import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ---------- 2. Generate Synthetic Data ----------
def generate_dataset(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "student_id": range(1, n+1),
        "attendance_percent": np.random.normal(85, 10, n).clip(50, 100),
        "avg_grade": np.random.normal(75, 12, n).clip(40, 100),
        "assignment_completion": np.random.normal(80, 10, n).clip(50, 100),
        "participation_score": np.random.randint(0, 11, n),
        "internet_access": np.random.choice(["Yes", "No"], n, p=[0.8, 0.2]),
        "parental_education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n),
    })

    df["dropout"] = ((df["attendance_percent"] < 70) | (df["avg_grade"] < 60) |
                      (df["assignment_completion"] < 65) | (df["participation_score"] < 3)).astype(int)
    return df

df = generate_dataset()

# ---------- 3. Preprocess Data ----------
df.drop(columns=["student_id"], inplace=True)
df["internet_access"] = df["internet_access"].map({"Yes": 1, "No": 0})
df = pd.get_dummies(df, columns=["parental_education"], drop_first=True)

X = df.drop("dropout", axis=1)
y = df["dropout"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 4. Train Random Forest Model ----------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

import joblib

# Save the trained model
joblib.dump(model, "rf_model.pkl")

# Save the fitted scaler
joblib.dump(scaler, "scaler.pkl")

# ---------- 5. Evaluate Model ----------
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------- 6. Save Model and Scaler ----------
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")


