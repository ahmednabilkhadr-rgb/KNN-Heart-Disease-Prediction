import pandas as pd
import joblib

# Load Trained KNN Model
model = joblib.load("knn_model.pkl")

print("=" * 45)
print("       KNN HEART DISEASE PREDICTION")
print("=" * 45)

print("\nEnter Patient Information:\n")

# Get Patient Data
age = int(input("Age: "))
sex = int(input("Sex (1 = Male, 0 = Female): "))
cp = int(input("Chest Pain Type (1-4): "))
trestbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (0 = No, 1 = Yes): "))
restecg = int(input("Resting ECG (0-2): "))
thalach = int(input("Maximum Heart Rate: "))
exang = int(input("Exercise Induced Angina (0 = No, 1 = Yes): "))
oldpeak = float(input("Oldpeak: "))
slope = int(input("Slope (1-3): "))
ca = float(input("Number of Major Vessels (0-3): "))
thal = float(input("Thal (3, 6, or 7): "))

# Prepare Patient Data
patient_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# Make Prediction
prediction = model.predict(patient_data)[0]

# Display Result
print("\n" + "=" * 45)
print("             PREDICTION RESULT")
print("=" * 45)

if prediction == 1:
    print("Result: Heart Disease Predicted")
else:
    print("Result: No Heart Disease Predicted")

print("=" * 45)
print("Educational prediction only.")
print("This is not a medical diagnosis.")