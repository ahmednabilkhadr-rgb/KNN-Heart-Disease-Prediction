import streamlit as st
import pandas as pd
import joblib
import json
import random
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CardioAI | KNN Dashboard",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #26364D;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label, span {
    color: #CBD5E1;
}

.metric-card {
    background: linear-gradient(135deg, #17263D, #102B3A);
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #29465B;
    margin-bottom: 12px;
}

.metric-title {
    color: #94A3B8;
    font-size: 14px;
}

.metric-value {
    color: #2DD4BF;
    font-size: 27px;
    font-weight: bold;
}

.section-card {
    background-color: #142137;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #26364D;
    margin-bottom: 15px;
}

.stButton button {
    background: linear-gradient(90deg, #0D9488, #14B8A6);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: bold;
    min-height: 42px;
    width: 100%;
}

.stButton button:hover {
    background: #0F766E;
    color: white;
}

div[data-testid="stFormSubmitButton"] button {
    background: #0D9488;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL AND METRICS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "knn_model.pkl")

@st.cache_data
def load_metrics():
    with open(BASE_DIR / "metrics.json", "r") as file:
        return json.load(file)

try:
    model = load_model()
    metrics = load_metrics()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.info("Run your KNN training script first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()


# ==========================================
# 4. FEATURES
# ==========================================

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

BEST_K = metrics.get("best_k", "N/A")
ACCURACY = metrics.get("accuracy", 0)
CV_ACCURACY = metrics.get("cv_accuracy", 0)


# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown("# ❤️ CardioAI")
st.sidebar.caption("KNN Heart Disease Prediction")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Visualizations",
        "Patient Prediction",
        "Random Prediction",
        "Model Information"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Project")
st.sidebar.caption("Algorithm: K-Nearest Neighbors")


# ==========================================
# 6. HELPER FUNCTIONS
# ==========================================

def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def show_prediction(patient_df):
    prediction = int(model.predict(patient_df)[0])

    st.markdown("### Prediction Result")

    if prediction == 1:
        st.error("⚠️ Model Prediction: Heart Disease")
    else:
        st.success("✓ Model Prediction: No Heart Disease")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(patient_df)[0]

        probability_df = pd.DataFrame({
            "Class": ["No Heart Disease", "Heart Disease"],
            "Probability": probabilities * 100
        })

        fig = px.bar(
            probability_df,
            x="Class",
            y="Probability",
            color="Class",
            text_auto=".2f",
            color_discrete_sequence=["#2DD4BF", "#F87171"]
        )

        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability (%)",
            xaxis_title="",
            yaxis_range=[0, 100],
            paper_bgcolor="#142137",
            plot_bgcolor="#142137",
            font_color="#FFFFFF",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Educational prediction only. This result is not a medical diagnosis."
    )


def generate_random_patient():
    return {
        "age": random.randint(25, 80),
        "sex": random.choice([0, 1]),
        "cp": random.randint(1, 4),
        "trestbps": random.randint(90, 180),
        "chol": random.randint(120, 350),
        "fbs": random.choice([0, 1]),
        "restecg": random.randint(0, 2),
        "thalach": random.randint(70, 200),
        "exang": random.choice([0, 1]),
        "oldpeak": round(random.uniform(0, 6), 1),
        "slope": random.randint(1, 3),
        "ca": random.randint(0, 3),
        "thal": random.choice([3, 6, 7])
    }


# ==========================================
# 7. OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.title("CardioAI Dashboard")
    st.caption("Heart Disease Prediction Using Machine Learning")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Model Accuracy", f"{ACCURACY * 100:.2f}%")

    with col2:
        metric_card("Best K Value", BEST_K)

    with col3:
        metric_card("Cross Validation", f"{CV_ACCURACY * 100:.2f}%")

    st.divider()

    st.subheader("Model Performance")

    report = metrics.get("classification_report", {})

    class_names = ["No Heart Disease", "Heart Disease"]

    report_data = []

    for class_name in class_names:
        if class_name in report:
            report_data.append({
                "Class": class_name,
                "Precision": report[class_name].get("precision", 0),
                "Recall": report[class_name].get("recall", 0),
                "F1 Score": report[class_name].get("f1-score", 0)
            })

    if report_data:
        report_df = pd.DataFrame(report_data)

        fig = px.bar(
            report_df,
            x="Class",
            y=["Precision", "Recall", "F1 Score"],
            barmode="group",
            title="Classification Report",
            color_discrete_sequence=["#2DD4BF", "#60A5FA", "#FBBF24"]
        )

        fig.update_layout(
            paper_bgcolor="#142137",
            plot_bgcolor="#142137",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This project uses the K-Nearest Neighbors algorithm "
        "to predict heart disease based on patient data."
    )


# ==========================================
# 8. VISUALIZATIONS PAGE
# ==========================================

elif page == "Visualizations":

    st.title("Model Visualizations")
    st.caption("Charts generated during model training")

    st.divider()

    plots = [
        ("KNN K Comparison", "knn_k_comparison.png"),
        ("Cross Validation Results", "knn_cross_validation.png"),
        ("Confusion Matrix", "knn_confusion_matrix.png")
    ]

    for title, filename in plots:
        st.subheader(title)

        image_path = BASE_DIR / "plots" / filename

        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning(f"Plot not found: {filename}")


# ==========================================
# 9. MANUAL PATIENT PREDICTION
# ==========================================

elif page == "Patient Prediction":

    st.title("Patient Prediction")
    st.caption("Enter patient information to predict the result.")

    st.divider()

    with st.form("patient_form"):

        st.subheader("Patient Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 1, 100, 45)
            sex = st.selectbox("Sex", [1, 0])
            cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
            trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
            chol = st.number_input("Cholesterol", 50, 600, 200)

        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
            restecg = st.selectbox("Resting ECG", [0, 1, 2])
            thalach = st.number_input("Maximum Heart Rate", 50, 250, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])

        with col3:
            oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
            slope = st.selectbox("Slope", [1, 2, 3])
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
            thal = st.selectbox("Thal", [3, 6, 7])

        submitted = st.form_submit_button(
            "Predict Heart Disease",
            use_container_width=True
        )

    if submitted:

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
        }], columns=FEATURES)

        st.divider()
        st.subheader("Entered Patient Data")
        st.dataframe(patient_data, use_container_width=True)

        show_prediction(patient_data)


# ==========================================
# 10. RANDOM DATA PREDICTION
# ==========================================

elif page == "Random Prediction":

    st.title("Random Data Prediction")
    st.caption("Generate random patient data and test the trained KNN model.")

    st.divider()

    st.markdown("""
    <div class="section-card">
        <h3>🎲 Random Patient Generator</h3>
        <p>
        Generate a random set of patient features, then let the trained
        KNN model predict the result.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "random_patient" not in st.session_state:
        st.session_state.random_patient = None

    col1, col2 = st.columns(2)

    with col1:
        generate = st.button("Generate Random Patient")

    with col2:
        predict_random = st.button("Generate & Predict")

    if generate:
        st.session_state.random_patient = generate_random_patient()

    if predict_random:
        st.session_state.random_patient = generate_random_patient()
        st.session_state.random_prediction_requested = True

    if st.session_state.random_patient is not None:

        random_df = pd.DataFrame(
            [st.session_state.random_patient],
            columns=FEATURES
        )

        st.subheader("Generated Patient Data")
        st.dataframe(random_df, use_container_width=True)

        if predict_random or st.session_state.get(
            "random_prediction_requested", False
        ):
            st.divider()
            show_prediction(random_df)
            st.session_state.random_prediction_requested = False

        else:
            if st.button("Predict This Random Patient"):
                show_prediction(random_df)

    else:
        st.info("Click Generate Random Patient to create sample data.")

    st.warning(
        "Random values are for educational testing only. "
        "They do not represent a real patient or a medical diagnosis."
    )


# ==========================================
# 11. MODEL INFORMATION PAGE
# ==========================================

elif page == "Model Information":

    st.title("Model Information")
    st.caption("Understanding the K-Nearest Neighbors algorithm")

    st.divider()

    st.subheader("About KNN")

    st.write("""
    K-Nearest Neighbors (KNN) is a supervised machine learning algorithm.

    It predicts the class of a new data point by looking at the nearest
    training examples and choosing the most common class among them.
    """)

    st.subheader("Project Details")

    st.markdown(f"""
    - **Algorithm:** K-Nearest Neighbors
    - **Best K Value:** {BEST_K}
    - **Model Accuracy:** {ACCURACY * 100:.2f}%
    - **Cross Validation Accuracy:** {CV_ACCURACY * 100:.2f}%
    - **Number of Features:** 13
    - **Target:** Heart Disease / No Heart Disease
    """)

    st.subheader("Dataset Features")

    feature_descriptions = {
        "age": "Patient age",
        "sex": "Patient sex",
        "cp": "Chest pain type",
        "trestbps": "Resting blood pressure",
        "chol": "Serum cholesterol",
        "fbs": "Fasting blood sugar",
        "restecg": "Resting ECG results",
        "thalach": "Maximum heart rate",
        "exang": "Exercise induced angina",
        "oldpeak": "ST depression",
        "slope": "Slope of ST segment",
        "ca": "Major vessels",
        "thal": "Thalassemia"
    }

    features_df = pd.DataFrame({
        "Feature": feature_descriptions.keys(),
        "Description": feature_descriptions.values()
    })

    st.dataframe(features_df, use_container_width=True)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center><p>CardioAI | KNN Machine Learning Project</p>"
    "<p>Educational use only - Not a medical diagnosis</p></center>",
    unsafe_allow_html=True
)