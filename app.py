import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# Load Model Files
model = joblib.load("Logistic_Regression_heart.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# Custom CSS
st.markdown("""
<style>
.big-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ff4b4b;
}

.subtitle {
    text-align: center;
    color: grey;
    font-size: 18px;
    margin-bottom: 25px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<p class="big-title">❤️ Heart Disease Risk Predictor</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Based Heart Disease Prediction System</p>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("📌 About Project")

    st.write("""
    This application predicts the likelihood of heart disease
    using a Machine Learning model trained on patient health data.
    """)

    st.info(
        "⚠️ This tool is for educational purposes only and should not replace medical advice."
    )

    st.markdown("---")
    st.write("Developed by Yashasvi Tomar")

# Input Section
st.subheader("📝 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)

    sex = st.selectbox(
        "Sex",
        ["M", "F"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=250,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# Predict Button
predict = st.button("🔍 Predict Heart Disease Risk")

if predict:

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")

    st.metric(
        label="Heart Disease Risk Score",
        value=f"{probability * 100:.1f}%"
    )

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Heart Disease\n\nRisk Score: {probability * 100:.1f}%"
        )
    else:
        st.success(
            f"✅ Low Risk of Heart Disease\n\nConfidence: {(1 - probability) * 100:.1f}%"
        )

# Footer
st.markdown("---")
st.caption(
    "❤️ Heart Disease Prediction using Machine Learning | Developed by Yashasvi Tomar"
)
