import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero-card {
    background: linear-gradient(135deg, #7F0000 0%, #C0392B 60%, #922B21 100%);
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
    letter-spacing: -0.5px;
}

.hero-sub {
    color: rgba(255,255,255,0.7);
    font-size: 1rem;
    margin-top: 0.4rem;
}

.heart-beat {
    font-size: 3.5rem;
    display: inline-block;
    animation: heartbeat 1s ease-in-out infinite;
    filter: drop-shadow(0 0 10px rgba(255,100,100,0.8));
}

@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    15%       { transform: scale(1.3); }
    30%       { transform: scale(1.05); }
    45%       { transform: scale(1.2); }
    60%       { transform: scale(1); }
}

.hero-pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.85);
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 99px;
    display: inline-block;
}

.ecg-strip {
    background: #0a0a0a;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.5rem;
    text-align: center;
}

.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #888;
    margin: 1.5rem 0 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #f0f0f0;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 56px;
    font-size: 1rem;
    font-weight: 600;
    background: linear-gradient(135deg, #C0392B, #922B21);
    color: white;
    border: none;
    letter-spacing: 0.02em;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #A93226, #7B241C);
    transform: translateY(-1px);
}

.result-card {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #e5e5e5;
    margin-top: 1.5rem;
}

.result-header-high {
    background: linear-gradient(135deg, #FCEBEB, #FADBD8);
    padding: 1.5rem;
    border-bottom: 1px solid #F5B7B1;
}

.result-header-mod {
    background: linear-gradient(135deg, #FEFDE7, #FDEBD0);
    padding: 1.5rem;
    border-bottom: 1px solid #FAD7A0;
}

.result-header-low {
    background: linear-gradient(135deg, #EAFAF1, #D5F5E3);
    padding: 1.5rem;
    border-bottom: 1px solid #A9DFBF;
}

.big-score-high { font-size: 3.5rem; font-weight: 700; color: #C0392B; line-height: 1; }
.big-score-mod  { font-size: 3.5rem; font-weight: 700; color: #D68910; line-height: 1; }
.big-score-low  { font-size: 3.5rem; font-weight: 700; color: #1D8348; line-height: 1; }

.verdict-high { color: #922B21; font-size: 0.9rem; margin-top: 4px; }
.verdict-mod  { color: #9A7D0A; font-size: 0.9rem; margin-top: 4px; }
.verdict-low  { color: #1D8348; font-size: 0.9rem; margin-top: 4px; }

.tier-badge-high { background:#F1948A; color:#7B241C; padding:6px 16px; border-radius:8px; font-size:0.85rem; font-weight:600; display:inline-block; }
.tier-badge-mod  { background:#F8C471; color:#784212; padding:6px 16px; border-radius:8px; font-size:0.85rem; font-weight:600; display:inline-block; }
.tier-badge-low  { background:#82E0AA; color:#1D6A39; padding:6px 16px; border-radius:8px; font-size:0.85rem; font-weight:600; display:inline-block; }

.metric-row {
    display: flex;
    gap: 0;
    border-top: 1px solid #eee;
}

.metric-item {
    flex: 1;
    padding: 1rem 1.25rem;
    border-right: 1px solid #eee;
    background: #fafafa;
}

.metric-item:last-child { border-right: none; }
.metric-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #999; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #222; margin-top: 2px; }
.metric-sub   { font-size: 0.7rem; color: #aaa; margin-top: 1px; }

.advice-block {
    background: #fff;
    padding: 1.25rem 1.5rem;
    border-top: 1px solid #eee;
}

.advice-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #999;
    margin-bottom: 0.75rem;
}

.advice-item {
    display: flex;
    gap: 10px;
    margin-bottom: 8px;
    align-items: flex-start;
    font-size: 0.85rem;
    color: #555;
    line-height: 1.5;
}

.advice-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
}

.footer {
    text-align: center;
    color: #bbb;
    font-size: 0.75rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #f0f0f0;
}

.stSlider > div > div > div { background: #C0392B !important; }
.stSelectbox > div > div { border-radius: 8px !important; }
.stNumberInput > div > div { border-radius: 8px !important; }

div[data-testid="stSidebarContent"] {
    background: #1a1a2e;
    color: white;
}
</style>
""", unsafe_allow_html=True)

model = joblib.load("Logistic_Regression_heart.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.markdown("""
<div class="hero-card">
    <div style="display:flex;align-items:center;gap:1.25rem;">
        <span class="heart-beat">❤️</span>
        <div>
            <p class="hero-title">Heart Disease<br>Risk Predictor</p>
            <p class="hero-sub">ML-powered cardiovascular risk assessment</p>
        </div>
    </div>
    <div class="hero-pills">
        <span class="pill">🧠 Logistic Regression</span>
        <span class="pill">📊 918-patient dataset</span>
        <span class="pill">⚠️ Educational use only</span>
        <span class="pill">🔬 Clinical features</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ecg-strip">
    <svg width="100%" height="54" viewBox="0 0 640 54" xmlns="http://www.w3.org/2000/svg">
        <polyline fill="none" stroke="#E74C3C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
            points="0,27 20,27 28,27 30,18 32,27 36,27 40,5 44,48 48,10 52,27 56,27
                    80,27 88,27 90,18 92,27 96,27 100,5 104,48 108,10 112,27 116,27
                    140,27 148,27 150,18 152,27 156,27 160,5 164,48 168,10 172,27 176,27
                    200,27 208,27 210,18 212,27 216,27 220,5 224,48 228,10 232,27 236,27
                    260,27 268,27 270,18 272,27 276,27 280,5 284,48 288,10 292,27 296,27
                    320,27 328,27 330,18 332,27 336,27 340,5 344,48 348,10 352,27 356,27
                    380,27 388,27 390,18 392,27 396,27 400,5 404,48 408,10 412,27 416,27
                    440,27 448,27 450,18 452,27 456,27 460,5 464,48 468,10 472,27 476,27
                    500,27 508,27 510,18 512,27 516,27 520,5 524,48 528,10 532,27 536,27
                    560,27 568,27 570,18 572,27 576,27 580,5 584,48 588,10 592,27 640,27"/>
        <text x="8" y="14" font-size="9" fill="#E74C3C" font-family="monospace" opacity="0.8">ECG MONITOR</text>
        <circle cx="625" cy="12" r="4" fill="#2ECC71" opacity="0.9"/>
        <text x="633" y="16" font-size="9" fill="#2ECC71" font-family="monospace">LIVE</text>
    </svg>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="color:white;">
        <h2 style="color:#E74C3C;font-size:1.3rem;margin-bottom:0.5rem;">❤️ About</h2>
        <p style="color:rgba(255,255,255,0.75);font-size:0.85rem;line-height:1.6;">
            This app predicts cardiovascular disease risk using a Logistic Regression model
            trained on the <strong style="color:white;">Heart Failure Prediction Dataset</strong>
            (918 patients, Kaggle).
        </p>
        <hr style="border-color:rgba(255,255,255,0.1);margin:1rem 0;">
        <p style="color:rgba(255,255,255,0.5);font-size:0.75rem;">
            ⚠️ For educational purposes only.<br>Not a substitute for medical advice.
        </p>
        <hr style="border-color:rgba(255,255,255,0.1);margin:1rem 0;">
        <p style="color:rgba(255,255,255,0.4);font-size:0.72rem;">
            Developed by <strong style="color:rgba(255,255,255,0.7);">Yashasvi Tomar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="section-header">👤 Patient Demographics</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (years)", 18, 100, 40)
with col2:
    sex = st.selectbox("Biological Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")

st.markdown('<p class="section-header">❤️ Cardiovascular Markers</p>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
        format_func=lambda x: {"ATA":"ATA — Atypical Angina","NAP":"NAP — Non-Anginal","TA":"TA — Typical Angina","ASY":"ASY — Asymptomatic"}[x])
with col4:
    resting_bp = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=250, value=120)
with col5:
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)

col6, col7, col8 = st.columns(3)
with col6:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col7:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"],
        format_func=lambda x: {"Normal":"Normal","ST":"ST Abnormality","LVH":"LVH"}[x])
with col8:
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], format_func=lambda x: "Yes" if x == "Y" else "No")

st.markdown('<p class="section-header">📈 Stress Test Results</p>', unsafe_allow_html=True)
col9, col10, col11 = st.columns(3)
with col9:
    max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
with col10:
    oldpeak = st.slider("Oldpeak — ST Depression (mm)", 0.0, 6.0, 1.0, step=0.1)
with col11:
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"],
        format_func=lambda x: {"Up":"Up — Normal","Flat":"Flat — Borderline","Down":"Down — Abnormal"}[x])

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("❤️ Analyse Cardiovascular Risk")

if predict:
    raw_input = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
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
    pct = round(probability * 100, 1)
    conf = round(pct if pct >= 50 else 100 - pct, 1)

    if prediction == 1:
        tier = "High" if pct >= 65 else "Moderate"
    else:
        tier = "Low"

    advice = {
        "High": [
            "Consult a cardiologist promptly — do not delay professional evaluation.",
            "Discuss medication options (statins, beta-blockers) with your physician.",
            "Adopt a heart-healthy diet: reduce sodium, saturated fat, and processed foods.",
            "Begin supervised, low-intensity cardiac rehab before increasing exercise.",
        ],
        "Moderate": [
            "Prioritise a Mediterranean-style diet rich in fibre, healthy fats, and vegetables.",
            "Aim for 150 min/week of moderate aerobic activity — walking, cycling, swimming.",
            "Monitor blood pressure and cholesterol every 6 months.",
            "Manage stress through mindfulness, adequate sleep, and social connection.",
        ],
        "Low": [
            "Great indicators — keep up your current heart-healthy habits.",
            "Maintain regular physical activity: 150+ min/week of moderate exercise.",
            "Continue a balanced diet low in saturated fat and high in plant-based foods.",
            "Annual cardiovascular check-ups remain important even at low risk.",
        ],
    }

    dot_color   = {"High":"#C0392B","Moderate":"#D68910","Low":"#1D8348"}[tier]
    header_cls  = {"High":"result-header-high","Moderate":"result-header-mod","Low":"result-header-low"}[tier]
    score_cls   = {"High":"big-score-high","Moderate":"big-score-mod","Low":"big-score-low"}[tier]
    verdict_cls = {"High":"verdict-high","Moderate":"verdict-mod","Low":"verdict-low"}[tier]
    badge_cls   = {"High":"tier-badge-high","Moderate":"tier-badge-mod","Low":"tier-badge-low"}[tier]
    verdict_msg = {
        "High":"⚠️ Elevated cardiovascular risk detected",
        "Moderate":"🔶 Moderate risk — action recommended",
        "Low":"✅ Low risk — keep up the good work"
    }[tier]
    tier_label  = {"High":"🔴 High Risk","Moderate":"🟡 Moderate Risk","Low":"🟢 Low Risk"}[tier]
    tier_sub    = {"High":"urgent review","Moderate":"monitor closely","Low":"maintain habits"}[tier]

    advice_items = "".join([
        f'<div class="advice-item">'
        f'<div class="advice-dot" style="background:{dot_color}"></div>'
        f'<span>{a}</span></div>'
        for a in advice[tier]
    ])

    gauge_pct = int(pct)
    gauge_color = dot_color

    st.markdown(f"""
    <div class="result-card">
        <div class="{header_cls}">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;">
                <div>
                    <div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#999;margin-bottom:4px;">Risk Score</div>
                    <div class="{score_cls}">{pct}%</div>
                    <div class="{verdict_cls}">{verdict_msg}</div>
                </div>
                <div class="{badge_cls}">{tier_label}</div>
            </div>
            <div style="margin-top:1rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:#999;margin-bottom:5px;">
                    <span>0%</span><span>Low</span><span>Moderate</span><span>High</span><span>100%</span>
                </div>
                <div style="height:8px;border-radius:99px;background:#e5e5e5;overflow:hidden;">
                    <div style="width:{gauge_pct}%;height:100%;border-radius:99px;background:{gauge_color};transition:width 1s;"></div>
                </div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-label">Risk probability</div>
                <div class="metric-value" style="color:{dot_color}">{pct}%</div>
                <div class="metric-sub">model output</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{conf}%</div>
                <div class="metric-sub">prediction certainty</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Risk category</div>
                <div class="metric-value" style="color:{dot_color}">{tier}</div>
                <div class="metric-sub">{tier_sub}</div>
            </div>
        </div>
        <div class="advice-block">
            <div class="advice-title">Personalised recommendations</div>
            {advice_items}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    ❤️ Heart Disease Risk Predictor &nbsp;·&nbsp; Developed by Yashasvi Tomar &nbsp;·&nbsp; Educational use only
</div>
""", unsafe_allow_html=True)
