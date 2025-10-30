import streamlit as st
import numpy as np
from PIL import Image
import os
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Explainable CXR Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f5f7fa;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<h1 class="main-header">🩺 Patient Chest X-ray Diagnosis</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="sub-header">
Explainable CXR AI with vitals fusion and rich clinical UX.<br>
CNN · GradCAM-ready · High-trust portfolio reporting.
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📊 Model Performance")
    colA, colB = st.columns(2)
    with colA:
        st.metric("Test Accuracy", "90.0%")
    with colB:
        st.metric("ROC AUC", "0.96")

    st.markdown("---")

    st.subheader("🧠 Model Details")
    st.write("**Architecture:** Custom CNN")
    st.write("**Input Size:** 64×64×3")
    st.write("**Epochs:** 25")
    st.write("**Metrics:** F1, ROC, Confusion Matrix")

    st.markdown("---")

    # === Insert your plots here ===
    st.subheader("📈 Evaluation Visuals")

    # Example placeholders — replace with your actual figure paths or matplotlib objects
    roc_img_path = "plots/roc_curve.jpg"
    conf_matrix_path = "plots/confusion_matrix.jpg"

    tab1, tab2 = st.tabs(["ROC Curve", "ConfMatrix"])
    with tab1:
        st.image(roc_img_path, caption="ROC Curve", use_container_width=True)
    with tab2:
        st.image(conf_matrix_path, caption="Confusion Matrix", use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Usage Dashboard")
    usage_data = {
        'Date': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
        'Reports': [15, 28, 40, 32, 20]
    }
    fig_usage = px.bar(usage_data, x='Date', y='Reports',
                       title='Reports Generated (This Week)',
                       color='Reports', color_continuous_scale='blues')
    fig_usage.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_usage, use_container_width=True)

    st.markdown("---")
    st.caption("💙 Made with care at Northeastern AI Lab")

# ---------------- MAIN CONTENT ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🧾 Patient Info & Vitals")
    name = st.text_input("Full Name", placeholder="Patient Name")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", 1, 120, 30)
    exam_date = st.date_input("Exam Date")
    temp = st.number_input("Temperature (°F)", 90.0, 110.0, 98.6)
    spo2 = st.slider("SpO₂ (%)", 70, 100, 97)
    spirometer = st.slider("Spirometer (L)", 1, 6, 3)
    symptoms = st.text_area("Symptoms")
    chronic = st.text_area("Chronic Conditions")

    st.markdown("### 📤 Upload Chest X-ray Image")
    uploaded_file = st.file_uploader("Choose a Chest X-ray image (JPG/PNG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)
        if st.button("🧠 Analyze X-ray", type="primary"):
            with st.spinner("Running model inference..."):
                # Placeholder logic for model
                st.session_state["diagnosis"] = "Normal"

with col2:
    st.subheader("🩻 Diagnosis Results")
    if "diagnosis" in st.session_state:
        diag = st.session_state["diagnosis"]
        if diag == "Normal":
            st.success("✅ **Normal Chest X-ray detected**")
        else:
            st.error(f"⚠️ **Abnormality Detected: {diag}**")
    else:
        st.info("👆 Upload a chest X-ray and click 'Analyze' to view results.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Clinical Disclaimer</h4>
    <p>This is an educational and research demo — not for real-world diagnosis or treatment decisions.</p>
</div>
""", unsafe_allow_html=True)

st.caption("Made by Anju Vilashni Nandhakumar | Powered by AI | [GitHub](https://github.com/Av1352) | [Portfolio](https://vxanju.com)")
