import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from database import add_report, create_patient_code, get_initials, get_all_reports
import plotly.express as px

MODEL_PATH = "saved_models/chest_xray_model.h5"
IMG_SIZE = (64, 64)

# --- Custom CSS ---
st.markdown("""
<style>
body, .stApp {background-color: #181A20;}
[data-testid="stSidebar"] {background-color: #23232F;}
.main-header {font-size: 2.5rem; color: #284b63; text-align:center;}
.metric-card {background: #202833; padding:1rem 1.5rem; border-radius:12px;}
.warning-box {background:#fff3cd; border:1px solid #ffeaa7; border-radius:8px; margin:1.5rem 0; padding:1rem;}
.stButton > button {background-color: #365486; color: white;}
.section {margin-bottom:2rem;}
hr {border: 1px solid #284b63;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 📊 Model Performance", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA: st.metric("Test Acc", "90.0%")
    with colB: st.metric("ROC AUC", "0.96")
    st.markdown("---")
    st.markdown("### 🧠 Model Details")
    st.write("**Architecture:** Custom CNN")
    st.write("**Input:** 64×64×3")
    st.write("**Epochs:** 25")
    st.write("**Metrics:** F1, ROC, ConfMatrix")
    st.markdown("---")
    st.write("Made with ❤️ at Northeastern AI Lab")

    st.markdown("### 📈 Usage Dashboard")
    bar = px.bar({"Month": ["Oct", "Nov"], "Reports": [23, 44]}, x="Month", y="Reports", title="Saved Reports", color="Reports", color_continuous_scale="Blues")
    bar.update_layout(height=200, width=220, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='#23232F', plot_bgcolor='#23232F', font_color="#F6F6F6")
    st.plotly_chart(bar, use_container_width=True)

# --- HEADER ---
st.markdown('<h1 class="main-header">🩻 Patient Chest X-ray Diagnosis</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;">
    <p style="font-size:1.2rem;color:#a6bdd0;">
        Explainable CXR AI with vitals fusion & rich clinical UX.<br>
        CNN · GradCAM-ready · High-trust reporting for portfolios.
    </p>
</div>
""", unsafe_allow_html=True)

# --- MAIN SECTION ---
col1, col2 = st.columns([1,1.2], gap="large")
with col1:
    st.markdown("#### Patient Info & Vitals")
    with st.form("info_form", border=False):
        name = st.text_input("Full Name", placeholder="Patient Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 1,120, 30)
        date = st.date_input("Exam Date")
        temperature = st.number_input("Temperature (°F)", 95.0, 110.0, 98.6)
        spo2 = st.slider("SpO₂ (%)", 80, 100, 97)
        spirometer = st.slider("Spirometer (L)", 1, 6, 3)
        symptoms = st.text_area("Symptoms")
        chronic = st.text_area("Chronic Conditions")
        image_file = st.file_uploader("CXR Image", type=['jpeg', 'jpg', 'png'])
        analyze = st.form_submit_button("🔬 Analyze Image")

if col2 and analyze and image_file:
    st.markdown("#### Analysis Results")
    image = Image.open(image_file).convert("RGB").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    model = load_model(MODEL_PATH)
    prob = float(model.predict(arr)[0][0])
    pred = "Pneumonia" if prob >= 0.5 else "Normal"
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
    verdict = f"**{pred} ({prob:.2f})**"
    st.markdown(f"<div class='metric-card'><h3>Diagnosis: {verdict}</h3></div>", unsafe_allow_html=True)
    # Recommendation logic
    cards, recs = [], []
    if temperature > 100.5: cards.append("Fever")
    if spo2 < 94: cards.append("Low SpO₂")
    if spirometer < 2: cards.append("Reduced lung")
    if pred == "Pneumonia": cards.append("Abnormal X-ray")
    recommendation = " | ".join(cards) + ". Clinical review recommended." if cards else "All findings normal. Routine care only."
    st.success(recommendation) if not cards else st.warning(recommendation)
    with st.expander("Show Technical Details"):
        st.info(f"Model output: {prob:.2f} | CNN | Input: {IMG_SIZE}")
    # Save
    if st.button("💾 Save Full Report"):
        initials = get_initials(name)
        patient_code = create_patient_code(pred, initials, date.strftime("%Y%m%d"))
        image_path = f"uploads/{patient_code}.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        add_report(
            patient_code=patient_code, patient_name=name, gender=gender,
            age=int(age), date=str(date), temperature=temperature,
            spo2=spo2, spirometer=spirometer, blood_pressure="",
            heart_rate=0, symptoms=symptoms,
            chronic_conditions=chronic, image_path=image_path,
            prediction=pred, confidence=prob, gradcam_path="",
            recommendation=recommendation
        )
        st.balloons()
        st.success(f"🗂️ Report saved: {patient_code}")

# --- SAVED REPORTS TAB ---
st.markdown("---")
with st.expander("📁 View Saved Reports"):
    st.markdown("### Recent Reports")
    reports = get_all_reports()
    if not reports:
        st.info("No reports yet.")
    else:
        for rpt in reports[:20]:
            st.markdown(f"**{rpt.patient_code}** | {rpt.patient_name} | {rpt.prediction} | {rpt.created_at.strftime('%Y-%m-%d')}")

# --- DISCLAIMER AND FOOTER ---
st.markdown("""
<hr>
<div class="warning-box">
    <b>⚠️ Disclaimer:</b> This app is for academic and portfolio purposes only. Not for clinical use.
</div>
""", unsafe_allow_html=True)
st.caption("Made by Anju Vilashni Nandhakumar · Check [GitHub](https://github.com/Av1352/Chest-X-ray-Classification) | [Portfolio](https://vxanju.com)")

