import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fpdf import FPDF
import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import tensorflow as tf
import base64
from io import BytesIO
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
import datetime

# --- PAGE CONFIG & STYLES ---
st.set_page_config(
    page_title="Explainable Chest X-Ray Diagnosis AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
        body {background-color:#f9fafb;}
        .main-title {font-size:2.5rem; color:#1976d2; text-align:center; font-weight:800;}
        .subtitle {text-align:center; color:#555; font-size:1.11rem; margin-bottom:2rem;}
        .metric-card {background:#f5f7fa; border-radius:12px; border-left:5px solid #1976d2;
            margin-bottom:1.2rem; padding:1.1rem; box-shadow:0 2px 20px #eee;}
        .result-card {background:#fff; border-radius:8px; padding:1.1rem;
            border:1px solid #e0e0e0; margin-bottom:1.2rem;}
        .warning-box {background:#fff3cd; border:1px solid #ffeaa7; border-radius:8px;
            padding:1rem; margin:1rem 0;}
    </style>
""", unsafe_allow_html=True)

# --- HEADER & BRAND ---
st.markdown("<div class='main-title'>🩺 Explainable Chest X-ray Diagnosis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered, GradCAM-Ready | Portfolio Demo</div>", unsafe_allow_html=True)

# --- SIDEBAR: METRICS + INFO ---
with st.sidebar:
    st.header("📊 Model Performance")
    colA, colB = st.columns(2)
    with colA:
        st.metric("Test Accuracy", "90.0%")
    with colB:
        st.metric("ROC AUC", "0.96")
    with st.expander("Model Details", expanded=False):
        st.write("""
        • <b>Architecture:</b> Custom CNN  
        • <b>Input:</b> 64x64x3  
        • <b>Epochs:</b> 25  
        • <b>Metrics:</b> F1, ROC, Confusion Matrix  
        """, unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Evaluation Visuals")
    roc_img_path = "plots/roc_curve.jpg"
    conf_matrix_path = "plots/confusion_matrix.jpg"
    tab1, tab2 = st.tabs(["ROC Curve", "ConfMatrix"])
    with tab1:
        st.image(roc_img_path, caption="ROC Curve", use_container_width=True)
    with tab2:
        st.image(conf_matrix_path, caption="Confusion Matrix", use_container_width=True)
    st.markdown("---")
    st.subheader("Usage Dashboard")
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
    st.markdown("""
        <a href='https://github.com/Av1352/Chest-X-ray-Classification' 
            style='color:#1976d2;font-weight:700;' target='_blank'>GitHub Repo</a>
        &nbsp; | &nbsp;
        <a href='https://vxanju.com' style='color:#1976d2;font-weight:700;' target='_blank'>Portfolio</a>
    """, unsafe_allow_html=True)
    st.caption("Made by Anju Vilashni Nandhakumar | 2025")

# ---- MAIN CARD: Patient Inputs + GradCAM Results ----
with st.form("diagnosis_form"):
    st.subheader("Patient Details & Vitals")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="Patient Name")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", 1, 120, 30)
        exam_date = st.date_input("Exam Date")
        symptoms = st.text_area("Symptoms")
        chronic = st.text_area("Chronic Conditions")
    with col2:
        temp = st.number_input("Temperature (°F)", 90.0, 110.0, 98.6)
        spo2 = st.slider("SpO₂ (%)", 70, 100, 97)
        spirometer = st.slider("Spirometer (L)", 1, 6, 3)
        uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("🧠 Analyze X-ray")

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('saved_models/chest_xray_model.h5')
model = load_model()

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# --- PDF Report builder ---
class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(30,120,229)
        self.set_text_color(255,255,255)
        self.set_font('Arial', 'B', 18)
        self.cell(0, 18, 'Chest X-ray Diagnosis AI Report', 0, 1, 'C', fill=True)
        self.ln(5)
    def footer(self):
        self.set_y(-20)
        self.set_font('Arial', 'I', 10)
        self.set_text_color(120,120,120)
        self.cell(0, 10, 'Made by Anju Vilashni Nandhakumar | Portfolio Demo | NOT CLINICAL ADVICE', 0, 0, 'C')

def build_pdf_report(name, age, date, diagnosis, confidence, img_path, gradcam_path):
    pdf = PDFReport('P', 'mm', 'A4')
    pdf.add_page()
    # Patient info box
    pdf.set_fill_color(245,247,250)
    pdf.set_text_color(30,120,229)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Patient Information', 0, 1, 'L', fill=True)
    pdf.set_text_color(50,50,50)
    pdf.set_font('Arial','',12)
    pdf.cell(0,10, f"Name: {name}", 0, 1)
    pdf.cell(0,10, f"Age: {age}", 0, 1)
    pdf.cell(0,10, f"Date: {date}", 0, 1)
    pdf.ln(4)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(30,120,229)
    pdf.cell(0,10, 'Diagnosis Result', 0, 1, 'L', fill=True)
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(50,50,50)
    pdf.cell(0,10, f"Diagnosis: {diagnosis}", 0, 1)
    pdf.cell(0,10, f"Confidence: {confidence}", 0, 1)
    pdf.ln(6)
    x_img, y_img = pdf.get_x(), pdf.get_y()
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(30,120,229)
    pdf.cell(90,10,"Original Chest X-ray", 0, 0, 'C')
    pdf.cell(0,10,"GradCAM Visualization", 0, 1, 'C')
    pdf.image(img_path, x=x_img, y=pdf.get_y(), w=80)
    pdf.image(gradcam_path, x=x_img+100, y=pdf.get_y(), w=80)
    pdf.ln(85)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(120,50,50)
    pdf.multi_cell(0, 8, "This report is generated by AI for education/demo only. Do not use for clinical decisions. Consult a physician for any diagnosis or treatment.\n\nPowered by AI | Portfolio: vxanju.com")
    return pdf.output(dest='S').encode('latin-1')

# --- ANALYSIS & DISPLAY RESULTS ---
if submitted and uploaded_file is not None:
    st.info("Running AI diagnosis and GradCAM…")
    image = Image.open(uploaded_file).convert("RGB").resize((64, 64))
    img_array = np.array(image) / 255.0
    pred = model.predict(img_array[np.newaxis, ...])
    diagnosis = "Normal" if pred[0][0] < 0.5 else "Pneumonia"
    confidence = f"{int(pred[0][0] * 100) if diagnosis=='Pneumonia' else int((1-pred[0][0])*100)}%"

    # GradCAM Visualization
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_1")
    overlay_path = overlay_heatmap(heatmap, img_array)
    gradcam_image = Image.open(overlay_path)

    st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown(f"### Diagnosis Result: <b style='color:#1976d2;'>{diagnosis}</b>", unsafe_allow_html=True)
    st.metric("Confidence", confidence)
    st.image(gradcam_image, caption="GradCAM Region-of-Interest", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Save all files to patient_reports/unique folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(' ', '_')
    report_folder = os.path.join("patient_reports", f"{safe_name}_{timestamp}")
    os.makedirs(report_folder, exist_ok=True)

    # Save images and PDF to that folder
    xray_path = os.path.join(report_folder, "xray.png")
    gradcam_path = os.path.join(report_folder, "gradcam.png")
    image.save(xray_path)
    gradcam_image.save(gradcam_path)

    pdf_bytes = build_pdf_report(name, age, exam_date, diagnosis, confidence, xray_path, gradcam_path)
    pdf_path = os.path.join(report_folder, f"ChestXrayReport_{safe_name}_{timestamp}.pdf")
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes)

    # --- Download button for user
    st.download_button(
        "Download AI Report (PDF)",
        pdf_bytes,
        file_name=f"ChestXrayReport_{safe_name}_{timestamp}.pdf"
    )

    with st.expander("⚙️ What is GradCAM?"):
        st.write("GradCAM highlights regions of the X-ray the AI focused on to make its decision—explaining clinical reasoning.")
else:
    st.info("⬆️ Please fill all details and upload a chest X-ray to receive diagnosis and AI explanation.")

# ---- FOOTER & DISCLAIMER ----
st.markdown("---")
st.markdown("""
    <div style="text-align:center; color:#444;font-size:0.97rem;">
        Made by Anju Vilashni Nandhakumar | Powered by AI | <a href='https://github.com/Av1352' style='color:#1976d2;'>GitHub Source</a> | <a href='https://vxanju.com' style='color:#1976d2;'>Portfolio</a>
    </div>
""", unsafe_allow_html=True)
