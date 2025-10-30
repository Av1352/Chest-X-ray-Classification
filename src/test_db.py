from src.database import (
    add_report, get_all_reports, get_report_by_code,
    create_patient_code, get_initials
)

# Create a test patient code and dummy report
name = "John Doe"
initials = get_initials(name)
code = create_patient_code("Normal", initials)
print(f"Generated code: {code}")

# Add a test report
report = add_report(
    patient_code=code,
    patient_name=name,
    gender="Male",
    age=34,
    date="2025-10-29",
    temperature=98.6,
    spo2=97.0,
    spirometer=3.7,
    blood_pressure="120/80",
    heart_rate=76,
    symptoms="None",
    chronic_conditions="None",
    image_path="images/test_image.jpeg",
    prediction="Normal",
    confidence=0.98,
    gradcam_path="plots/gradcam_test_image.jpg",
    recommendation="Routine follow-up in 1 month"
)
print(f"Report added with ID: {report.id}")

# Fetch all reports
all_reports = get_all_reports()
print(f"Total reports in DB: {len(all_reports)}")
for rpt in all_reports:
    print(rpt.patient_code, rpt.prediction, rpt.confidence)

# Fetch by code
found = get_report_by_code(code)
print(f"Found by code: {found.patient_name}, Prediction: {found.prediction}, Confidence: {found.confidence}")
