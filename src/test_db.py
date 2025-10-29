# test_db.py
from src.database import add_report, create_patient_code, get_initials, get_all_reports

data = {
    "patient_code": create_patient_code("POS", "AJ", "20251029"),
    "patient_name": "Anju V",
    "gender": "Female",
    "age": 30,
    "doctor": "Dr. Test",
    "hospital": "Test Hospital",
    "date": "20251029",
    "temperature": 98.6,
    "spo2": 98.7,
    "spirometer": 2.5,
    "blood_pressure": "120/80",
    "heart_rate": 79,
    "symptoms": "Cough, Fatigue",
    "chronic_conditions": "None",
    "image_path": "sample.jpg",
    "prediction": "Positive",
    "confidence": 0.92,
    "gradcam_path": "sample_gradcam.jpg",
    "recommendation": "Consult Doctor"
}
add_report(**data)
print(get_all_reports())
