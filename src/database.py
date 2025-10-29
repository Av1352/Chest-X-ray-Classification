from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import secrets

DATABASE_URL = "sqlite:///./medical_reports.db"
Base = declarative_base()

class PatientReport(Base):
    __tablename__ = "patient_reports"
    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(64), unique=True, index=True)
    patient_name = Column(String(128))
    gender = Column(String(16))
    age = Column(Integer)
    doctor = Column(String(128))
    hospital = Column(String(128))
    date = Column(String(32))
    temperature = Column(Float)
    spo2 = Column(Float)
    spirometer = Column(Float)
    blood_pressure = Column(String(16))
    heart_rate = Column(Integer)
    symptoms = Column(Text)
    chronic_conditions = Column(Text)
    image_path = Column(String(256))
    prediction = Column(String(32))
    confidence = Column(Float)
    gradcam_path = Column(String(256))
    recommendation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_initials(name):
    """Return uppercase initials from a name."""
    return ''.join([word[0] for word in name.strip().split() if word]).upper()

def create_patient_code(prediction, initials, date_str=None):
    """Create a unique patient code based on prediction, initials, date, and random 4-char string."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    rand_str = ''.join(secrets.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(4))
    return f"{prediction.upper()}_{initials}_{date_str}_{rand_str}"

def add_report(**kwargs):
    """Add report to the database."""
    db = SessionLocal()
    report = PatientReport(**kwargs)
    db.add(report)
    db.commit()
    db.refresh(report)
    db.close()
    return report

def get_all_reports():
    """Get all reports from the database (newest first)."""
    db = SessionLocal()
    reports = db.query(PatientReport).order_by(PatientReport.created_at.desc()).all()
    db.close()
    return reports

def get_report_by_code(code):
    """Find a report by its unique code."""
    db = SessionLocal()
    report = db.query(PatientReport).filter(PatientReport.patient_code == code).first()
    db.close()
    return report
