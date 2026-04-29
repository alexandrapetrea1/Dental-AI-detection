from sqlalchemy import Column, Integer, String, DateTime, Text
import datetime
from database import Base

class AnalysisReport(Base):
    __tablename__ = "analysis_reports"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    saved_path = Column(String)
    summary = Column(String)
    findings_json = Column(Text, nullable=True)  
    treatment_plan = Column(Text, nullable=True)
    patient_name = Column(String)
    patient_age = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
