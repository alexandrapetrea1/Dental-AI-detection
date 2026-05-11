from sqlalchemy import Column, Integer, String, DateTime, Text
import datetime
from database import Base

class AnalysisReport(Base):
    __tablename__ = "analysis_reports"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    saved_path = Column(String)
    heatmap_path = Column(String, nullable=True)
    summary = Column(String)
    findings_json = Column(Text, nullable=True)  
    treatment_plan = Column(Text, nullable=True)
    patient_name = Column(String)
    patient_age = Column(Integer)
    img_width = Column(Integer, nullable=True)
    img_height = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
