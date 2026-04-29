from sqlalchemy.orm import Session
from models.analysis_report import AnalysisReport
import shutil
from pathlib import Path
from ml_engine import detect_anomalies
import json 
import os
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

UPLOAD_DIR = Path("uploads/radiographs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def process_and_save_analysis(db: Session, file, filename: str, patient_name: str, patient_age: int):
    # 1. Salvăm fișierul pe disk
    file_location = UPLOAD_DIR / filename
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)

    # 2. Analizăm cu noul motor Faster R-CNN (care returnează findings și calea pozei)
    findings, processed_path = detect_anomalies(str(file_location))

    # 3. Rezumat sub formă de listă (ex: "1: Caries, 2: Deep Caries")
    if findings:
        summary_text = ", ".join([f["finding_type"].replace("#", "") for f in findings])
    else:
        summary_text = "No anomalies detected"
        # === GENERATE TREATMENT PLAN USING GEMINI AI ===
    treatment_plan_text = "No conditions requiring immediate treatment were detected at this time."
    
    if findings:
        # Prepare a summary of the found anomalies
        anomalies_list = "\n".join([f"- {f['finding_type']} with {int(f['confidence']*100)}% confidence" for f in findings])
        
        # Build the medical prompt
        prompt = f"""You are an expert dental professional. 
A patient's dental radiograph has been analyzed by a Computer Vision model, which detected the following anomalies:
{anomalies_list}

Please generate a professional, step-by-step Treatment Plan in Markdown format. 
CRITICAL RULE: You MUST order the treatment plan by clinical priority (urgency). 
1. Address acute infections, severe lesions, or deep caries first (Immediate Priority).
2. Address moderate decay, restorative needs, or fillings (Secondary Priority).
3. Address impacted teeth, implants, or long-term interventions last (Long-term Planning).

Use clear headings and bullet points. Do not include any personal identifiable information.
"""
        try:
            import os
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            treatment_plan_text = response.text
        except Exception as e:
            print("Gemini API Error:", e)
            treatment_plan_text = f"The AI treatment planning service is currently unavailable. Error details: {str(e)}"

    # 4. Salvăm în baza de date
    db_report = AnalysisReport(
    filename=filename,
    saved_path=str(processed_path) if processed_path else str(file_location),
    summary=summary_text,
    findings_json=json.dumps(findings),   
    patient_name=patient_name,
    patient_age=patient_age,
    treatment_plan=treatment_plan_text  
)
    db.add(db_report)
    db.commit()      
    db.refresh(db_report)

    # Returnăm pentru API
    return db_report, {"findings": findings, "summary": summary_text}, str(processed_path)

def get_report_by_id(db: Session, report_id: int):
    return db.query(AnalysisReport).filter(AnalysisReport.id == report_id).first()

def get_all_history(db: Session):
    return db.query(AnalysisReport).order_by(AnalysisReport.created_at.desc()).all()

def get_analytics(db: Session):
    reports = db.query(AnalysisReport).all()
    
    # Pregatim niste variabile pe 0 ca sa tinem socoteala
    total_patients = len(reports)
    disease_counts = {
        "Caries/Cavities": 0,
        "Periapical Lesion": 0,
        "Impacted Tooth": 0
    }
    
    # Parcurgem fiecare raport din baza de date si numaram ce găsim
    for report in reports:
        summary_lower = report.summary.lower()
        if "caries" in summary_lower or "cavity" in summary_lower:
            disease_counts["Caries/Cavities"] += 1
        if "periapical" in summary_lower or "lesion" in summary_lower:
             disease_counts["Periapical Lesion"] += 1
        if "impacted" in summary_lower:
             disease_counts["Impacted Tooth"] += 1
            
    # Pentru Frontend (Recharts), trebuie să le ambalam intr-o lista frumoasa.
    chart_data = [{"name": key, "value": value} for key, value in disease_counts.items() if value > 0]
    
    return {
        "total_patients": total_patients,
        "chart_data": chart_data
    }
