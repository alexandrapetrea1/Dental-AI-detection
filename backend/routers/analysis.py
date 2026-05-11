import os
import time
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from services import analysis_service

router = APIRouter(prefix="/api", tags=["analysis"])

@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...), 
    patient_name: str = Form("Necunoscut"),
    patient_age: int = Form(0),
    db: Session = Depends(get_db)
):
    try:
        db_report, model_report, file_location = analysis_service.process_and_save_analysis(
            db=db, 
            file=file, 
            filename=file.filename, 
            patient_name=patient_name, 
            patient_age=patient_age
        )

        return {
            "id": db_report.id,
            "filename": file.filename,
            "patient_name": db_report.patient_name,
            "patient_age": db_report.patient_age,
            "saved_path": file_location,
            "status": "success",
            "findings": model_report.get("findings", []),
            "summary": model_report.get("summary", "Analiza finalizata"),
            "img_width": db_report.img_width,
            "img_height": db_report.img_height,
            "width": db_report.img_width,
            "height": db_report.img_height,
            "processed_image_url": f"http://localhost:8000/images/analyzed_{file.filename}?v={int(time.time())}" if file_location and "analyzed_" in file_location else None,
            "heatmap_image_url": f"http://localhost:8000/images/heatmap_{file.filename}?v={int(time.time())}" if db_report.heatmap_path else None
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ EROARE CRITICA IN ROUTER:\n{error_details}")
        raise HTTPException(status_code=500, detail={"message": str(e), "traceback": error_details})

@router.post("/analyze-normalized/{filename}")
async def analyze_normalized(filename: str, patient_name: str = "Unknown", patient_age: int = 0, db: Session = Depends(get_db)):
    file_location = f"uploads/radiographs/{filename}"
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="Normalized image not found.")
    
    try:
        db_report, model_report, _ = analysis_service.process_and_save_analysis(
            db, None, filename, patient_name, patient_age
        )
        return {
            "status": "success",
            "findings": model_report.get("findings", []),
            "summary": model_report.get("summary", ""),
            "img_width": db_report.img_width,
            "img_height": db_report.img_height,
            "report_id": db_report.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/report/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)):
    import json
    report = analysis_service.get_report_by_id(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Citim findings-urile reale salvate ca JSON
    findings = json.loads(report.findings_json) if report.findings_json else []
    
    processed_url = f"http://localhost:8000/images/analyzed_{report.filename}"
    
    return {
        "id": report.id,
        "filename": report.filename,
        "patient_name": report.patient_name,
        "patient_age": report.patient_age,
        "img_width": report.img_width,
        "img_height": report.img_height,
        "summary": report.summary,
        "treatment_plan": report.treatment_plan,
        "status": "success",
        "processed_image_url": processed_url,
        "heatmap_image_url": f"http://localhost:8000/images/heatmap_{report.filename}" if report.heatmap_path else None,
        "disclaimer": "Diagnostic bazat pe modelul Faster R-CNN (DENTEX Edition).",
        "findings": findings
    }


@router.get("/history")
def get_history(db: Session = Depends(get_db)):
    reports = analysis_service.get_all_history(db)
    return reports


@router.get("/analytics")
def get_dashboard_analytics(db: Session = Depends(get_db)):
    stats = analysis_service.get_analytics(db)
    return stats
