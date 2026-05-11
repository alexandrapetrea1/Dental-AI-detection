
from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from pathlib import Path
from services.dicom_service import process_dicom

router = APIRouter(prefix="/api/dicom", tags=["dicom"])

UPLOAD_DIR = "uploads/dicom_raw"
PROCESSED_DIR = "uploads/radiographs" # Refolosim folderul existent pentru compatibilitate cu AI

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@router.post("/upload")
async def upload_dicom(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.dcm', '.dicom')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .dcm file.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Salvăm fișierul brut
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Procesăm DICOM-ul
        metadata, png_path = process_dicom(file_path, PROCESSED_DIR)
        
        return {
            "status": "success",
            "metadata": metadata,
            "converted_image_url": f"/images/{os.path.basename(png_path)}",
            "original_filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DICOM: {str(e)}")
