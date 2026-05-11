
import pydicom
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import os

def process_dicom(dicom_path: str, output_folder: str):
    """
    Procesează un fișier DICOM: extrage metadatele și salvează o variantă PNG normalizată.
    """
    ds = pydicom.dcmread(dicom_path)
    
    # 1. Extracție Metadate Tehnice
    metadata = {
        "patient_name": str(ds.get("PatientName", "Unknown")),
        "patient_id": str(ds.get("PatientID", "Unknown")),
        "modality": str(ds.get("Modality", "Unknown")),
        "kvp": str(ds.get("KVP", "N/A")),
        "exposure_time": str(ds.get("ExposureTime", "N/A")),
        "tube_current": str(ds.get("XRayTubeCurrent", "N/A")),
        "exposure": str(ds.get("Exposure", "N/A")),
        "radiation_dose": str(ds.get("RelativeRadiationExposure", "N/A")),
        "manufacturer": str(ds.get("Manufacturer", "Unknown")),
        "model_name": str(ds.get("ManufacturerModelName", "Unknown"))
    }
    
    # 2. Image Processing (Normalize 16-bit to 8-bit)
    print(f"📸 Processing pixels for: {dicom_path}")
    try:
        # For certain compressed DICOMs, we might need to trigger decompression
        pixel_array = ds.pixel_array.astype(float)
    except Exception as e:
        print(f"❌ Decompression error: {str(e)}")
        # If standard decompression fails, we return a clear error
        raise Exception(f"Could not decompress DICOM image. Format might not be supported without extra plugins: {str(e)}")

    print(f"📐 Image shape: {pixel_array.shape}")
    
    # Apply Window Center and Window Width for correct medical contrast
    if "WindowCenter" in ds and "WindowWidth" in ds:
        print("🪟 Applying Windowing (VOI LUT)")
        center = ds.WindowCenter
        width = ds.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue): center = center[0]
        if isinstance(width, pydicom.multival.MultiValue): width = width[0]
        
        low = center - width / 2
        high = center + width / 2
        pixel_array = np.clip(pixel_array, low, high)
        pixel_array = (pixel_array - low) / (high - low)
    else:
        print("⚖️ Falling back to Min-Max normalization")
        pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
    
    # Convert to 8-bit (0-255)
    img_8bit = (pixel_array * 255).astype(np.uint8)
    
    # 3. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_8bit)
    
    # 4. Save PNG
    filename = Path(dicom_path).stem + ".png"
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img_enhanced)
    print(f"✅ DICOM converted to PNG: {output_path}")
    
    return metadata, output_path
