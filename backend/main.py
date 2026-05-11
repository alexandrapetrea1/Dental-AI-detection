from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base

import models.user
import models.analysis_report

from routers import auth_router, analysis_router, dicom_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Dental AI API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="uploads/radiographs"), name="images")

app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(dicom_router)

@app.get("/")
def read_root():
    # Server is alive, checked bcrypt version.
    return {"status": "API MVC is running cu Baza de Date activa!"}
