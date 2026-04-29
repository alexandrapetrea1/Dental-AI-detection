# Dental AI Detection - Bachelor Thesis

A comprehensive system for dental pathology detection using deep learning (Faster R-CNN). This project aims to assist dentists in identifying various dental conditions from X-ray images, including caries, impacted teeth, and periapical lesions.

## 🚀 Features

- **Automated Detection**: Utilizes a Faster R-CNN model trained on dental X-rays.
- **Modern Web Interface**: Built with React and Vite for a smooth user experience.
- **Robust Backend**: FastAPI-powered server for efficient image processing and AI inference.
- **Comprehensive Analysis**: Provides prioritized treatment plans based on detected anomalies.

## 📁 Project Structure

```
.
├── backend/            # FastAPI server, ML engine, and database
├── frontend/           # React + Vite web application
├── dataset/            # Data processing scripts (ignored by git)
└── docs/               # Thesis documentation and bibliography
```

## 🛠️ Tech Stack

- **ML Framework**: PyTorch (Faster R-CNN)
- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: React, Vite, TailwindCSS (optional)
- **Inference**: Google Gemini API (for treatment planning integration)

## 📥 Installation

### Backend
1. Navigate to `backend/`
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Mac/Linux) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `uvicorn main:app --reload`

### Frontend
1. Navigate to `frontend/`
2. Install dependencies: `npm install`
3. Run the development server: `npm run dev`

## 🧠 Model Training

Training scripts are located in the root directory:
- `train_dentex_fasterrcnn.py`: Main training script for the Dentex dataset.
- `format_dentex.py`: Data preprocessing and formatting.

## 📄 License

This project is part of a Bachelor's Thesis (Licență). All rights reserved.
