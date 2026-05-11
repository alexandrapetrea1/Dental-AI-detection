
import React, { useState } from 'react';
import axios from 'axios';
import { Upload, FileText, Activity, Zap, CheckCircle, AlertCircle, Info, Stethoscope } from 'lucide-react';

const API_BASE = 'http://192.168.1.103:8000';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dicomData, setDicomData] = useState(null);
  const [aiResult, setAiResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.toLowerCase().endsWith('.dcm')) {
      setFile(selectedFile);
      setError(null);
      setDicomData(null);
      setAiResult(null);
    } else {
      setError("Please select a valid .dcm file (DICOM)");
    }
  };

  const uploadAndProcess = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_BASE}/api/dicom/upload`, formData);
      setDicomData(res.data);
    } catch (err) {
      setError("Error processing DICOM file. Please check the backend server.");
    } finally {
      setLoading(false);
    }
  };

  const triggerAIAnalysis = async () => {
    if (!dicomData) return;
    setLoading(true);
    setError(null);
    
    try {
      // Extraction of the filename from the URL (e.g. /images/image.png -> image.png)
      const filename = dicomData.converted_image_url.split('/').pop();
      const patientName = dicomData.metadata.patient_name || "Unknown";
      
      const res = await axios.post(`${API_BASE}/api/analyze-normalized/${filename}?patient_name=${patientName}`);
      setAiResult(res.data);
    } catch (err) {
      setError("AI analysis failed. Ensure the server is online.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="portal-container">
      {/* Header */}
      <header className="header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 className="title" style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <Zap size={36} color="#38bdf8" fill="#38bdf8" />
              Dental DICOM Gateway
            </h1>
            <p className="subtitle">Professional Radiology Processing & Normalization Suite</p>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.1)', padding: '8px 16px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 'bold', letterSpacing: '1px' }}>
            TECHNICAL PORTAL
          </div>
        </div>
      </header>

      <div className="grid">
        {/* Left: Upload Section */}
        <div className="card">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Upload color="#38bdf8" />
            Upload Raw Radiograph
          </h2>
          
          <div className="upload-area" onClick={() => document.getElementById('dicom-input').click()}>
            <input 
              type="file" 
              id="dicom-input" 
              style={{ display: 'none' }} 
              onChange={handleFileChange}
              accept=".dcm,.dicom"
            />
            <div style={{ background: 'rgba(56, 189, 248, 0.1)', width: '64px', height: '64px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifySelf: 'center', justifyContent: 'center', margin: '0 auto 16px auto' }}>
              <FileText color="#38bdf8" size={32} />
            </div>
            <p style={{ fontWeight: '600' }}>{file ? file.name : 'Choose a DICOM file'}</p>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '8px' }}>Drag & drop or click to browse</p>
          </div>

          {error && (
            <div style={{ marginTop: '20px', padding: '15px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '12px', color: '#fca5a5', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          <button 
            onClick={uploadAndProcess}
            disabled={!file || loading}
            className="btn-primary"
            style={{ opacity: (!file || loading) ? 0.5 : 1 }}
          >
            {loading ? 'Processing...' : 'Normalize Image & Extract Specs'}
          </button>

          {/* Metadata Display */}
          {dicomData && (
            <div style={{ marginTop: '30px' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 'bold', marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Info size={18} color="#38bdf8" />
                Technical Metadata (DICOM Tags)
              </h3>
              <div className="meta-grid">
                {Object.entries(dicomData.metadata).map(([key, value]) => (
                  <div key={key} className="meta-item">
                    <span className="meta-label">{key.replace('_', ' ')}</span>
                    <span className="meta-value">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Preview & AI Analysis */}
        <div className="card" style={{ minHeight: '400px', display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Activity color="#38bdf8" />
            Preview & AI Verification
          </h2>

          {!dicomData ? (
            <div style={{ flex: 1, border: '2px solid rgba(255,255,255,0.05)', borderRadius: '16px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', background: 'rgba(0,0,0,0.1)' }}>
              <Zap size={48} style={{ opacity: 0.1, marginBottom: '16px' }} />
              <p>Processed image will appear here</p>
            </div>
          ) : (
            <div>
              <div style={{ position: 'relative', marginBottom: '20px' }}>
                <img 
                  src={`${API_BASE}${dicomData.converted_image_url}`} 
                  alt="DICOM Processed" 
                  className="img-preview"
                />
                {/* Visualizer overlay for findings if present */}
                {aiResult && aiResult.findings.map((f, i) => (
                  <div 
                    key={i}
                    style={{
                      position: 'absolute',
                      border: '2px solid #ef4444',
                      background: 'rgba(239, 68, 68, 0.2)',
                      left: `${(f.box[0] / aiResult.img_width) * 100}%`,
                      top: `${(f.box[1] / aiResult.img_height) * 100}%`,
                      width: `${((f.box[2] - f.box[0]) / aiResult.img_width) * 100}%`,
                      height: `${((f.box[3] - f.box[1]) / aiResult.img_height) * 100}%`,
                      pointerEvents: 'none'
                    }}
                  />
                ))}
              </div>
              
              {!aiResult ? (
                <>
                  <div style={{ padding: '15px', background: 'rgba(56, 189, 248, 0.1)', border: '1px solid var(--border-color)', borderRadius: '12px', display: 'flex', gap: '12px', marginBottom: '20px' }}>
                    <CheckCircle color="#38bdf8" style={{ flexShrink: 0 }} />
                    <div>
                      <p style={{ fontWeight: 'bold', color: '#fff', fontSize: '0.9rem' }}>Image Normalized Successfully!</p>
                      <p style={{ fontSize: '0.75rem', color: 'var(--accent-primary)', marginTop: '4px' }}>16-bit raw data has been remapped and CLAHE enhanced.</p>
                    </div>
                  </div>

                  <button 
                    onClick={triggerAIAnalysis} 
                    className="btn-primary" 
                    style={{ background: '#fff', color: '#000', boxShadow: '0 10px 30px rgba(255,255,255,0.1)' }}
                  >
                    <Activity size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                    {loading ? 'Analyzing...' : 'Launch Dental AI Analysis'}
                  </button>
                </>
              ) : (
                <div style={{ padding: '20px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '16px' }}>
                  <h3 style={{ color: '#10b981', display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                    <Stethoscope />
                    Analysis Results: {aiResult.summary}
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    {aiResult.findings.map((f, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', background: 'rgba(255,255,255,0.05)', padding: '10px', borderRadius: '8px' }}>
                        <span style={{ fontWeight: 'bold' }}>{f.finding_type}</span>
                        <span style={{ color: '#10b981' }}>{Math.round(f.confidence * 100)}% Match</span>
                      </div>
                    ))}
                    {aiResult.findings.length === 0 && <p>No anomalies detected in this radiograph.</p>}
                  </div>
                  <a 
                    href={`http://localhost:5173/`} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="btn-secondary"
                    style={{ display: 'block', textAlign: 'center', textDecoration: 'none', marginTop: '20px' }}
                  >
                    View Full Report in Doctor Dashboard
                  </a>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <footer style={{ marginTop: '60px', padding: '40px', borderTop: '1px solid rgba(255,255,255,0.05)', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
        Dental AI Suite - Technical Radiology Module
      </footer>
    </div>
  );
}

export default App;
