import { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, FileText, AlertTriangle, Clock, Download, Search, Lock, BarChart2 } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';
import html2pdf from 'html2pdf.js';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");

  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return localStorage.getItem("isLoggedIn") === "true";
  });
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");


  const [history, setHistory] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");

  const [analyticsData, setAnalyticsData] = useState(null);

  const formatSummary = (summaryStr) => {
    return summaryStr || "No findings detected";
  };


  // Helper pentru a asocia fiecărei afecțiuni o tematică de culoare și o iconiță medicală
  const getFindingColor = (type) => {
    const t = type.toLowerCase();
    if (t.includes('caries') || t.includes('cavity') || t.includes('fracture') || t.includes('lesion')) {
      return { bg: '#fee2e2', text: '#ef4444', icon: '🚨' }; // Roșu de Alertă
    }
    if (t.includes('implant') || t.includes('restoration') || t.includes('filling')) {
      return { bg: '#dbeafe', text: '#3b82f6', icon: '⚙️' }; // Albastru/Tehnic (Intervenție umană neutră)
    }
    return { bg: '#f3f4f6', text: '#6b7280', icon: 'ℹ️' }; // Gri pentru restul
  };


  const fetchHistory = async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/history');
      setHistory(res.data);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  const fetchAnalytics = async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/analytics');
      setAnalyticsData(res.data);
    } catch (error) {
      console.error("Error fetching analytics:", error);
    }
  };


  useEffect(() => {
    fetchHistory();
    fetchAnalytics();
  }, []);

  const [isSignUP, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [isForgotPassword, setIsForgotPassword] = useState(false);
  const [resetToken, setResetToken] = useState("");

  useEffect(() => {
    // Verificăm dacă suntem pe link-ul de Resetare de la Mail
    const params = new URLSearchParams(window.location.search);
    const token = params.get("token");
    if (token) {
      setResetToken(token);
    }
  }, []);

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:8000/api/auth/login', { username, password });
      if (res.status === 200) {
        localStorage.setItem("isLoggedIn", "true");
        setIsLoggedIn(true);
      }
    } catch (error) {
      alert("Invalid credentials! " + (error.response?.data?.detail || ""));
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:8000/api/auth/signup', { username, email, password });
      if (res.status === 200) {
        alert("Account created successfully! Please login.");
        setIsSignUp(false);
      }
    } catch (error) {
      alert("Registration failed! " + (error.response?.data?.detail || ""));
    }
  };

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:8000/api/auth/forgot-password', { email });
      alert(res.data.message);
      setIsForgotPassword(false);
    } catch (error) {
      alert("Error: " + (error.response?.data?.detail || "Nu am putut procesa cererea"));
    }
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:8000/api/auth/reset-password', { token: resetToken, new_password: password });
      alert("Password changed successfully!");
      setResetToken("");
      window.history.pushState({}, document.title, "/");
    } catch (error) {
      alert("Failed to reset password: " + (error.response?.data?.detail || ""));
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("isLoggedIn");
    setIsLoggedIn(false);
    setUsername("");
    setPassword("");
    setEmail("");
  };


  const handleDownloadPDF = () => {
    const element = document.getElementById('report-pdf-content');
    const opt = {
      margin: 10,
      filename: `Medical_Report_${file?.name || 'Patient'}.pdf`,
      image: { type: 'jpeg', quality: 1 },
      html2canvas: { scale: 4, useCORS: true, letterRendering: true },
      jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };
    html2pdf().set(opt).from(element).save();
  };

  const handleSelectHistory = async (reportId) => {
    try {
      setLoading(true);
      const response = await axios.get(`http://localhost:8000/api/report/${reportId}`);
      setReport(response.data);

      // Sincronizăm căsuțele de input cu numele și vârsta pacientului selectat din istoric
      if (response.data.patient_name && response.data.patient_name !== 'Unknown Patient') {
        setPatientName(response.data.patient_name);
      } else {
        setPatientName('');
      }

      if (response.data.patient_age) {
        setPatientAge(response.data.patient_age);
      } else {
        setPatientAge('');
      }

      setFile(null);
      setPreviewUrl(null);
    } catch (error) {
      console.error("Error selecting history:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleNewAnalysis = () => {
    setReport(null);
    setFile(null);
    setPreviewUrl(null);
    setPatientName("");
    setPatientAge("");
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setReport(null);

    if (selectedFile) {
      const imageUrl = URL.createObjectURL(selectedFile);
      setPreviewUrl(imageUrl);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select an image!");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_name', patientName || 'Unknown Patient');
    formData.append('patient_age', patientAge || 0);

    try {
      const response = await axios.post('http://localhost:8000/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setReport(response.data);
      fetchHistory();
    } catch (error) {
      console.error("Error uploading image", error);
      alert("An error occurred while connecting to the AI server.");
    } finally {
      setLoading(false);
    }
  };

  // AFIȘAREA ECRANULUI DE LOGIN
  if (!isLoggedIn) {
    return (
      <div className="login-wrapper">
        <div className="login-card">
          <Lock size={60} style={{ marginBottom: '16px' }} />
          <h2 className="login-title">Secure Medical Access</h2>
          <p className="login-subtitle">AI-Powered Dental Diagnostic Platform</p>
          {resetToken ? (
            /* ===== RESET PASSWORD VIEW ===== */
            <>
              <p style={{ textAlign: "center", marginBottom: "20px" }}>Introdu noua parolă pentru contul tău.</p>
              <form onSubmit={handleResetPassword}>
                <input
                  type="password"
                  placeholder="New Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="login-input"
                  required
                />
                <button type="submit" className="login-btn">
                  Set New Password
                </button>
              </form>
            </>
          ) : isForgotPassword ? (
            /* ===== FORGOT PASSWORD VIEW ===== */
            <>
              <p style={{ textAlign: "center", marginBottom: "20px", color: 'rgba(255,255,255,0.6)' }}>
                Enter your registered Email to receive reset instructions.
              </p>
              <form onSubmit={handleForgotPassword}>
                <input
                  type="email"
                  placeholder="Doctor Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="login-input"
                  required
                />
                <button type="submit" className="login-btn" style={{ background: '#f59e0b' }}>
                  Send Reset Link
                </button>
                <div style={{ textAlign: 'center', marginTop: '15px' }}>
                  <a href="#" onClick={(e) => { e.preventDefault(); setIsForgotPassword(false); }} style={{ color: '#3c7d92', textDecoration: 'none' }}>
                    Back to Login
                  </a>
                </div>
              </form>
            </>
          ) : (
            /* ===== LOGIN / REGISTER VIEW ===== */
            <>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '24px', justifyContent: 'center', background: 'rgba(255,255,255,0.08)', borderRadius: '12px', padding: '5px' }}>
                <button
                  onClick={() => setIsSignUp(false)}
                  style={{ flex: 1, padding: '9px 16px', background: !isSignUP ? 'linear-gradient(135deg,#3c7d92,#1ea8c8)' : 'transparent', color: 'white', border: 'none', borderRadius: '9px', cursor: 'pointer', fontWeight: !isSignUP ? '700' : '500', fontSize: '0.9rem', transition: 'all 0.25s', boxShadow: !isSignUP ? '0 2px 10px rgba(30,168,200,0.4)' : 'none', fontFamily: 'inherit' }}
                >
                  Login
                </button>
                <button
                  onClick={() => setIsSignUp(true)}
                  style={{ flex: 1, padding: '9px 16px', background: isSignUP ? 'linear-gradient(135deg,#3c7d92,#1ea8c8)' : 'transparent', color: 'white', border: 'none', borderRadius: '9px', cursor: 'pointer', fontWeight: isSignUP ? '700' : '500', fontSize: '0.9rem', transition: 'all 0.25s', boxShadow: isSignUP ? '0 2px 10px rgba(30,168,200,0.4)' : 'none', fontFamily: 'inherit' }}
                >
                  Register
                </button>
              </div>

              <form onSubmit={isSignUP ? handleSignup : handleLogin}>
                <input
                  type="text"
                  placeholder="Doctor Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="login-input"
                  required
                />

                {isSignUP && (
                  <input
                    type="email"
                    placeholder="Doctor Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="login-input"
                    required
                  />
                )}

                <input
                  type="password"
                  placeholder="Doctor Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="login-input"
                  required
                />
                <button type="submit" className="login-btn">
                  {isSignUP ? "Create Account" : "Access Platform"}
                </button>

                {!isSignUP && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '15px' }}>
                    <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.9rem', margin: 0 }}>
                      Forgot pass?
                    </p>
                    <a href="#" onClick={(e) => { e.preventDefault(); setIsForgotPassword(true); }} style={{ color: '#4dd6f0', textDecoration: 'none', fontSize: '0.9rem', fontWeight: 'bold' }}>
                      Reset Here
                    </a>
                  </div>
                )}
              </form>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="layout-wrapper">
      {/* SIDEBAR WITH HISTORY AND SEARCH */}
      <div className="sidebar">
        
        {/* NEW ANALYSIS BUTTON */}
        <button 
          onClick={handleNewAnalysis}
          style={{
            width: '100%',
            background: 'var(--accent)',
            color: '#0b1120',
            border: 'none',
            padding: '12px 15px',
            borderRadius: '10px',
            fontWeight: '700',
            fontSize: '0.9rem',
            cursor: 'pointer',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            gap: '8px',
            marginBottom: '20px',
            boxShadow: '0 4px 12px rgba(56, 189, 248, 0.25)',
            transition: 'all 0.2s',
            fontFamily: 'inherit'
          }}
          onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
          onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
        >
          <span style={{ fontSize: '1.2rem', lineHeight: '1' }}>+</span> New AI Analysis
        </button>

        <h2 className="sidebar-title">
          <Clock size={20} />
          Analysis History
        </h2>

        {/* SEARCH BOX */}
        <div className="search-container">
          <Search className="search-icon" size={18} />
          <input
            type="text"
            placeholder="Search patient..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
        </div>
        {/* PARTEA DE STATISTICI / ANALYTICS */}
        {analyticsData && analyticsData.total_patients > 0 && (
          <div style={{ marginBottom: '20px', padding: '15px', background: 'var(--card)', borderRadius: '12px', border: '1px solid var(--sidebar-border)' }}>
            <h3 style={{ fontSize: '13px', color: 'var(--sidebar-text)', marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <BarChart2 size={16} color="var(--sidebar-muted)" /> Clinic Overview ({analyticsData.total_patients} Patients)
            </h3>

            <div style={{ width: '100%', height: '180px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={analyticsData.chart_data}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={70}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {/* Dam culori pastelate bolilor pentru aspect premium */}
                    {analyticsData.chart_data.map((entry, index) => {
                      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
                      return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                    })}
                  </Pie>
                  <Tooltip
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}


        {history.length === 0 ? (
          <p className="no-history">No patients yet.</p>
        ) : (
          <div className="history-list">
            {history
              .filter(record =>
                (record.patient_name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
                record.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
                record.summary.toLowerCase().includes(searchTerm.toLowerCase())
              )
              .map((record) => (
                <div
                  key={record.id}
                  className="history-card"
                  onClick={() => handleSelectHistory(record.id)}
                  style={{ cursor: 'pointer' }}
                >
                  <span className="history-name">📄 {record.patient_name || record.filename}</span>
                  <span className="history-summary">{formatSummary(record.summary)}</span>
                  <span className="history-date">
                    🕒 {new Date(record.created_at).toLocaleString('en-US')}
                  </span>
                </div>
              ))}
          </div>
        )}

        {/* LOGOUT BUTTON */}
        <div style={{ marginTop: 'auto', paddingTop: '20px', textAlign: 'center' }}>
          <button onClick={handleLogout} className="logout-btn">
            Log Out Clinic Account
          </button>
        </div>
      </div>

      {/* MAIN CONTENT (Upload & Analysis) */}
      <div className="main-content">
        <div className="app-container">
          <h1 className="title">🦷 Dental AI Assistant</h1>

          {/* Upload Section */}
          <div className="upload-card">
            <UploadCloud size={56} className="upload-icon" />
            <h3 className="upload-title">Upload patient radiograph</h3>
            <p className="upload-subtitle">Accepted formats: PNG, JPG, JPEG</p>

            {/* Patient Data Input */}
            <div className="patient-inputs-container">
              <input
                type="text"
                placeholder="Patient Name (e.g. John Doe)"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="patient-input name-input"
              />
              <input
                type="number"
                placeholder="Age"
                value={patientAge}
                onChange={(e) => setPatientAge(e.target.value)}
                className="patient-input age-input"
              />
            </div>

            {/* File Drop Zone */}
            <div className="file-drop-zone">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="file-input"
              />
            </div>

            {file && (
              <div style={{ marginTop: '20px' }}>
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Radiograph preview"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '300px',
                      borderRadius: '16px',
                      marginBottom: '20px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                    }}
                  />
                )}

                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="primary-btn"
                  >
                    {loading ? 'AI Analyzing...' : 'Generate Report'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Report Section */}
          {report && (
            <div style={{ marginTop: '50px' }}>
              <div className="report-card" id="report-pdf-content" style={{ marginTop: '0' }}>

                <div className="report-header">
                  <FileText size={28} color="#3c7d92" />
                  <h2 className="report-title">AI Medical Report</h2>
                </div>

                <div className="patient-info-box">
                  <p className="patient-info-text"><strong>👤 Patient:</strong> {report.patient_name || 'Unknown Patient'}</p>
                  <p className="patient-info-text"><strong>⏳ Age:</strong> {report.patient_age ? `${report.patient_age} years` : 'Not specified'}</p>
                </div>

                {/* Analyzed Image Section */}
                {report.processed_image_url && (
                  <div style={{ textAlign: 'center', marginBottom: '30px' }}>
                    <h3 style={{ marginBottom: '15px' }}>Analyzed Radiograph</h3>
                    <img
                      src={report.processed_image_url}
                      crossOrigin="anonymous"
                      alt="AI Processed Radiograph"
                      style={{
                        maxWidth: '100%',
                        maxHeight: '400px',
                        borderRadius: '16px',
                        boxShadow: '0 8px 20px rgba(0,0,0,0.3)',
                        border: '1px solid var(--sidebar-border)'
                      }}
                    />
                  </div>
                )}

                <div className="disclaimer-box">
                  <AlertTriangle size={24} color="#b89300" style={{ flexShrink: 0 }} />
                  <p className="disclaimer-text">{report.disclaimer}</p>
                </div>

                <h3>Evaluation Summary</h3>
                <p className="summary-text">{formatSummary(report.summary)}</p>
                <h3>Findings</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '15px' }}>
                  {report.findings.map((f, idx) => {
                    // Reparam virgulele si underscore-urile urate de la backend (ex: Restoration_/_Fillings)
                    const cleanType = f.finding_type.replace(/_/g, ' ').toUpperCase();
                    const style = getFindingColor(cleanType);

                    return (
                      <div key={idx} style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '16px', background: '#ffffff', borderRadius: '12px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.06)', borderLeft: `6px solid ${style.text}`
                      }}>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '1.2rem' }}>{style.icon}</span>
                            <span style={{
                              fontWeight: '700', color: style.text, background: style.bg,
                              padding: '4px 12px', borderRadius: '20px', fontSize: '0.85rem', letterSpacing: '0.5px'
                            }}>
                              {cleanType}
                            </span>
                          </div>
                          <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
                            <strong>Locație:</strong> {f.location}
                          </span>
                        </div>

                        <div style={{ textAlign: 'right' }}>
                          <span style={{
                            background: 'rgba(255,255,255,0.05)', color: 'var(--text-muted)', padding: '4px 8px',
                            borderRadius: '6px', fontSize: '0.8rem', fontWeight: 'bold', border: '1px solid var(--sidebar-border)'
                          }}>
                            {f.severity}
                          </span>
                          <div style={{ fontWeight: '900', color: 'var(--text-main)', marginTop: '6px', fontSize: '1.1rem' }}>
                            {Math.round(f.confidence * 100)}% Match
                          </div>
                        </div>

                      </div>
                    )
                  })}
                </div>

                {/* --- AI TREATMENT PLAN SECTION --- */}
                {report.treatment_plan && (
                  <div style={{ marginTop: '30px' }}>
                    <h3 style={{
                      display: 'flex', alignItems: 'center', gap: '8px',
                      color: 'var(--accent)', marginBottom: '15px'
                    }}>
                      ✨ AI Proposed Treatment Plan
                    </h3>
                    <div style={{
                      background: 'rgba(56, 189, 248, 0.05)',
                      border: '1px solid rgba(56, 189, 248, 0.2)',
                      borderRadius: '12px',
                      padding: '20px',
                      color: 'var(--text-main)',
                      lineHeight: '1.6',
                      fontFamily: 'inherit',
                      fontSize: '0.95rem'
                    }} className="markdown-container">
                      <ReactMarkdown>
                        {report.treatment_plan}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>

              {/* PDF DOWNLOAD BUTTON */}
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: '30px', marginBottom: '40px' }}>
                <button
                  onClick={handleDownloadPDF}
                  className="primary-btn"
                  style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#48bb78', color: 'white' }}
                >
                  <Download size={20} />
                  Download Medical Report (PDF)
                </button>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
