import { useState, useEffect } from 'react';
import axios from 'axios';
import { UploadCloud, FileText, AlertTriangle, Clock, Download, Search, Lock, BarChart2, Zap } from 'lucide-react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';
import html2pdf from 'html2pdf.js';
import { jsPDF } from 'jspdf';
import html2canvas from 'html2canvas';
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
  const [sharedReport, setSharedReport] = useState(null);

  useEffect(() => {
    const path = window.location.pathname;
    if (path.startsWith('/share/')) {
      const reportId = path.split('/').pop();
      axios.get(`http://192.168.1.103:8000/api/report/${reportId}`)
        .then(res => setSharedReport(res.data))
        .catch(err => console.error("Could not load shared report", err));
    }
  }, []);

  if (sharedReport) {
    return (
      <div style={{ background: '#0f172a', minHeight: '100vh', color: '#fff', padding: '20px', fontFamily: 'Inter, sans-serif' }}>
        <div style={{ maxWidth: '600px', margin: '0 auto' }}>
          <header style={{ textAlign: 'center', marginBottom: '30px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: '900' }}>DENTAL<span style={{ color: '#06b6d4' }}>AI</span></h1>
            <p style={{ color: '#94a3b8', fontSize: '14px' }}>Patient Digital Portal</p>
          </header>
          
          <div style={{ background: '#1e293b', borderRadius: '24px', padding: '24px', boxShadow: '0 20px 50px rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)' }}>
            <h2 style={{ fontSize: '18px', marginBottom: '15px' }}>Hello, {sharedReport.patient_name}</h2>
            <p style={{ fontSize: '14px', color: '#94a3b8', marginBottom: '25px' }}>Below are your digital dental results processed by our AI system.</p>
            
            <div style={{ position: 'relative', borderRadius: '16px', overflow: 'hidden', marginBottom: '20px' }}>
              <img src={sharedReport.processed_image_url} style={{ width: '100%', display: 'block' }} alt="Dental X-ray" />
              {sharedReport.findings.map((f, i) => (
                <div key={i} style={{
                  position: 'absolute',
                  border: '1.5px solid #06b6d4',
                  left: `${(f.box[0] / sharedReport.img_width) * 100}%`,
                  top: `${(f.box[1] / sharedReport.img_height) * 100}%`,
                  width: `${((f.box[2] - f.box[0]) / sharedReport.img_width) * 100}%`,
                  height: `${((f.box[3] - f.box[1]) / sharedReport.img_height) * 100}%`,
                  borderRadius: '3px'
                }} />
              ))}
            </div>

            <div style={{ display: 'grid', gap: '12px' }}>
              <h3 style={{ fontSize: '14px', textTransform: 'uppercase', color: '#64748b', letterSpacing: '1px' }}>AI Observations</h3>
              {sharedReport.findings.map((f, i) => (
                <div key={i} style={{ background: 'rgba(255,255,255,0.05)', padding: '12px', borderRadius: '12px', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontWeight: 'bold' }}>{f.finding_type}</span>
                  <span style={{ color: '#06b6d4' }}>{Math.round(f.confidence * 100)}% match</span>
                </div>
              ))}
            </div>
            
            <div style={{ marginTop: '30px', padding: '15px', background: 'rgba(6, 182, 212, 0.1)', borderRadius: '12px', fontSize: '13px', border: '1px solid rgba(6, 182, 212, 0.2)' }}>
              <strong>AI Summary:</strong> {sharedReport.summary}
            </div>
          </div>
          
          <footer style={{ textAlign: 'center', marginTop: '40px', fontSize: '11px', color: '#475569' }}>
            Powered by Dental AI Engine • Secured Clinical Data
          </footer>
        </div>
      </div>
    );
  }

  const [analyticsData, setAnalyticsData] = useState(null);

  const formatSummary = (summaryStr) => {
    return summaryStr || "No findings detected";
  };

  // Helper pentru calcularea cadranului anatomic (Sistemul FDI dinamic)
  const getAnatomicalLocation = (box, imgW, imgH) => {
    // Fallback pentru dimensiuni daca vin sub alte nume
    const w = imgW || report?.width || report?.img_width || 0;
    const h = imgH || report?.height || report?.img_height || 0;

    if (!box || box.length < 4 || w <= 0 || h <= 0) {
      console.log("📍 Quadrant Debug:", { box, w, h });
      return "Dental Arch";
    }
    
    const [x1, y1, x2, y2] = box;
    const centerX = (x1 + x2) / 2;
    const centerY = (y1 + y2) / 2;

    const isUpper = centerY < (h * 0.55);
    const isLeft = centerX > (w * 0.5);

    // FDI Notation Calculation
    let quadrant = "";
    let toothNum = 0;

    if (isUpper && !isLeft) {
      quadrant = "Q1 (Upper Right)";
      // Teeth 18 (far right) to 11 (center)
      const relX = centerX / (w * 0.5); // 0 to 1
      toothNum = 10 + Math.max(1, Math.min(8, Math.ceil(relX * 8)));
    } else if (isUpper && isLeft) {
      quadrant = "Q2 (Upper Left)";
      // Teeth 21 (center) to 28 (far left)
      const relX = (centerX - (w * 0.5)) / (w * 0.5); // 0 to 1
      toothNum = 20 + Math.max(1, Math.min(8, Math.ceil(relX * 8)));
    } else if (!isUpper && isLeft) {
      quadrant = "Q3 (Lower Left)";
      // Teeth 31 (center) to 38 (far left)
      const relX = (centerX - (w * 0.5)) / (w * 0.5); // 0 to 1
      toothNum = 30 + Math.max(1, Math.min(8, Math.ceil(relX * 8)));
    } else {
      quadrant = "Q4 (Lower Right)";
      // Teeth 41 (center) to 48 (far right)
      const relX = centerX / (w * 0.5); // 0 to 1
      toothNum = 40 + Math.max(1, Math.min(8, Math.ceil(relX * 8)));
    }

    return `${quadrant} - Tooth ${toothNum}`;
    return "Dental Arch";
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
    const doc = new jsPDF('p', 'mm', 'a4');
    
    doc.html(element, {
      callback: function (doc) {
        doc.save(`Medical_Report_${patientName || 'Patient'}.pdf`);
      },
      x: 10,
      y: 10,
      width: 190, // Lățimea țintă în PDF (mm)
      windowWidth: 750, // Lățimea de randare (trebuie să fie aproape de lățimea CSS de 720px)
      html2canvas: {
        useCORS: true,
        backgroundColor: '#ffffff',
        logging: false
      }
    });
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

        <a 
          href="http://localhost:5175" 
          target="_blank" 
          rel="noopener noreferrer"
          style={{
            width: '100%',
            background: '#0f172a',
            color: 'white',
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
            boxShadow: '0 4px 12px rgba(15, 23, 42, 0.25)',
            transition: 'all 0.2s',
            textDecoration: 'none',
            fontFamily: 'inherit'
          }}
          onMouseOver={(e) => e.target.style.transform = 'translateY(-2px)'}
          onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
        >
          <Zap size={18} color="#fbbf24" />
          Radiology Technical Portal
        </a>

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
                  <div style={{ position: 'relative', display: 'inline-block' }}>
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
                  </div>
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

          {/* Report Section - ULTRA-CLEAR ISOLATED VIEW */}
          {report && (
            <div className="report-card" id="report-pdf-content" style={{ 
              background: '#ffffff', 
              color: '#000000', 
              padding: '40px',
              width: '720px', 
              margin: '0 auto',
              borderRadius: '0', 
              fontFamily: 'Arial, sans-serif',
              lineHeight: '1.4',
              opacity: '1',
              colorScheme: 'light', // Ignoră Dark Mode-ul browserului
              filter: 'none',
              transform: 'none'
            }}>
                
                {/* MODERN HEADER */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '3px solid #000000', paddingBottom: '15px', marginBottom: '20px' }}>
                  <div>
                    <h1 style={{ color: '#000000', margin: 0, fontSize: '26px', fontWeight: '900', letterSpacing: '-0.5px' }}>DENTAL<span style={{ color: '#06b6d4' }}>AI</span></h1>
                    <p style={{ margin: '2px 0', color: '#000000', fontSize: '12px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '1px' }}>Precision Radiological Diagnostics</p>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ background: '#000000', color: '#ffffff', padding: '3px 10px', borderRadius: '4px', fontSize: '10px', fontWeight: 'bold', marginBottom: '5px' }}>
                      OFFICIAL CLINICAL REPORT
                    </div>
                    <p style={{ margin: 0, fontWeight: '800', color: '#000000', fontSize: '13px' }}>Ref: {report.id || 'N/A'}</p>
                    <p style={{ margin: 0, color: '#000000', fontSize: '12px' }}>
                      {new Date().toLocaleDateString('ro-RO', { day: '2-digit', month: 'long', year: 'numeric' })}
                    </p>
                  </div>
                </div>

                {/* PATIENT & SCANNER INFO */}
                <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '20px', marginBottom: '25px' }}>
                  <div style={{ borderLeft: '3px solid #06b6d4', paddingLeft: '15px' }}>
                    <h4 style={{ margin: '0 0 8px 0', color: '#000000', textTransform: 'uppercase', fontSize: '10px', fontWeight: '800', letterSpacing: '0.5px' }}>Patient Profile</h4>
                    <div style={{ fontSize: '14px', color: '#000000' }}>
                      <p style={{ margin: '2px 0' }}><strong>Full Name:</strong> {report.patient_name || 'Anonymous Patient'}</p>
                      <p style={{ margin: '2px 0' }}><strong>Age:</strong> {report.patient_age || '--'} Years old</p>
                    </div>
                  </div>
                  <div style={{ background: '#f1f5f9', padding: '10px 15px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                    <h4 style={{ margin: '0 0 8px 0', color: '#000000', textTransform: 'uppercase', fontSize: '10px', fontWeight: '800' }}>Analysis Metadata</h4>
                    <div style={{ fontSize: '12px', color: '#000000' }}>
                      <p style={{ margin: '2px 0' }}><strong>System:</strong> Faster R-CNN + Eigen-CAM XAI</p>
                      <p style={{ margin: '2px 0' }}><strong>Scan Type:</strong> Panoramic Digital OPG</p>
                    </div>
                  </div>
                </div>

                {/* VISUAL EVIDENCE SECTION - STACKED FOR MAXIMUM DETAIL */}
                <div style={{ marginBottom: '30px' }}>
                  <div style={{ marginBottom: '20px', pageBreakInside: 'avoid', breakInside: 'avoid' }}>
                    <h4 style={{ fontSize: '12px', color: '#64748b', marginBottom: '10px', textTransform: 'uppercase', fontWeight: '800', letterSpacing: '0.5px', borderLeft: '3px solid #0f172a', paddingLeft: '10px' }}>
                      1. AI Diagnostic Analysis (Primary View)
                    </h4>
                    <div style={{ position: 'relative', overflow: 'hidden', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                      <img
                        src={report.processed_image_url}
                        crossOrigin="anonymous"
                        alt="Primary Analysis"
                        style={{ width: '100%', maxHeight: '350px', objectFit: 'contain', display: 'block' }}
                      />
                      {/* SMALL DISCRETE NUMBERED BADGES */}
                      {report.findings.map((f, i) => (
                        <div key={i} style={{
                          position: 'absolute',
                          left: `${(f.box[0] / report.img_width) * 100}%`,
                          top: `${(f.box[1] / report.img_height) * 100}%`,
                          pointerEvents: 'none'
                        }}>
                          <div style={{
                            position: 'absolute',
                            top: '-11px',
                            left: '-7px',
                            background: 'rgba(0,0,0,0.6)',
                            color: '#fff',
                            fontSize: '8px',
                            fontWeight: 'bold',
                            width: '12px',
                            height: '12px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            borderRadius: '50%',
                            border: '0.5px solid #fff',
                            zIndex: 10
                          }}>
                            {i+1}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{ pageBreakInside: 'avoid', breakInside: 'avoid' }}>
                    <h4 style={{ fontSize: '12px', color: '#06b6d4', marginBottom: '10px', textTransform: 'uppercase', fontWeight: '800', letterSpacing: '0.5px', borderLeft: '3px solid #06b6d4', paddingLeft: '10px' }}>
                      2. Clinical Evidence (AI Attention Heatmap)
                    </h4>
                    <div style={{ position: 'relative', overflow: 'hidden', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
                      <img
                        src={report.heatmap_image_url}
                        crossOrigin="anonymous"
                        alt="Evidence Map"
                        style={{ width: '100%', maxHeight: '350px', objectFit: 'contain', display: 'block' }}
                      />
                    </div>
                  </div>
                </div>

                {/* FINDINGS TABLE */}
                <h3 style={{ fontSize: '18px', fontWeight: '800', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '12px', height: '12px', background: '#06b6d4', borderRadius: '2px' }}></div>
                  Diagnostic Observations
                </h3>
                <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: '0 8px', marginBottom: '40px' }}>
                  <thead>
                    <tr style={{ textAlign: 'left', color: '#64748b', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px' }}>
                      <th style={{ padding: '0 15px' }}>#</th>
                      <th style={{ padding: '0 15px' }}>Anomaly</th>
                      <th style={{ padding: '0 15px' }}>Location</th>
                      <th style={{ padding: '0 15px' }}>Confidence</th>
                      <th style={{ padding: '0 15px' }}>Recommendation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.findings.map((f, index) => (
                      <tr key={index} style={{ background: '#f8fafc', borderRadius: '10px' }}>
                        <td style={{ padding: '15px', fontWeight: '800', color: '#64748b', borderTopLeftRadius: '10px', borderBottomLeftRadius: '10px' }}>
                          {index + 1}
                        </td>
                        <td style={{ padding: '15px 12px', fontWeight: '700', fontSize: '14px' }}>
                          {f.finding_type.split(':')[1] || f.finding_type}
                        </td>
                        <td style={{ padding: '15px 12px', fontSize: '13px' }}>
                          {getAnatomicalLocation(f.box)}
                        </td>
                        <td style={{ padding: '15px 12px', fontSize: '13px', fontWeight: '600', color: '#06b6d4' }}>
                          {Math.round(f.confidence * 100)}%
                        </td>
                        <td style={{ padding: '15px 12px', borderRadius: '0 8px 8px 0', fontSize: '13px', color: '#475569' }}>
                          {f.recommendation || 'Clinical review advised.'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* AI TREATMENT STRATEGY */}
                {report.treatment_plan && (
                  <div style={{ padding: '25px', background: '#0f172a', borderRadius: '16px', color: '#fff' }}>
                    <h3 style={{ margin: '0 0 15px 0', fontSize: '18px', fontWeight: '800', color: '#06b6d4' }}>✨ AI-Powered Treatment Strategy</h3>
                    <div style={{ fontSize: '14px', color: '#cbd5e1', lineHeight: '1.7' }} className="markdown-container">
                      <ReactMarkdown>{report.treatment_plan}</ReactMarkdown>
                    </div>
                </div>
                )}

                {/* PREMIUM QR CODE SECTION */}
                <div style={{ 
                  marginTop: '40px', 
                  paddingTop: '20px', 
                  borderTop: '1px solid #e2e8f0', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between' 
                }}>
                  <div style={{ maxWidth: '400px' }}>
                    <h4 style={{ margin: '0 0 5px 0', fontSize: '13px', fontWeight: '800', color: '#000000' }}>Patient Digital Access</h4>
                    <p style={{ margin: 0, fontSize: '11px', color: '#64748b' }}>
                      Scan this code to view your radiological results and AI-enhanced imagery securely on your mobile device.
                    </p>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <img 
                      src={`https://api.qrserver.com/v1/create-qr-code/?size=80x80&data=${encodeURIComponent(`http://192.168.1.103:5173/share/${report.id}`)}`} 
                      alt="Report QR Code"
                      style={{ border: '4px solid #fff', boxShadow: '0 2px 10px rgba(0,0,0,0.1)', borderRadius: '4px' }}
                    />
                    <p style={{ margin: '5px 0 0 0', fontSize: '9px', fontWeight: 'bold', color: '#000000' }}>SCAN FOR MOBILE VIEW</p>
                  </div>
                </div>

                {/* FOOTER */}
                <div style={{ marginTop: '30px', textAlign: 'center', fontSize: '9px', color: '#94a3b8' }}>
                  This report is generated by Dental AI Assistant. Clinical correlation by a licensed professional is required.
                </div>

                <div style={{ marginTop: '50px', paddingTop: '25px', borderTop: '1px solid #e2e8f0', fontSize: '10px', color: '#94a3b8', textAlign: 'justify', lineHeight: '1.4' }}>
                  <p><strong>LEGAL DISCLAIMER:</strong> This report is generated by a clinical-grade Artificial Intelligence system. It is intended to assist dental professionals by highlighting potential areas of interest. Final diagnosis and treatment decisions must be made by a qualified dentist based on a physical examination and comprehensive clinical history.</p>
                </div>
              </div>
          )}

          {/* PDF DOWNLOAD BUTTON */}
          {report && (
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
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
