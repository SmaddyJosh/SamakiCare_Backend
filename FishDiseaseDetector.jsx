import { useState, useCallback, useRef } from "react";

const API_URL = "http://localhost:8000";


const STATUS_CONFIG = {
  healthy: {
    color: "#00e676",
    bg: "rgba(0,230,118,0.08)",
    border: "rgba(0,230,118,0.3)",
    icon: "🐟",
    label: "Healthy",
    glow: "0 0 30px rgba(0,230,118,0.25)",
  },
  diseased: {
    color: "#ff5252",
    bg: "rgba(255,82,82,0.08)",
    border: "rgba(255,82,82,0.3)",
    icon: "⚠️",
    label: "Disease Detected",
    glow: "0 0 30px rgba(255,82,82,0.25)",
  },
  uncertain: {
    color: "#ffab40",
    bg: "rgba(255,171,64,0.08)",
    border: "rgba(255,171,64,0.3)",
    icon: "🔍",
    label: "Uncertain",
    glow: "0 0 30px rgba(255,171,64,0.25)",
  },
};


const WaterBg = () => (
  <svg style={{ position: "fixed", inset: 0, width: "100%", height: "100%", zIndex: 0, opacity: 0.07 }} preserveAspectRatio="xMidYMid slice">
    <defs>
      <radialGradient id="rg1" cx="30%" cy="40%">
        <stop offset="0%" stopColor="#00b4d8" />
        <stop offset="100%" stopColor="transparent" />
      </radialGradient>
      <radialGradient id="rg2" cx="70%" cy="60%">
        <stop offset="0%" stopColor="#0077b6" />
        <stop offset="100%" stopColor="transparent" />
      </radialGradient>
    </defs>
    <rect width="100%" height="100%" fill="url(#rg1)" />
    <rect width="100%" height="100%" fill="url(#rg2)" />
  </svg>
);

const ConfBar = ({ label, confidence, isTop }) => {
  const pct = (confidence * 100).toFixed(1);
  const isHealthy = label.toLowerCase().includes("healthy") || label.toLowerCase().includes("normal");
  const barColor = isHealthy ? "#00e676" : (isTop ? "#ff5252" : "#546e7a");

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: isTop ? "#e0e0e0" : "#90a4ae", fontSize: 13, fontFamily: "'DM Mono', monospace", textTransform: "capitalize" }}>
          {label.replace(/_/g, " ")}
        </span>
        <span style={{ color: barColor, fontSize: 13, fontWeight: 700, fontFamily: "'DM Mono', monospace" }}>{pct}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
        <div style={{
          height: "100%", borderRadius: 3,
          width: `${pct}%`,
          background: barColor,
          transition: "width 0.8s cubic-bezier(0.16,1,0.3,1)",
          boxShadow: isTop ? `0 0 8px ${barColor}` : "none",
        }} />
      </div>
    </div>
  );
};

export default function FishDiseaseDetector() {
  const [file, setFile]           = useState(null);
  const [preview, setPreview]     = useState(null);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [dragging, setDragging]   = useState(false);
  const inputRef                  = useRef();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Server error");
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message || "Failed to reach API. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null); setPreview(null); setResult(null); setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const cfg = result ? STATUS_CONFIG[result.status] : null;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#080e14",
      color: "#e0e0e0",
      fontFamily: "'DM Sans', system-ui, sans-serif",
      position: "relative",
      overflowX: "hidden",
    }}>
      <WaterBg />

      {/* ── Header ── */}
      <header style={{ position: "relative", zIndex: 1, textAlign: "center", padding: "52px 24px 0" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
          <span style={{ fontSize: 36 }}>🐠</span>
          <h1 style={{
            margin: 0, fontSize: "clamp(26px, 5vw, 42px)",
            fontFamily: "'DM Serif Display', Georgia, serif",
            fontWeight: 400, letterSpacing: "-0.5px",
            background: "linear-gradient(135deg, #90e0ef 0%, #00b4d8 50%, #0077b6 100%)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          }}>AquaHealth AI</h1>
        </div>
        <p style={{ color: "#546e7a", fontSize: 15, margin: 0, letterSpacing: "0.03em" }}>
          Freshwater fish disease detection · Powered by EfficientNet
        </p>
      </header>

 
      <main style={{
        position: "relative", zIndex: 1,
        maxWidth: 720, margin: "40px auto",
        padding: "0 20px 60px",
      }}>

      
        {!result && (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            style={{
              border: `2px dashed ${dragging ? "#00b4d8" : preview ? "#0077b6" : "rgba(255,255,255,0.12)"}`,
              borderRadius: 20,
              padding: "36px 24px",
              textAlign: "center",
              cursor: "pointer",
              background: dragging ? "rgba(0,180,216,0.06)" : "rgba(255,255,255,0.02)",
              transition: "all 0.25s ease",
              backdropFilter: "blur(8px)",
            }}
          >
            {preview ? (
              <div>
                <img src={preview} alt="preview" style={{
                  maxHeight: 260, maxWidth: "100%", borderRadius: 12,
                  boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
                }} />
                <p style={{ marginTop: 14, color: "#546e7a", fontSize: 13 }}>
                  {file?.name} · Click to change
                </p>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: 48, marginBottom: 14 }}>🌊</div>
                <p style={{ color: "#90a4ae", margin: 0, fontSize: 16 }}>
                  <strong style={{ color: "#00b4d8" }}>Drop a fish photo here</strong>
                  <br />
                  <span style={{ fontSize: 13 }}>or click to browse</span>
                </p>
                <p style={{ color: "#37474f", fontSize: 12, marginTop: 10 }}>JPG, PNG, WEBP accepted</p>
              </div>
            )}
            <input ref={inputRef} type="file" accept="image/*" style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])} />
          </div>
        )}

     
        {preview && !result && (
          <button
            onClick={analyze}
            disabled={loading}
            style={{
              display: "block", width: "100%", marginTop: 18,
              padding: "16px 0", borderRadius: 14, border: "none",
              background: loading
                ? "rgba(0,180,216,0.2)"
                : "linear-gradient(135deg, #00b4d8, #0077b6)",
              color: "#fff", fontSize: 16, fontWeight: 700,
              letterSpacing: "0.04em", cursor: loading ? "not-allowed" : "pointer",
              transition: "opacity 0.2s",
              boxShadow: loading ? "none" : "0 4px 20px rgba(0,180,216,0.35)",
            }}
          >
            {loading ? "🔬 Analysing …" : "🔍 Analyse Fish"}
          </button>
        )}

        {/* Error */}
        {error && (
          <div style={{
            marginTop: 18, padding: "14px 18px", borderRadius: 12,
            background: "rgba(255,82,82,0.08)", border: "1px solid rgba(255,82,82,0.25)",
            color: "#ff5252", fontSize: 14,
          }}>⚠️ {error}</div>
        )}

        {/* Result card */}
        {result && cfg && (
          <div style={{
            borderRadius: 24, overflow: "hidden",
            border: `1px solid ${cfg.border}`,
            background: cfg.bg,
            boxShadow: cfg.glow,
            animation: "fadeUp 0.5s ease",
          }}>
            {/* Status header */}
            <div style={{
              padding: "28px 28px 20px",
              borderBottom: `1px solid ${cfg.border}`,
              display: "flex", alignItems: "center", gap: 16,
            }}>
              {preview && (
                <img src={preview} alt="fish"
                  style={{ width: 80, height: 80, borderRadius: 12, objectFit: "cover",
                    border: `2px solid ${cfg.color}`, flexShrink: 0 }} />
              )}
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 24 }}>{cfg.icon}</span>
                  <span style={{
                    fontSize: 20, fontWeight: 700, color: cfg.color,
                    fontFamily: "'DM Serif Display', Georgia, serif",
                  }}>{cfg.label}</span>
                  <span style={{
                    marginLeft: "auto", fontSize: 12, padding: "3px 10px", borderRadius: 20,
                    background: cfg.border, color: cfg.color, fontWeight: 700,
                    fontFamily: "'DM Mono', monospace",
                  }}>{result.confidence_label}</span>
                </div>
                <p style={{ margin: 0, color: "#b0bec5", fontSize: 14, lineHeight: 1.5 }}>
                  {result.message}
                </p>
              </div>
            </div>

            {/* Top prediction */}
            <div style={{ padding: "18px 28px 6px" }}>
              <p style={{ color: "#546e7a", fontSize: 12, textTransform: "uppercase",
                letterSpacing: "0.1em", marginBottom: 14 }}>All predictions</p>
              {result.all_predictions.map((p, i) => (
                <ConfBar key={p.label} label={p.label}
                  confidence={p.confidence} isTop={i === 0} />
              ))}
            </div>

            {/* Actions */}
            <div style={{ padding: "16px 28px 28px", display: "flex", gap: 12 }}>
              <button onClick={reset} style={{
                flex: 1, padding: "12px 0", borderRadius: 12, border: "1px solid rgba(255,255,255,0.1)",
                background: "rgba(255,255,255,0.04)", color: "#90a4ae", cursor: "pointer",
                fontSize: 14, fontWeight: 600,
              }}>← New Analysis</button>
              {result.status === "diseased" && (
                <div style={{
                  flex: 2, padding: "12px 16px", borderRadius: 12,
                  background: "rgba(255,82,82,0.07)", border: "1px solid rgba(255,82,82,0.2)",
                  color: "#ff5252", fontSize: 13, display: "flex", alignItems: "center", gap: 8,
                }}>
                  🩺 <span>Consult an aquatic veterinarian for proper diagnosis.</span>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Serif+Display&family=DM+Mono&display=swap');
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}
