"use client";
// ─────────────────────────────────────────────────────────────────────────────
//  app/page.tsx  —  LOGIN
//
//  Flow:
//    1. User enters their numeric ID
//    2. POST /api/login
//       • 200 OK  → user exists  → router.push("/app?user_id=X")
//       • 4xx     → new user     → show CV textarea
//    3. User pastes CV → click "Analyze my CV"
//       → router.push("/app?user_id=X&scan=1")
//       → /app runs the JobScan SSE pipeline and shows the dashboard
// ─────────────────────────────────────────────────────────────────────────────

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { GRAD, FONT, MONO, C, S, GLOBAL_CSS } from "@/app/lib/theme";

// ── Small helpers local to this page ─────────────────────────────────────────

function GradBar() {
  return <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: GRAD }} />;
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <label style={{ display: "block", fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8 }}>
      {children}
    </label>
  );
}

// ── Pipeline preview chips (shown before scan starts) ─────────────────────────
const PIPELINE_PREVIEW = [
  "🌍 Detect language",
  "🤖 Extract title",
  "🧬 Structure CV",
  "💾 Save profile",
  "🕷️ Scrape 6 boards",
  "📐 Cosine ≥ 0.60",
  "🧠 AI scoring",
  "🗄️ Save jobs",
];

// ── Main component ────────────────────────────────────────────────────────────
export default function LoginPage() {
  const router = useRouter();

  // Step: "id" = enter user ID,  "cv" = paste CV
  const [step,      setStep]      = useState<"id" | "cv">("id");
  const [uidInput,  setUidInput]  = useState("");
  const [cvText,    setCvText]    = useState("");
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");

  // Restore previous session values
  useEffect(() => {
    const sid = sessionStorage.getItem("jobscan_user_id");
    const scv = sessionStorage.getItem("jobscan_cv_text");
    if (sid) setUidInput(sid);
    if (scv) setCvText(scv);
  }, []);

  // ── Step 1: check if user exists ──────────────────────────────────────────
  async function handleCheckUser() {
    const uid = parseInt(uidInput, 10);
    if (!uid || uid < 1) { setError("Please enter a valid numeric ID"); return; }
    setLoading(true); setError("");
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: uid }),
      });
      sessionStorage.setItem("jobscan_user_id", String(uid));
      if (res.ok) {
        // ✅ Existing user → go straight to dashboard
        router.push(`/app?user_id=${uid}`);
      } else {
        // ❌ New user → ask for CV
        setStep("cv");
      }
    } catch {
      setError("Connection error — is the Career Assistant API running on port 8001?");
    } finally {
      setLoading(false);
    }
  }

  // ── Step 2: save CV and redirect (scan will run inside /app) ─────────────
  function handleAnalyzeCV() {
    const uid = parseInt(uidInput, 10);
    const cv  = cvText.trim();
    if (cv.length < 30) { setError("Please paste your CV (minimum 30 characters)"); return; }
    sessionStorage.setItem("jobscan_cv_text", cv);
    // ?scan=1 tells /app to start the SSE pipeline immediately
    router.push(`/app?user_id=${uid}&scan=1`);
  }

  const charCount = cvText.trim().length;
  const charOk    = charCount >= 30;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ ...S.page, display: "flex", flexDirection: "column" }}>
      <style>{GLOBAL_CSS}</style>

      {/* ── Minimal header ── */}
      <header style={{ background: C.white, borderBottom: `1px solid ${C.border}`, height: 60, display: "flex", alignItems: "center", padding: "0 32px" }}>
        <span style={{ fontWeight: 800, fontSize: 18, background: GRAD, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
          CareerAssistant
        </span>
      </header>

      {/* ── Centered card ── */}
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 16px" }}>
        <div style={{
          background: C.white, border: `1px solid ${C.border}`, borderRadius: 20,
          padding: "48px 44px", width: "100%", maxWidth: step === "cv" ? 560 : 440,
          boxShadow: "0 8px 40px rgba(122,63,176,.13)",
          position: "relative", overflow: "hidden", animation: "fadeUp .4s ease",
        }}>
          <GradBar />

          {/* Logo */}
          <div style={{ textAlign: "center", marginBottom: 34 }}>
            <div style={{ fontSize: 46, marginBottom: 10 }}>🎯</div>
            <h1 style={{ fontSize: 26, fontWeight: 800, margin: 0, background: GRAD, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
              CareerAssistant
            </h1>
            <p style={{ fontSize: 12, color: "#9f8fb0", marginTop: 8 }}>
              Powered by JobScan AI · Azure OpenAI + MiniLM
            </p>
          </div>

          {/* ════════════════ STEP 1 — USER ID ════════════════ */}
          {step === "id" && (
            <>
              <Label>Your User ID</Label>
              <div style={{ position: "relative", marginBottom: 8 }}>
                <span style={{ position: "absolute", left: 14, top: "50%", transform: "translateY(-50%)", fontSize: 16, pointerEvents: "none" }}>👤</span>
                <input
                  type="number" placeholder="e.g. 1001" autoFocus
                  value={uidInput}
                  onChange={e => setUidInput(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleCheckUser()}
                  style={{ ...S.input, paddingLeft: 44, fontSize: 20, fontWeight: 700, letterSpacing: 4, borderRadius: 12 }}
                />
              </div>
              <p style={{ fontSize: 11, color: "#9f8fb0", textAlign: "center", marginBottom: 24 }}>
                Existing ID → load your jobs &nbsp;·&nbsp; New ID → create profile
              </p>
              <button onClick={handleCheckUser} disabled={loading}
                style={{ ...S.btn, width: "100%", padding: "14px", fontSize: 15, borderRadius: 12, opacity: loading ? 0.7 : 1 }}>
                {loading ? "Checking…" : "Continue →"}
              </button>
            </>
          )}

          {/* ════════════════ STEP 2 — CV INPUT ════════════════ */}
          {step === "cv" && (
            <>
              {/* Sub-heading */}
              <div style={{ marginBottom: 18 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#9f8fb0", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4, fontFamily: MONO }}>
                  New profile · User #{uidInput}
                </div>
                <h2 style={{ fontSize: 20, fontWeight: 800, color: C.text, margin: 0 }}>Paste your CV / Profile Summary</h2>
                <p style={{ fontSize: 12, color: "#9f8fb0", marginTop: 6, lineHeight: 1.6 }}>
                  The AI extracts your profile, scrapes 6 job boards, scores each match and computes your skills gap in real time.
                </p>
              </div>

              {/* Textarea */}
              <Label>CV / Profile Summary</Label>
              <textarea
                rows={9}
                placeholder={"Paste your full CV or profile summary here…\n\n• Work experience, job titles, companies\n• Technical skills: Python, SQL, Docker, AWS…\n• Education, certifications\n• Projects and achievements"}
                value={cvText}
                onChange={e => setCvText(e.target.value)}
                style={{ ...S.input, fontSize: 12, fontFamily: MONO, lineHeight: 1.75, resize: "vertical", minHeight: 200, borderRadius: 12 }}
              />
              {/* Character counter */}
              <div style={{ fontSize: 10, textAlign: "right", marginTop: 4, marginBottom: 16, fontFamily: MONO, color: charCount === 0 ? "#ccc" : charOk ? C.green : C.amber }}>
                {charCount === 0 ? "0 characters" : charOk ? `${charCount} characters ✓` : `${charCount} — need ${30 - charCount} more`}
              </div>

              {/* Pipeline preview */}
              <div style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10, padding: "10px 14px", marginBottom: 20 }}>
                <div style={{ fontSize: 9, fontWeight: 700, color: "#9f8fb0", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8, fontFamily: MONO }}>
                  Pipeline after Analyze
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                  {PIPELINE_PREVIEW.map(s => (
                    <span key={s} style={{ fontSize: 9, padding: "3px 8px", borderRadius: 6, border: `1px solid ${C.border}`, background: C.white, color: C.muted, fontFamily: MONO, fontWeight: 600 }}>
                      {s}
                    </span>
                  ))}
                </div>
              </div>

              {/* Buttons */}
              <div style={{ display: "flex", gap: 10 }}>
                <button onClick={() => { setStep("id"); setError(""); }} style={{ ...S.btnOut, flex: 1 }}>
                  ← Back
                </button>
                <button onClick={handleAnalyzeCV} disabled={!charOk}
                  style={{ ...S.btn, flex: 2, padding: "13px", fontSize: 14, borderRadius: 12, opacity: charOk ? 1 : 0.45 }}>
                  🔍 Analyze my CV
                </button>
              </div>
            </>
          )}

          {/* Error */}
          {error && (
            <div style={{ marginTop: 14, fontSize: 12, color: C.red, textAlign: "center" }}>{error}</div>
          )}

          {/* Footer hint */}
          <div style={{ marginTop: 22, textAlign: "center", fontSize: 11, color: "#9f8fb0" }}>
            2 AI scores: <span style={{ color: C.p2, fontWeight: 600 }}>Cosine · AI Match</span>
          </div>
        </div>
      </div>
    </div>
  );
}