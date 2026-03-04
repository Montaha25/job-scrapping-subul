"""
api.py — Career Assistant FastAPI backend (port 8001)
Scores : cosine_score + match_score uniquement (combined supprimé)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Career Assistant API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── DB Pool ────────────────────────────────────────────────────────────────
_pool: Optional[asyncpg.Pool] = None

def _get_dsn() -> str:
    dsn = os.getenv("POSTGRES_DSN", "").strip()
    if dsn:
        return dsn
    return (f"postgresql://{os.getenv('POSTGRES_USER','')}:{os.getenv('POSTGRES_PASSWORD','')}"
            f"@{os.getenv('POSTGRES_HOST','')}:{os.getenv('POSTGRES_PORT','5432')}"
            f"/{os.getenv('POSTGRES_DB','jobscan')}?sslmode=require")

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=_get_dsn(), min_size=2, max_size=10)
    return _pool

@app.on_event("startup")
async def startup():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY, name TEXT, role TEXT,
                experience INTEGER DEFAULT 0, skills TEXT, location TEXT,
                salary_min INTEGER DEFAULT 0, updated_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY, user_id INTEGER, role TEXT,
                content TEXT, created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id);
        """)
    logger.info("[startup] Career Assistant DB tables ready")

@app.on_event("shutdown")
async def shutdown():
    global _pool
    if _pool:
        await _pool.close()

# ── Learning meta ──────────────────────────────────────────────────────────
LEARNING_META = {
    "Python":       {"weeks": 8,  "difficulty": "beginner",     "resources": ["Python.org", "Real Python"]},
    "JavaScript":   {"weeks": 10, "difficulty": "beginner",     "resources": ["MDN", "javascript.info"]},
    "TypeScript":   {"weeks": 4,  "difficulty": "intermediate", "resources": ["TypeScript Handbook"]},
    "React":        {"weeks": 6,  "difficulty": "intermediate", "resources": ["React Docs", "Scrimba"]},
    "FastAPI":      {"weeks": 3,  "difficulty": "beginner",     "resources": ["FastAPI Docs"]},
    "Docker":       {"weeks": 3,  "difficulty": "intermediate", "resources": ["Docker Docs"]},
    "Kubernetes":   {"weeks": 6,  "difficulty": "advanced",     "resources": ["k8s.io", "KodeKloud"]},
    "AWS":          {"weeks": 8,  "difficulty": "intermediate", "resources": ["AWS Training", "A Cloud Guru"]},
    "GCP":          {"weeks": 6,  "difficulty": "intermediate", "resources": ["Google Cloud Skills Boost"]},
    "Azure":        {"weeks": 6,  "difficulty": "intermediate", "resources": ["Microsoft Learn"]},
    "PostgreSQL":   {"weeks": 4,  "difficulty": "intermediate", "resources": ["PostgreSQL Tutorial"]},
    "MongoDB":      {"weeks": 3,  "difficulty": "beginner",     "resources": ["MongoDB University"]},
    "Redis":        {"weeks": 2,  "difficulty": "intermediate", "resources": ["Redis University"]},
    "TensorFlow":   {"weeks": 8,  "difficulty": "advanced",     "resources": ["TensorFlow Docs", "DeepLearning.AI"]},
    "PyTorch":      {"weeks": 8,  "difficulty": "advanced",     "resources": ["PyTorch Tutorials", "Fast.ai"]},
    "scikit-learn": {"weeks": 4,  "difficulty": "intermediate", "resources": ["scikit-learn Docs", "Kaggle"]},
    "Spark":        {"weeks": 6,  "difficulty": "advanced",     "resources": ["Databricks Academy"]},
    "Kafka":        {"weeks": 4,  "difficulty": "advanced",     "resources": ["Confluent Courses"]},
    "Terraform":    {"weeks": 4,  "difficulty": "intermediate", "resources": ["HashiCorp Learn"]},
    "LangChain":    {"weeks": 3,  "difficulty": "intermediate", "resources": ["LangChain Docs"]},
    "Go":           {"weeks": 6,  "difficulty": "intermediate", "resources": ["Tour of Go"]},
    "Rust":         {"weeks": 10, "difficulty": "advanced",     "resources": ["The Rust Book"]},
    "dbt":          {"weeks": 3,  "difficulty": "intermediate", "resources": ["dbt Learn"]},
    "Airflow":      {"weeks": 3,  "difficulty": "intermediate", "resources": ["Airflow Docs"]},
    "Tableau":      {"weeks": 3,  "difficulty": "beginner",     "resources": ["Tableau Public"]},
    "Power BI":     {"weeks": 3,  "difficulty": "beginner",     "resources": ["Microsoft Learn Power BI"]},
}
DIFFICULTY_ORDER = {"beginner": 1, "intermediate": 2, "advanced": 3}

# ── Pydantic models ────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    user_id: int

class ProfileSaveRequest(BaseModel):
    user_id: int; name: str = ""; role: str = ""; experience: int = 0
    skills: str = ""; location: str = ""; salary_min: int = 0

class MatchRequest(BaseModel):
    user_id: int; role: str = ""; location: str = ""; min_fit: float = 0.0

class GapRequest(BaseModel):
    user_id: int; job_url: str = ""

class RoadmapRequest(BaseModel):
    user_id: int; job_url: str = ""

class ChatRequest(BaseModel):
    user_id: int; message: str

class ReportRequest(BaseModel):
    user_id: int

# ── DB helpers ─────────────────────────────────────────────────────────────
async def _get_user(user_id: int) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
    return dict(row) if row else None

async def _get_profile(user_id: int) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM profiles WHERE user_id = $1", user_id)
    return dict(row) if row else None

async def _get_jobs(user_id: int) -> List[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM jobs WHERE $1 = ANY(id_user) ORDER BY match_score DESC NULLS LAST",
            user_id,
        )
    result = []
    for row in rows:
        gap_missing = []
        try:
            gap_missing = json.loads(row["skills_gap"] or "[]")
        except Exception:
            pass
        match_raw  = row["match_score"]
        cosine_raw = row["cosine_score"] or 0.0
        result.append({
            "url":         row["url"],
            "source":      row["source"] or "",
            "title":       row["title"]  or "",
            "company":     row["industry"] or "",
            "location":    row["location"] or "",
            "remote":      row["remote"]   or "",
            "salary":      row["salary"]   or "",
            "contract":    row["contract"] or "",
            "education":   row["education"] or "",
            "experience":  row["seniority"] or "",
            "description": row["description"] or "",
            "skills_req":  row["must_have"]   or "",
            "skills_bon":  row["nice_to_have"] or "",
            "cosine_score": cosine_raw,
            "match_score":  match_raw if match_raw is not None else -1.0,
            "gap_missing":  gap_missing,
            "gap_total":    len(gap_missing),
        })
    return result

def _compute_skills_gap(cv_skills: str, job: dict) -> dict:
    cv_set     = set(s.strip().lower() for s in cv_skills.split(",") if s.strip())
    job_skills = [s.strip() for s in (job.get("skills_req") or "").split(",") if s.strip()]
    if not job_skills:
        return {"missing": [], "matched": [], "coverage": 1.0, "total": 0}
    missing, matched = [], []
    for skill in job_skills:
        if skill.lower() in cv_set or any(s in skill.lower() for s in cv_set):
            matched.append(skill)
        else:
            missing.append(skill)
    total = len(job_skills)
    return {"missing": missing, "matched": matched, "coverage": round(len(matched)/total, 3) if total else 1.0, "total": total}

def _xai_explain_scores(job: dict) -> dict:
    cosine = job.get("cosine_score", 0.0)
    ai     = job.get("match_score",  -1.0)
    explanations = []
    cos_pct = round(cosine * 100, 1)
    if cos_pct >= 80:
        explanations.append(f"🎯 Title Match {cos_pct}% — Your job title strongly aligns with this role.")
    elif cos_pct >= 60:
        explanations.append(f"✅ Title Match {cos_pct}% — Good alignment between your profile title and this position.")
    else:
        explanations.append(f"⚠️ Title Match {cos_pct}% — Title similarity is lower; the role may differ from your expertise.")
    if ai >= 0:
        ai_pct = round(ai * 100, 1)
        if ai_pct >= 75:
            explanations.append(f"🧠 AI Match {ai_pct}% — Strong compatibility between your full CV and this job.")
        elif ai_pct >= 55:
            explanations.append(f"🧠 AI Match {ai_pct}% — Moderate fit. Your profile covers most requirements.")
        else:
            explanations.append(f"🧠 AI Match {ai_pct}% — Lower AI match. Limited overlap between your CV and job details.")
    else:
        explanations.append("🧠 AI Match — Not available (model not loaded).")
    n_missing = len(job.get("gap_missing", []))
    n_total   = job.get("gap_total", 0)
    if n_total > 0:
        cov = round((1 - n_missing / n_total) * 100)
        miss_preview = ', '.join(job.get('gap_missing', [])[:4])
        explanations.append(f"📊 Skills Coverage {cov}% — You have {n_total-n_missing}/{n_total} required skills. Missing: {miss_preview}{'...' if n_missing > 4 else ''}.")
    ai_pct_val = round(ai * 100, 1) if ai >= 0 else 0
    return {
        "cosine_score":   cosine,
        "match_score":    ai,
        "explanations":   explanations,
        "score_formula":  "ranking = AI Match Score (fine-tuned BiEncoder)",
        "interpretation": ("excellent" if ai_pct_val >= 75 else "good" if ai_pct_val >= 55 else "moderate" if ai_pct_val >= 40 else "low"),
    }

def _generate_roadmap(missing_skills: List[str], weeks_available: int = 12) -> List[dict]:
    roadmap = []
    week_cursor = 1
    sorted_skills = sorted(missing_skills, key=lambda s: DIFFICULTY_ORDER.get(LEARNING_META.get(s, {}).get("difficulty", "intermediate"), 2))
    for skill in sorted_skills:
        meta  = LEARNING_META.get(skill, {"weeks": 3, "difficulty": "intermediate", "resources": [f"Search: {skill} tutorial"]})
        weeks = meta["weeks"]
        if week_cursor + weeks > weeks_available + 4:
            break
        roadmap.append({"skill": skill, "week_start": week_cursor, "week_end": week_cursor+weeks-1,
                         "duration": f"{weeks} weeks", "difficulty": meta["difficulty"],
                         "resources": meta["resources"], "priority": "high" if weeks <= 3 else "medium" if weeks <= 6 else "long-term"})
        week_cursor += weeks
    return roadmap

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/api/login")
async def login(req: LoginRequest):
    user = await _get_user(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {req.user_id} not found. Please run a scan first on JobScan.")
    return {"user_id": req.user_id, "status": "ok",
            "name":     f"{user.get('first_name') or ''} {user.get('last_name') or ''}".strip() or f"User #{req.user_id}",
            "role":     user.get("role", ""), "skills": user.get("skills", ""), "seniority": user.get("seniority", "")}

@app.get("/api/status")
async def status():
    pool = await get_pool()
    async with pool.acquire() as conn:
        total   = await conn.fetchval("SELECT COUNT(*) FROM jobs") or 0
        users   = await conn.fetchval("SELECT COUNT(*) FROM users") or 0
        avg_ai  = await conn.fetchval("SELECT AVG(match_score) FROM jobs WHERE match_score >= 0") or 0
        avg_cos = await conn.fetchval("SELECT AVG(cosine_score) FROM jobs") or 0
        sources = await conn.fetch("SELECT source, COUNT(*) as n FROM jobs GROUP BY source ORDER BY n DESC LIMIT 6")
    return {"total_jobs": total, "total_users": users,
            "avg_ai_score": round(float(avg_ai)*100, 1), "avg_cosine_score": round(float(avg_cos)*100, 1),
            "sources": [{"source": r["source"], "count": r["n"]} for r in sources]}

@app.get("/api/profile")
async def get_profile_endpoint(user_id: int):
    user    = await _get_user(user_id)
    profile = await _get_profile(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user_id,
            "name":       f"{user.get('first_name') or ''} {user.get('last_name') or ''}".strip(),
            "role":       profile.get("role") if profile else user.get("role", ""),
            "experience": profile.get("experience", 0) if profile else 0,
            "skills":     profile.get("skills") if profile else user.get("skills", ""),
            "location":   profile.get("location", "") if profile else "",
            "salary_min": profile.get("salary_min", 0) if profile else 0,
            "seniority":  user.get("seniority", ""), "education": user.get("education", ""),
            "industry":   user.get("industry", ""),  "cv_skills": user.get("skills", "")}

@app.post("/api/profile")
async def save_profile(req: ProfileSaveRequest):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO profiles (user_id,name,role,experience,skills,location,salary_min,updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                name=EXCLUDED.name, role=EXCLUDED.role, experience=EXCLUDED.experience,
                skills=EXCLUDED.skills, location=EXCLUDED.location,
                salary_min=EXCLUDED.salary_min, updated_at=NOW()
        """, req.user_id, req.name, req.role, req.experience, req.skills, req.location, req.salary_min)
    return {"status": "saved", "user_id": req.user_id}

@app.post("/api/matches")
async def get_matches(req: MatchRequest):
    jobs = await _get_jobs(req.user_id)
    user = await _get_user(req.user_id)
    if not jobs:
        return {"matches": [], "total": 0, "message": "No jobs found. Run a scan on JobScan first."}
    filtered = []
    for job in jobs:
        if req.role and req.role.lower() not in job["title"].lower():
            continue
        if req.location and req.location.lower() not in (job["location"] or "").lower():
            continue
        if req.min_fit > 0 and job["match_score"] < req.min_fit:
            continue
        job["xai"] = _xai_explain_scores(job)
        if user and user.get("skills"):
            gap = _compute_skills_gap(user["skills"], job)
            job["gap_missing"] = gap["missing"]; job["gap_matched"] = gap["matched"]
            job["gap_coverage"] = gap["coverage"]; job["gap_total"] = gap["total"]
        filtered.append(job)
    return {"matches": filtered, "total": len(filtered), "user_id": req.user_id}

@app.post("/api/gap")
async def skills_gap(req: GapRequest):
    user = await _get_user(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    jobs = await _get_jobs(req.user_id)
    if not jobs:
        return {"gap": [], "message": "No jobs found"}
    cv_skills = user.get("skills", "")
    if req.job_url:
        target = next((j for j in jobs if j["url"] == req.job_url), None)
        if not target:
            raise HTTPException(status_code=404, detail="Job not found")
        gap = _compute_skills_gap(cv_skills, target)
        return {"job_title": target["title"], "job_url": req.job_url, "xai": _xai_explain_scores(target), **gap}
    all_missing: dict = {}
    for job in jobs[:20]:
        for skill in _compute_skills_gap(cv_skills, job)["missing"]:
            all_missing[skill] = all_missing.get(skill, 0) + 1
    top_missing = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)[:15]
    return {"top_missing_skills": [{"skill": s, "frequency": f, "appears_in": f"{f} jobs"} for s, f in top_missing],
            "total_jobs_analyzed": min(len(jobs), 20), "cv_skills": cv_skills}

@app.post("/api/roadmap")
async def learning_roadmap(req: RoadmapRequest):
    user = await _get_user(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    jobs = await _get_jobs(req.user_id)
    if not jobs:
        return {"roadmap": [], "message": "No jobs found"}
    cv_skills = user.get("skills", "")
    skill_freq: dict = {}
    xai_per_job = []
    for job in jobs[:10]:
        for skill in _compute_skills_gap(cv_skills, job)["missing"]:
            skill_freq[skill] = skill_freq.get(skill, 0) + 1
        xai_per_job.append({"job_title": job["title"], "job_company": job["company"],
                             "match_score": job["match_score"], "xai": _xai_explain_scores(job)})
    top_skills  = [s for s, _ in sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)][:10]
    roadmap     = _generate_roadmap(top_skills)
    total_weeks = sum(int(r["duration"].split()[0]) for r in roadmap)
    return {"roadmap": roadmap, "total_weeks": total_weeks, "skills_to_learn": top_skills,
            "xai_top_jobs": xai_per_job[:5], "message": f"Learning plan: {len(roadmap)} skills over ~{total_weeks} weeks."}

@app.get("/api/market")
async def market_insights(user_id: int):
    jobs = await _get_jobs(user_id)
    if not jobs:
        return {"message": "No jobs found. Run a scan on JobScan first."}
    skill_count: dict = {}
    for job in jobs:
        for s in (job.get("skills_req") or "").split(","):
            s = s.strip()
            if s and s.lower() != "not specified":
                skill_count[s] = skill_count.get(s, 0) + 1
    loc_count: dict = {}
    for job in jobs:
        loc = (job.get("location") or "").strip()
        if loc: loc_count[loc] = loc_count.get(loc, 0) + 1
    company_count: dict = {}
    for job in jobs:
        co = (job.get("company") or "").strip()
        if co: company_count[co] = company_count.get(co, 0) + 1
    scores    = [j["match_score"] for j in jobs if j["match_score"] and j["match_score"] > 0]
    avg_score = round(sum(scores)/len(scores)*100, 1) if scores else 0
    return {
        "top_skills":      [{"skill": s, "count": c} for s, c in sorted(skill_count.items(), key=lambda x: x[1], reverse=True)[:12]],
        "top_locations":   [{"location": l, "count": c} for l, c in sorted(loc_count.items(), key=lambda x: x[1], reverse=True)[:8]],
        "top_companies":   [{"company": co, "count": c} for co, c in sorted(company_count.items(), key=lambda x: x[1], reverse=True)[:10]],
        "total_jobs": len(jobs), "avg_ai_score": avg_score,
        "score_breakdown": {
            "excellent": sum(1 for s in scores if s >= 0.75),
            "good":      sum(1 for s in scores if 0.55 <= s < 0.75),
            "moderate":  sum(1 for s in scores if 0.40 <= s < 0.55),
            "low":       sum(1 for s in scores if s < 0.40),
        },
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    pool = await get_pool()
    user = await _get_user(req.user_id)
    jobs = await _get_jobs(req.user_id)
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO chat_history (user_id,role,content) VALUES ($1,$2,$3)", req.user_id, "user", req.message)
    msg  = req.message.lower()
    role = (user.get("role", "") if user else "") or "professional"
    if any(w in msg for w in ["match", "job", "best", "top"]):
        top = sorted(jobs, key=lambda j: j["match_score"] if j["match_score"] >= 0 else -1, reverse=True)[:3]
        if top:
            lines = [f"🏆 Your top {len(top)} matches:\n"]
            for i, j in enumerate(top, 1):
                ai  = round(j["match_score"]*100, 1) if j["match_score"] >= 0 else None
                cos = round(j["cosine_score"]*100, 1)
                lines.append(f"**{i}. {j['title']}** at {j['company']}\n   Cosine: {cos}%{f' | AI: {ai}%' if ai else ''}\n   📍 {j['location']}")
            response = "\n".join(lines)
        else:
            response = "No jobs found yet. Run a scan on JobScan first!"
    elif any(w in msg for w in ["skill", "missing", "learn", "gap"]):
        if user and user.get("skills") and jobs:
            all_missing: dict = {}
            for job in jobs[:10]:
                for s in _compute_skills_gap(user["skills"], job)["missing"]:
                    all_missing[s] = all_missing.get(s, 0) + 1
            top5 = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)[:5]
            response = (f"📚 Top skills to learn:\n\n" + "\n".join(f"• **{s}** — {f} jobs" for s, f in top5)) if top5 else "You have all required skills! 🎉"
        else:
            response = "Please run a scan on JobScan first."
    elif any(w in msg for w in ["score", "explain", "why", "xai"]):
        if jobs:
            best = max(jobs, key=lambda j: j["match_score"] if j["match_score"] >= 0 else -1)
            xai  = _xai_explain_scores(best)
            response = f"🔍 **{best['title']}** at {best['company']}:\n\n" + "\n".join(xai["explanations"])
        else:
            response = "No jobs to explain yet."
    elif any(w in msg for w in ["roadmap", "plan", "weeks"]):
        if user and user.get("skills") and jobs:
            all_missing: dict = {}
            for job in jobs[:10]:
                for s in _compute_skills_gap(user["skills"], job)["missing"]:
                    all_missing[s] = all_missing.get(s, 0) + 1
            roadmap = _generate_roadmap([s for s, _ in sorted(all_missing.items(), key=lambda x: x[1], reverse=True)][:6], 16)
            response = ("🗺️ Roadmap:\n\n" + "\n".join(f"• **{r['skill']}** — W{r['week_start']}-{r['week_end']} ({r['difficulty']})" for r in roadmap)) if roadmap else "You're ready! 🎉"
        else:
            response = "Run a scan first."
    elif any(w in msg for w in ["market", "demand", "trend"]):
        if jobs:
            sc: dict = {}
            for job in jobs:
                for s in (job.get("skills_req") or "").split(","):
                    s = s.strip()
                    if s: sc[s] = sc.get(s, 0) + 1
            top = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:5]
            response = "📊 Top in-demand skills:\n\n" + "\n".join(f"• **{s}** — {c} jobs" for s, c in top)
        else:
            response = "No data yet."
    elif any(w in msg for w in ["hello", "hi", "bonjour", "salut", "hey"]):
        name = (user.get("first_name") or f"User #{req.user_id}") if user else f"User #{req.user_id}"
        response = (f"👋 Hello {name}! I'm your Career Assistant.\n\n"
                    "I can help you with:\n• 🏆 Top matches\n• 📚 Skills gap\n• 🗺️ Learning roadmap\n• 🔍 Score explanation\n• 📊 Market trends")
    else:
        response = f"I found **{len(jobs)} matched jobs**. Ask about matches, skills gap, roadmap, or market trends."
    async with pool.acquire() as conn:
        await conn.execute("INSERT INTO chat_history (user_id,role,content) VALUES ($1,$2,$3)", req.user_id, "assistant", response)
    return {"response": response, "user_id": req.user_id}

@app.post("/api/report")
async def generate_report(req: ReportRequest):
    user    = await _get_user(req.user_id)
    jobs    = await _get_jobs(req.user_id)
    profile = await _get_profile(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    name    = f"{user.get('first_name') or ''} {user.get('last_name') or ''}".strip() or f"User #{req.user_id}"
    role    = (profile.get("role") if profile else None) or user.get("role", "")
    skills  = user.get("skills", "")
    all_missing: dict = {}
    if skills:
        for job in jobs[:15]:
            for s in _compute_skills_gap(skills, job)["missing"]:
                all_missing[s] = all_missing.get(s, 0) + 1
    top_missing = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)[:8]
    roadmap     = _generate_roadmap([s for s, _ in top_missing[:6]])
    report = f"""# Career Analysis Report
**Generated:** {datetime.now().strftime('%B %d, %Y')}
**Candidate:** {name} | **Role:** {role} | **Seniority:** {user.get('seniority','N/A')}

---

## Profile
- **Skills:** {skills[:200]}{"..." if len(skills) > 200 else ""}
- **Education:** {user.get('education','N/A')} | **Industry:** {user.get('industry','N/A')}

---

## Scoring System
| Score | Method | Measures |
|-------|--------|----------|
| Cosine Score | Sentence-Transformer | Title alignment |
| AI Match Score | Fine-tuned BiEncoder | Full CV ↔ Job fit |

---

## Top Matched Jobs ({len(jobs)} total)

"""
    for i, job in enumerate(jobs[:5], 1):
        ai  = round(job["match_score"]*100, 1) if job["match_score"] >= 0 else "N/A"
        cos = round(job["cosine_score"]*100, 1)
        xai = _xai_explain_scores(job)
        gap = _compute_skills_gap(skills, job) if skills else {"missing": []}
        report += f"### {i}. {job['title']}\n**Company:** {job['company']} | **Location:** {job['location']}\n**AI Match:** {ai}% | **Cosine:** {cos}% | **{xai['interpretation'].upper()}**\n\n"
        for exp in xai["explanations"]: report += f"- {exp}\n"
        if gap["missing"]: report += f"\n**Missing:** {', '.join(gap['missing'][:6])}\n"
        report += f"\n🔗 [Apply]({job['url']})\n\n---\n\n"
    report += "## Skills Gap\n\n| Skill | Jobs | Difficulty | Weeks |\n|-------|------|-----------|-------|\n"
    for skill, freq in top_missing:
        meta = LEARNING_META.get(skill, {"difficulty": "intermediate", "weeks": 4})
        report += f"| {skill} | {freq} | {meta['difficulty']} | {meta['weeks']}w |\n"
    report += "\n\n## Roadmap\n\n"
    for r in roadmap:
        report += f"**{r['skill']}** (W{r['week_start']}–{r['week_end']}) — {r['difficulty']} — {', '.join(r['resources'])}\n\n"
    report += "\n---\n*Generated by Career Assistant + JobScan AI Pipeline*\n"
    return {"report": report, "user_id": req.user_id}

@app.post("/api/report/pdf")
async def generate_pdf_report(req: ReportRequest):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        import io
    except ImportError:
        raise HTTPException(status_code=500, detail="ReportLab not installed. Run: pip install reportlab")
    user  = await _get_user(req.user_id)
    jobs  = await _get_jobs(req.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    PURPLE = colors.HexColor("#6c63ff")
    title_style = ParagraphStyle("Title", parent=styles["Title"],   textColor=PURPLE, fontSize=20, spaceAfter=6)
    h1_style    = ParagraphStyle("H1",    parent=styles["Heading1"],textColor=PURPLE, fontSize=14, spaceAfter=4)
    h2_style    = ParagraphStyle("H2",    parent=styles["Heading2"],textColor=colors.HexColor("#1a1a2e"), fontSize=11, spaceAfter=3)
    body_style  = ParagraphStyle("Body",  parent=styles["Normal"],  fontSize=9, spaceAfter=3, leading=13)
    small_style = ParagraphStyle("Small", parent=styles["Normal"],  fontSize=8, textColor=colors.grey)
    name   = f"{user.get('first_name') or ''} {user.get('last_name') or ''}".strip() or f"User #{req.user_id}"
    skills = user.get("skills", "")
    story  = []
    story.append(Paragraph("Career Analysis Report", title_style))
    story.append(Paragraph(f"Candidate: {name} — {user.get('role', '')}", h2_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", small_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Scoring System", h1_style))
    t = Table([["Score","Method","Measures"],["Cosine Score","Sentence-Transformer","Title alignment"],["AI Match Score","Fine-tuned BiEncoder","Full CV ↔ Job fit"]],
              colWidths=[4.5*cm, 6*cm, 6*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),PURPLE),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                            ("FONTSIZE",(0,0),(-1,-1),8),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5ff")]),
                            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#dddddd")),("PADDING",(0,0),(-1,-1),5)]))
    story.append(t); story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Top Matched Jobs ({len(jobs)} total)", h1_style))
    for i, job in enumerate(jobs[:5], 1):
        ai  = round(job["match_score"]*100, 1) if job["match_score"] >= 0 else "N/A"
        cos = round(job["cosine_score"]*100, 1)
        xai = _xai_explain_scores(job)
        story.append(Paragraph(f"{i}. {job['title']} — {job['company']}", h2_style))
        story.append(Paragraph(f"Location: {job['location']} | Source: {job['source']}", small_style))
        story.append(Paragraph(f"AI Match: <b>{ai}%</b> | Cosine: <b>{cos}%</b> | {xai['interpretation']}", body_style))
        for exp in xai["explanations"][:2]: story.append(Paragraph(f"• {exp}", small_style))
        story.append(Spacer(1, 0.3*cm))
    if skills and jobs:
        story.append(Paragraph("Skills Gap", h1_style))
        all_missing: dict = {}
        for job in jobs[:15]:
            for s in _compute_skills_gap(skills, job)["missing"]:
                all_missing[s] = all_missing.get(s, 0) + 1
        top_missing = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)[:8]
        if top_missing:
            gd = [["Skill","Appears In","Difficulty","Time"]] + [[s, f"{f} jobs", LEARNING_META.get(s,{"difficulty":"intermediate"})["difficulty"], f"{LEARNING_META.get(s,{'weeks':4})['weeks']}w"] for s,f in top_missing]
            t2 = Table(gd, colWidths=[5*cm, 3*cm, 4*cm, 4.5*cm])
            t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),PURPLE),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                    ("FONTSIZE",(0,0),(-1,-1),8),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5ff")]),
                                    ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#dddddd")),("PADDING",(0,0),(-1,-1),5)]))
            story.append(t2)
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Report generated by Career Assistant + JobScan AI Pipeline", small_style))
    doc.build(story); buffer.seek(0)
    return Response(content=buffer.read(), media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=career_report_{req.user_id}.pdf"})