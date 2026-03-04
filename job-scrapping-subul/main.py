"""
main.py — JobScan FastAPI backend

Full pipeline for POST /scan :
  0.  Detect CV language → translate to English if needed
  1.  LLM extracts cv_title from (translated) CV text
  2.  LLM structures CV → exact training format dict
  3.  Encode cv_title → embedding vector (paraphrase-multilingual-MiniLM-L12-v2)
  4.  Scrape 6 sources en parallèle :
        - aijobs.ai        (offres internationales AI/Tech)
        - remoteok.com     (offres remote worldwide)
        - emploitic.com    (offres Algérie)
        - tanitjobs.com    (offres Tunisie, via SerpApi)
        - greenhouse.io    (API JSON publique — Airbnb, Stripe, Anthropic, OpenAI…)
        - eluta.ca         (moteur emploi canadien, via SerpApi)
  5.  Per job passant le filtre cosine (≥ 0.60) :
        a. LLM extracts full job details + all_skills (full page text)
        b. model.predict(cv_structured, job_details) → AI match score
        c. compute_skills_gap(cv_structured, job_details) → missing skills
        d. compute_combined_score(ai_match, gap) → score final = AI × √(coverage)
        e. SSE {"event":"job", ...all fields + gap + combined_score...}

FIX user_id :
  - user_id est lu depuis req.user_id (int, défaut 0)
  - Si user_id <= 0 → scan anonyme (pas de sauvegarde DB)
  - Si user_id > 0 → upsert_user + insert_job après chaque card enrichi
  - Le frontend DOIT envoyer user_id dans le body JSON du POST /scan
  - Le patch JS intercepte le fetch et ajoute user_id depuis l'URL ?user_id=X

FIX renderCard cache :
  - window.renderCard est exposé globalement dans index.html
  - Le patch JS appelle window.renderCard(data) pour les jobs depuis le cache DB
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncAzureOpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import matcher as mtch
from database import init_db, close_pool, upsert_user, insert_job, get_jobs_for_user, update_user_profile
from llm_extractor import extract_with_llm
from scraper import (
    scrape_aijobs,
    scrape_emploitic,
    scrape_remoteok,
    scrape_tanitjobs,
    scrape_greenhouse,
    scrape_eluta,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="JobScan")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("[startup] PostgreSQL DB initialized OK")


@app.on_event("shutdown")
async def shutdown():
    await close_pool()


# ── Models loaded at startup ──────────────────────────────────────────────────
logger.info("Loading sentence-transformer (multilingual cosine filter)...")
EMBED_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
logger.info("Sentence-transformer (multilingual) ready ✓")

logger.info("Loading fine-tuned matching model...")
MATCH_MODEL, MATCH_TOKENIZER = mtch.load_model()
if MATCH_MODEL is None:
    logger.warning(
        "⚠  Fine-tuned model not found.\n"
        "   Place files in:\n"
        "     jobscan_model/finetuned_model.pt\n"
        "     jobscan_model/tokenizer/\n"
    )
else:
    logger.info("Fine-tuned model ready ✓")

# ── Config ────────────────────────────────────────────────────────────────────
COSINE_THRESHOLD           = 0.60
COSINE_THRESHOLD_EMPLOITIC = 0.60
MAX_AGE_DAYS               = 45
LLM_CONCURRENCY            = 4
NUM_SOURCES                = 6

SHARED_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Request schema ────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    cv_text: str
    user_id: int = 0    # 0 = anonyme (pas de sauvegarde DB)


class ProfileRequest(BaseModel):
    """
    Corps du PATCH /profile/{user_id}.
    Tous les champs sont optionnels : seuls ceux fournis sont mis à jour.
    """
    first_name: str = None
    last_name:  str = None
    email:      str = None
    linkedin:   str = None


# ── Utilities ─────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def pct(score: float) -> str:
    return f"{score * 100:.2f}"


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _azure_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key        = os.getenv("AZURE_OPENAI_API_KEY",  ""),
        api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_login():
    """Page 1 : saisie user_id (login.html)."""
    return HTMLResponse((BASE_DIR / "login.html").read_text(encoding="utf-8"))


@app.get("/cv")
async def serve_cv():
    """
    Page 2 : index.html (résultats + scan CV).
    Servi sur /cv?user_id=X
    Le patch JS injecté dans index.html gère le user_id et le cache DB.
    """
    cv_html = BASE_DIR / "cv.html"
    html    = BASE_DIR / "index.html"
    return HTMLResponse((cv_html if cv_html.exists() else html).read_text(encoding="utf-8"))


@app.get("/jobs/{user_id}")
async def stream_cached_jobs(user_id: int):
    """
    SSE stream des jobs depuis PostgreSQL pour ce user_id.
    Appelé par le frontend avant de lancer un nouveau scan.
    Si l'utilisateur a déjà des jobs → les stream directement sans scraping.
    """
    logger.info(f"[/jobs] GET /jobs/{user_id}")

    async def _stream():
        jobs = await get_jobs_for_user(user_id)
        if not jobs:
            logger.info(f"[/jobs] no_cache for user_id={user_id}")
            yield sse({"event": "no_cache", "user_id": user_id})
            return
        logger.info(f"[/jobs] streaming {len(jobs)} cached jobs for user_id={user_id}")
        yield sse({"event": "cached", "total": len(jobs), "user_id": user_id})
        for job in jobs:
            yield sse(job)
        yield sse({"event": "done", "total": len(jobs), "from_cache": True})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.get("/profile/{user_id}")
async def get_profile(user_id: int):
    """
    Retourne le profil complet : first_name, last_name, email, linkedin + données CV.
    Utilisé par la page de paramètres ou le header pour afficher le nom.
    """
    from database import get_user
    from fastapi import HTTPException
    logger.info(f"[/profile] GET /profile/{user_id}")
    user = await get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return {"user_id": user_id, **user}


@app.patch("/profile/{user_id}")
async def patch_profile(user_id: int, req: ProfileRequest):
    """
    Met à jour first_name, last_name, email, linkedin pour un user existant.
    Les champs non fournis (None) ne sont PAS modifiés (COALESCE en DB).

    Exemple fetch depuis le frontend :
        await fetch(`/profile/${USER_ID}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                first_name: 'Alice',
                last_name:  'Dupont',
                email:      'alice@example.com',
                linkedin:   'https://linkedin.com/in/alice'
            })
        });
    """
    from fastapi import HTTPException
    logger.info(f"[/profile] PATCH /profile/{user_id} data={req.dict(exclude_none=True)}")
    ok = await update_user_profile(
        user_id,
        first_name = req.first_name,
        last_name  = req.last_name,
        email      = req.email,
        linkedin   = req.linkedin,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found or update failed")
    return {"user_id": user_id, "updated": req.dict(exclude_none=True), "status": "ok"}


@app.post("/scan")
async def scan(req: ScanRequest):
    # FIX : log pour confirmer que user_id est bien reçu
    logger.info(f"[/scan] POST received — user_id={req.user_id} cv_len={len(req.cv_text)}")
    return StreamingResponse(
        pipeline(req.cv_text.strip(), req.user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── LLM helpers ───────────────────────────────────────────────────────────────

async def detect_and_translate_cv(cv_text: str) -> tuple[str, str, str]:
    """
    Detect the language of the CV text.
    If NOT English → translate the full text to English.
    Returns: (cv_text_english, detected_lang, translated: "yes"|"no"|"error")
    """
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    excerpt    = cv_text.strip()[:600]

    try:
        async with _azure_client() as az:
            resp = await az.chat.completions.create(
                model=deployment, max_tokens=8, temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Detect the language of the text. "
                            "Reply with ONLY the language name in English. "
                            "Examples: English, French, Spanish, Arabic, German, Italian"
                        ),
                    },
                    {"role": "user", "content": excerpt},
                ],
            )
        detected_lang = resp.choices[0].message.content.strip().strip(".").strip()
        logger.info(f"[lang] Detected: '{detected_lang}'")
    except Exception as e:
        logger.error(f"[lang] Detection failed: {e}")
        return cv_text, "Unknown", "no"

    if detected_lang.lower() in ("english", "en"):
        return cv_text, "English", "no"

    logger.info(f"[lang] Translating from {detected_lang} to English...")
    try:
        chunk_size       = 2000
        chunks           = [cv_text[i:i + chunk_size] for i in range(0, len(cv_text), chunk_size)]
        translated_parts = []
        for chunk in chunks:
            async with _azure_client() as az:
                resp = await az.chat.completions.create(
                    model=deployment, max_tokens=1000, temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"Translate the following {detected_lang} resume text to English. "
                                "Preserve all structure, formatting, bullet points, dates, and names. "
                                "Translate professional terms accurately. "
                                "Reply with ONLY the translated text, nothing else."
                            ),
                        },
                        {"role": "user", "content": chunk},
                    ],
                )
            translated_parts.append(resp.choices[0].message.content.strip())
        cv_translated = "\n".join(translated_parts)
        logger.info(f"[lang] Translation complete ({len(cv_text)} → {len(cv_translated)} chars)")
        return cv_translated, detected_lang, "yes"
    except Exception as e:
        logger.error(f"[lang] Translation failed: {e}")
        return cv_text, detected_lang, "error"


async def extract_cv_title(cv_text: str) -> str:
    """Fast LLM call: extract the main job title from raw CV text."""
    try:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        excerpt    = cv_text.strip()[:800]
        async with _azure_client() as az:
            resp = await az.chat.completions.create(
                model=deployment, max_tokens=15, temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the main job title from the resume. "
                            "Reply with ONLY the title (1-5 words, English). "
                            "Examples: 'Data Engineer', 'ML Engineer', 'Backend Developer'"
                        ),
                    },
                    {"role": "user", "content": f"Resume:\n{excerpt}"},
                ],
            )
        title = resp.choices[0].message.content.strip().strip('"\'')
        logger.info(f"[cv_title] '{title}'")
        return title or "Software Engineer"
    except Exception as e:
        logger.error(f"[cv_title] failed: {e}")
        for line in cv_text.split("\n"):
            line = line.strip()
            if 3 <= len(line) <= 60:
                return line
        return "Software Engineer"


async def structure_cv_for_model(cv_title: str, cv_text: str) -> dict:
    """
    Structure raw CV into exact training format fields.
    "skills" = ALL technical skills → used for gap calculation AND AI score.
    """
    try:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        excerpt    = cv_text.strip()[:2000]
        async with _azure_client() as az:
            resp = await az.chat.completions.create(
                model=deployment, max_tokens=400, temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract structured fields from a resume. "
                            "Respond ONLY with a valid JSON object. "
                            "For the 'skills' field: list EVERY technical skill, "
                            "tool, language, framework, and technology mentioned "
                            "ANYWHERE in the resume — be comprehensive."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""Extract these fields from the resume as JSON:
{{
  "role":             "main job title (e.g. 'Data Engineer')",
  "seniority":        "Junior | Mid | Senior | Lead",
  "years_experience": "number only (e.g. '5')",
  "industry":         "sector (e.g. 'Finance', 'Healthcare', 'Technology')",
  "education":        "highest degree (e.g. 'Master', 'Bachelor', 'PhD')",
  "skills":           "ALL technical skills: 'Python, SQL, AWS, Spark, Docker, ...'",
  "summary":          "1 sentence professional summary",
  "bullets":          "2-3 key achievements, space-separated"
}}

Resume:
{excerpt}""",
                    },
                ],
            )
        data = json.loads(resp.choices[0].message.content.strip())
        data["role"] = cv_title
        logger.info(
            f"[cv_struct] role={data.get('role')} | "
            f"seniority={data.get('seniority')} | "
            f"skills={str(data.get('skills', ''))[:80]}"
        )
        return data
    except Exception as e:
        logger.error(f"[cv_struct] failed: {e}")
        return {
            "role": cv_title, "seniority": "Mid",
            "years_experience": "3", "industry": "Technology",
            "education": "Bachelor", "skills": "",
            "summary": cv_title, "bullets": "",
        }


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def pipeline(cv_text: str, user_id: int = 0):
    """
    Pipeline complet : langue → titre → structuration → scraping → enrichissement.

    FIX PRINCIPAL : user_id est passé explicitement à chaque étape de sauvegarde.
    Les logs permettent de tracer exactement où une sauvegarde échoue.
    """
    cutoff    = datetime.now() - timedelta(days=MAX_AGE_DAYS)
    llm_sem   = asyncio.Semaphore(LLM_CONCURRENCY)
    has_model = MATCH_MODEL is not None

    logger.info(f"[pipeline] START — user_id={user_id} save_to_db={user_id > 0}")

    # ── 0. Detect language + translate ───────────────────────────────────────
    if cv_text:
        yield sse({"event": "lang_detecting"})
        cv_text, detected_lang, translated = await detect_and_translate_cv(cv_text)
        yield sse({"event": "lang_ready", "lang": detected_lang, "translated": translated})

    # ── 1. Extract CV title ───────────────────────────────────────────────────
    cv_title = await extract_cv_title(cv_text) if cv_text else "Software Engineer"
    yield sse({"event": "cv_title", "title": cv_title})

    # ── 2. Structure CV ───────────────────────────────────────────────────────
    cv_structured: dict = {}
    if cv_text:
        yield sse({"event": "cv_structuring"})
        cv_structured = await structure_cv_for_model(cv_title, cv_text)
        yield sse({
            "event":     "cv_ready",
            "role":      cv_structured.get("role", cv_title),
            "seniority": cv_structured.get("seniority", ""),
            "skills":    cv_structured.get("skills", ""),
        })

        # ── Sauvegarder le user en PostgreSQL ─────────────────────────────────
        # FIX : user_id > 0 requis, sinon scan anonyme
        if user_id > 0:
            logger.info(f"[pipeline] Saving user to DB: user_id={user_id}")
            ok = await upsert_user(user_id, cv_structured)
            if ok:
                yield sse({"event": "user_saved", "user_id": user_id})
                logger.info(f"[pipeline] ✅ user_id={user_id} saved in table 'users'")
            else:
                logger.error(f"[pipeline] ❌ Failed to save user_id={user_id} in table 'users'")
        else:
            logger.info("[pipeline] user_id=0 → anonymous scan, skipping DB save")

    # ── 3. Encode cv_title vector ─────────────────────────────────────────────
    cv_vec: np.ndarray = await asyncio.to_thread(
        lambda: EMBED_MODEL.encode(cv_title, convert_to_numpy=True)
    )

    # ── Shared state ──────────────────────────────────────────────────────────
    result_q   = asyncio.Queue()
    src_done_q = asyncio.Queue()
    pending    = {"n": 0}
    scrapers   = {"done": 0}
    all_done   = {"v": False}

    connector = aiohttp.TCPConnector(limit=30)
    async with aiohttp.ClientSession(headers=SHARED_HEADERS, connector=connector) as session:

        # ── 4. Per-job cosine filter ──────────────────────────────────────────
        async def handle_job(job: dict, source: str):
            job["source"] = source
            threshold = COSINE_THRESHOLD_EMPLOITIC if source == "emploitic" else COSINE_THRESHOLD
            job_vec = await asyncio.to_thread(
                lambda: EMBED_MODEL.encode(job["title"], convert_to_numpy=True)
            )
            cosine = cosine_sim(cv_vec, job_vec)
            if cosine < threshold:
                logger.info(
                    f"  [filter] SKIP cosine={cosine:.2f} < {threshold} | "
                    f"{source} | {job['title'][:50]}"
                )
                return
            logger.info(
                f"  [filter] PASS cosine={cosine:.2f} >= {threshold} | "
                f"{source} | {job['title'][:50]}"
            )
            pending["n"] += 1
            asyncio.create_task(enrich(job, cosine))

        # ── 5. Enrichissement LLM + scores ───────────────────────────────────
        async def enrich(job: dict, cosine: float):
            async with llm_sem:
                try:
                    source = job.get("source", "")

                    # ── EMPLOITIC : fetch page après filtre cosine ─────────────
                    if source == "emploitic":
                        from scraper import _scrape_emploitic_fetch_one
                        full_job = await _scrape_emploitic_fetch_one(job["url"], session)
                        if full_job is None:
                            logger.warning(f"  [enrich/emploitic] fetch failed: {job['url'][:60]}")
                            return
                        details = {
                            "title":       full_job.get("title", "")    or job.get("title", ""),
                            "industry":    full_job.get("company", ""),
                            "location":    full_job.get("location", ""),
                            "remote":      full_job.get("remote", ""),
                            "salary":      full_job.get("salary", "Non spécifié"),
                            "contract":    full_job.get("_emp_contract", ""),
                            "experience":  full_job.get("_emp_experience", ""),
                            "education":   full_job.get("_emp_education", ""),
                            "pub_date":    full_job.get("time_ago", ""),
                            "expired":     full_job.get("_emp_status", ""),
                            "description": full_job.get("_emp_description", ""),
                            "skills_req":  "",
                            "skills_bon":  "",
                            "all_skills":  ", ".join(full_job.get("_emp_skills", []) or []),
                            "tags":        ", ".join(full_job.get("_emp_tags",   []) or []),
                        }
                        if full_job.get("time_ago"):
                            job["time_ago"] = full_job["time_ago"]

                    # ── ELUTA : bypass LLM (SerpApi snippet) ──────────────────
                    elif source == "eluta":
                        details = {
                            "title":       job.get("title", ""),
                            "industry":    job.get("company", ""),
                            "location":    job.get("location", ""),
                            "remote":      job.get("remote", ""),
                            "salary":      job.get("salary", "Not specified"),
                            "contract":    job.get("_eluta_contract", ""),
                            "experience":  job.get("_eluta_experience", ""),
                            "education":   "",
                            "pub_date":    job.get("time_ago", ""),
                            "expired":     "",
                            "description": job.get("_eluta_description", ""),
                            "skills_req":  job.get("_eluta_skills", ""),
                            "skills_bon":  "",
                            "all_skills":  job.get("_eluta_skills", ""),
                            "tags":        "",
                        }

                    # ── GREENHOUSE : bypass LLM (API JSON) ────────────────────
                    elif source == "greenhouse":
                        details = {
                            "title":       job.get("title", ""),
                            "industry":    job.get("company", ""),
                            "location":    job.get("location", ""),
                            "remote":      job.get("_gh_remote", ""),
                            "salary":      job.get("_gh_salary", "Not specified"),
                            "contract":    job.get("_gh_contract", ""),
                            "experience":  job.get("_gh_experience", ""),
                            "education":   job.get("_gh_education", ""),
                            "pub_date":    job.get("time_ago", ""),
                            "expired":     "",
                            "description": job.get("_gh_description", ""),
                            "skills_req":  job.get("_gh_skills", ""),
                            "skills_bon":  job.get("_gh_bonus", ""),
                            "all_skills":  job.get("_gh_skills", ""),
                            "tags":        job.get("_gh_tags", ""),
                        }

                    # ── TANITJOBS : bypass LLM (Cloudflare 403) ───────────────
                    elif source == "tanitjobs":
                        details = {
                            "title":       job.get("title", ""),
                            "industry":    job.get("company", ""),
                            "location":    job.get("location", ""),
                            "remote":      job.get("remote", ""),
                            "salary":      job.get("salary", "Non spécifié"),
                            "contract":    job.get("_tnj_contract", ""),
                            "experience":  job.get("_tnj_experience", ""),
                            "education":   "",
                            "pub_date":    job.get("time_ago", ""),
                            "expired":     "",
                            "description": job.get("_tnj_description", ""),
                            "skills_req":  "",
                            "skills_bon":  "",
                            "all_skills":  job.get("_tnj_all_skills", ""),
                            "tags":        "",
                        }

                    # ── AIJOBS + REMOTEOK : extraction LLM ────────────────────
                    else:
                        details = await extract_with_llm(
                            url=job["url"],
                            session=session,
                            cutoff=cutoff,
                        )
                        if details is None:
                            logger.warning(f"  [enrich] SKIP extract_with_llm=None | {job['url'][:60]}")
                            return

                    # ── b. AI match score ──────────────────────────────────────
                    match_score = -1.0
                    if has_model and cv_structured:
                        match_score = await asyncio.to_thread(
                            lambda: mtch.predict(
                                MATCH_MODEL, MATCH_TOKENIZER,
                                cv_structured, details,
                            )
                        )

                    # ── c. Skills gap ──────────────────────────────────────────
                    gap = {"missing": [], "matched": [], "coverage": 1.0, "total": 0}
                    if cv_structured:
                        gap = await asyncio.to_thread(
                            lambda: mtch.compute_skills_gap(cv_structured, details)
                        )

                    # ── d. Score combiné = AI × √(coverage) ───────────────────
                    combined_score = mtch.compute_combined_score(match_score, gap)

                    logger.info(
                        f"  ✦ {job['title'][:35]:35s} "
                        f"[{job['source']:9s}] "
                        f"cos={cosine:.2f} ai={match_score:.2f} "
                        f"comb={combined_score:.2f} "
                        f"cov={gap['coverage']:.0%} "
                        f"miss={len(gap['missing'])}/{gap['total']}"
                    )

                    # ── e. Construire le card SSE ──────────────────────────────
                    # FIX MAPPING : toutes les clés utilisées par database.insert_job
                    # et par le frontend renderCard sont présentes et correctement nommées.
                    card = {
                        # ── Event ─────────────────────────────────────────────
                        "event":   "job",
                        # ── Identifiants ──────────────────────────────────────
                        "url":     job["url"],
                        "source":  job["source"],
                        # ── Infos job ─────────────────────────────────────────
                        "title":    details.get("title")    or job.get("title", ""),
                        "industry": details.get("industry") or job.get("company", ""),  # nom société
                        "location": details.get("location") or job.get("location", ""),
                        "remote":   details.get("remote")   or job.get("remote", ""),
                        "salary":   details.get("salary")   or job.get("salary", "Not specified"),
                        "time_ago": job.get("time_ago", ""),
                        # ── Scores ────────────────────────────────────────────
                        "cosine":              cosine,
                        "cosine_display":      pct(cosine),
                        "match_score":         match_score,
                        "match_score_display": pct(match_score) if match_score >= 0 else "—",
                        "combined_score":         combined_score,
                        "combined_score_display": pct(combined_score),
                        # ── Skills gap ────────────────────────────────────────
                        "gap_missing":  gap["missing"],
                        "gap_matched":  gap["matched"],
                        "gap_coverage": gap["coverage"],
                        "gap_total":    gap["total"],
                        # ── Détails ───────────────────────────────────────────
                        "contract":    details.get("contract", "")    or job.get("_emp_contract", ""),
                        "experience":  details.get("experience", "")  or job.get("_emp_experience", ""),
                        "education":   details.get("education", "")   or job.get("_emp_education", ""),
                        "pub_date":    details.get("pub_date", ""),
                        "expired":     details.get("expired", "")     or job.get("_emp_status", ""),
                        "description": details.get("description", "") or job.get("_emp_description", ""),
                        "skills_req":  details.get("skills_req", ""),
                        "skills_bon":  details.get("skills_bon", ""),
                        "all_skills":  details.get("all_skills", ""),
                        "tags":        details.get("skills_req", "") or details.get("tags", "") or ", ".join(job.get("_emp_tags", [])),  # requirements
                    }
                    await result_q.put(card)

                    # ── f. Sauvegarder le job en PostgreSQL ───────────────────
                    # FIX : le trigger PostgreSQL gère la déduplication
                    if user_id > 0:
                        logger.info(
                            f"[pipeline] Saving job to DB: user_id={user_id} "
                            f"url={card['url'][:50]}"
                        )
                        ok = await insert_job(user_id, card)
                        if ok:
                            logger.info(f"[pipeline] ✅ job saved in table 'jobs'")
                        else:
                            logger.error(f"[pipeline] ❌ Failed to save job in table 'jobs'")
                    else:
                        logger.debug("[pipeline] user_id=0 → skipping job DB save")

                except Exception as e:
                    logger.error(f"  [enrich] {job.get('url', '')[:60]}: {e}", exc_info=True)
                finally:
                    pending["n"] -= 1
                    if all_done["v"] and pending["n"] <= 0:
                        await result_q.put(None)

        # ── Scrapers (6 sources en parallèle) ─────────────────────────────────
        async def run_source(name: str, scrape_fn, session_):
            try:
                jobs = await scrape_fn(cv_title, session_)
                for job in jobs:
                    await handle_job(job, name)
            except Exception as e:
                logger.error(f"[{name}] scraper error: {e}")
            finally:
                await src_done_q.put(sse({"event": "source_done", "source": name}))

        asyncio.create_task(run_source("aijobs",    scrape_aijobs,     session))
        asyncio.create_task(run_source("remoteok",  scrape_remoteok,   session))
        asyncio.create_task(run_source("emploitic", scrape_emploitic,  session))
        asyncio.create_task(run_source("tanitjobs", scrape_tanitjobs,  session))
        asyncio.create_task(run_source("greenhouse",scrape_greenhouse, session))
        asyncio.create_task(run_source("eluta",     scrape_eluta,      session))

        # ── 6. Streaming loop ─────────────────────────────────────────────────
        job_count = 0

        while True:
            # Vider la queue des sources terminées
            while not src_done_q.empty():
                yield src_done_q.get_nowait()
                scrapers["done"] += 1

            # Vider la queue des jobs enrichis
            while not result_q.empty():
                item = result_q.get_nowait()
                if item is None:
                    logger.info(f"[pipeline] END — {job_count} jobs streamed (user_id={user_id})")
                    yield sse({"event": "done", "total": job_count})
                    return
                job_count += 1
                yield sse(item)

            # Tous les scrapers terminés → signaler all_done
            if scrapers["done"] >= NUM_SOURCES:
                all_done["v"] = True
                if pending["n"] <= 0:
                    logger.info(f"[pipeline] END — {job_count} jobs streamed (user_id={user_id})")
                    yield sse({"event": "done", "total": job_count})
                    return

            await asyncio.sleep(0.05)