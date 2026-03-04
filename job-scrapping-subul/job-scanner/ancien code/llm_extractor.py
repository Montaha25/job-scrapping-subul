"""
llm_extractor.py — Azure OpenAI Job Detail Extractor

KEY CHANGE vs previous version:
  Added "all_skills" field: the LLM scans the ENTIRE job page text
  (description, tags, requirements, responsibilities, title, everything)
  and returns ALL technical skills/competencies it finds.

  This fixes:
    - skills_req = "not specified"  → all_skills still populated
    - gap_skills = []               → now shows actual missing skills
    - match_score = 6%              → now uses real skills for scoring

Pipeline per job URL:
  1. Fetch the job detail page (async HTTP)
  2. Quick date check — skip if publication date > MAX_AGE_DAYS
  3. Clean HTML → readable plain text
  4. Send to Azure OpenAI GPT → structured JSON with all_skills
  5. Return normalized dict
"""

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv(Path(__file__).parent / ".env")

# ── Azure OpenAI config ────────────────────────────────────────────────────────
AZURE_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY",         "")
AZURE_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT",        "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION",     "2024-08-01-preview")
AZURE_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

print("=" * 55)
print("  LLM EXTRACTOR — Azure OpenAI Config")
print("=" * 55)
print(f"  ENDPOINT   : {AZURE_ENDPOINT or '❌ MISSING'}")
print(f"  API_VERSION: {AZURE_API_VERSION}")
print(f"  DEPLOYMENT : {AZURE_DEPLOYMENT or '❌ MISSING'}")
print(f"  API_KEY    : {'✅ set (' + AZURE_API_KEY[:6] + '...)' if AZURE_API_KEY else '❌ MISSING'}")
print("=" * 55)

MAX_PAGE_CHARS = 12_000
MAX_TOKENS     = 4_000
FETCH_TIMEOUT  = aiohttp.ClientTimeout(total=20)

NOISE_TAGS = [
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "iframe", "svg", "img", "button",
    "form", "input", "meta", "link", "picture",
]

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
}

# ─────────────────────────────────────────────────────────────────────────────
#  LLM Prompts
#  KEY ADDITION: "all_skills" — extract every tech skill from the ENTIRE text
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an advanced and extremely precise job offer data extraction engine.

Your ONLY task:
Read raw job posting text and return ONE strictly valid JSON object.

OUTPUT RULES (MANDATORY):
- Respond ONLY with raw valid JSON.
- No markdown.
- No backticks.
- No explanation.
- No comments.
- No text before or after JSON.
- JSON must be syntactically valid.

GENERAL EXTRACTION RULES:
- Never invent information.
- Never guess.
- If a STRING field is missing → return "not specified".
- If a LIST field is missing → return [].
- Trim extra spaces.
- Avoid duplicates in lists.

----------------------------------------
EXPERIENCE FIELD (VERY IMPORTANT)
----------------------------------------
Scan the ENTIRE job text carefully (title + description + requirements + responsibilities + benefits + footer).

Extract ANY mention of:
- Years of experience (e.g. 2+ years, 5 years minimum, 3-5 years)
- Seniority level (Junior, Mid-level, Senior, Lead, Principal, Staff)
- Experience domains (e.g. "6+ years in backend development", "engineering management experience")

Combine ALL findings into ONE concise string.
Examples:
"3+ years; Senior level, 3+ years of"
"5 years software engineering; Lead"
"Junior; 1-2 years"

CRITICAL:
- If ANY experience hint exists anywhere → NEVER return "not specified".
- Include combined experience and seniority in one string; do not separate.
- Include management experience if mentioned.

----------------------------------------
SALARY FIELD (VERY IMPORTANT)
----------------------------------------
- Extract the annual base salary range or total compensation (OTE/bonus included if specified).
- Keep full numbers and currency (e.g. "$300,000 - $485,000 USD").
- If multiple ranges are mentioned, take the one specific to the role.
- If missing → return "not specified".
- Do not truncate to $300 or any invalid value.

----------------------------------------
ALL_SKILLS FIELD (MOST IMPORTANT FIELD)
----------------------------------------
Scan the ENTIRE job text thoroughly, including:
- Job title, description, responsibilities, requirements, qualifications, nice-to-have, technologies, tools, frameworks, programming languages, certifications, methodologies, platforms, databases, cloud providers, DevOps tools, AI/ML tools, any technical domain expertise.

Extract EVERY technical skill, including but not limited to:
Programming languages (Python, Java, C++, TypeScript, Rust, etc.)
Frameworks (React, Spring Boot, Django, Angular, etc.)
Libraries (TensorFlow, PyTorch, Pandas, etc.)
Cloud (AWS, Azure, GCP)
DevOps (Docker, Kubernetes, CI/CD)
Databases (PostgreSQL, MySQL, MongoDB, Redis)
Concepts (REST API, Microservices, Machine Learning, Data Analysis)
Tools (Git, Linux, Jira, Jenkins)
Architectures (Event-driven, Serverless, Distributed systems)
Protocols (HTTP, HTTPS, GraphQL)
Data tools (Spark, Hadoop, Airflow)
Security (OAuth, JWT)
AI/ML specific tools (LLM inference, batching, caching, GPUs, TPUs, Trainium)

Rules:
- Include ALL mentioned technologies, even if optional.
- Include both required and optional skills.
- Do NOT filter or summarize.
- Return a clean list of short strings (1–3 words each).
- Remove duplicates.
- Preserve correct casing (Python not python).

----------------------------------------
SKILLS_REQ FIELD
----------------------------------------
Extract ONLY skills explicitly marked as:
- Required
- Mandatory
- Must have
- Required qualifications

----------------------------------------
SKILLS_BON FIELD
----------------------------------------
Extract ONLY skills explicitly marked as:
- Preferred
- Nice to have
- Bonus
- Plus

----------------------------------------
TAGS FIELD
----------------------------------------
Extract category or domain tags explicitly shown on the page.
If none are visible → return [].

----------------------------------------
FINAL INSTRUCTION
----------------------------------------
Return ONE clean JSON object with the following fields:
- TITLE
- LOCATION
- COMPANY
- DESCRIPTION
- EXPERIENCE
- ALL_SKILLS
- SKILLS_REQ
- SKILLS_BON
- TAGS
- SALARY

Do not include any extra fields or metadata. Ensure all fields follow the rules above.
"""
def _build_prompt(page_text: str, url: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"""Today's date: {today}
Job URL: {url}

Raw job posting text (read EVERYTHING carefully):
---
{page_text}
---

Return ONLY this JSON object (no markdown, no backticks, no extra text):
{{
  "title":       "job title or not specified",
  "company":     "company name or not specified",
  "location":    "City, Country or not specified",
  "remote":      "Remote 🌍 | Hybrid 🏠🏢 | On-site 🏢 | Full Remote — Worldwide 🌍 | not specified",
  "contract":    "CDI | CDD | Internship | Alternance | Freelance | Full-time | Part-time | not specified| stage ",
  "salary":      "e.g. '$90,000 – $120,000 / yr' or not specified",
  "experience":  "Extract ALL experience requirements from the text. e.g. '3+ years engineering management, 6+ years software development' — combine ALL years/experience mentions found ANYWHERE in requirements section into one string. Never return 'not specified' if years or experience level is mentioned.",
  "education":   "e.g. 'Master degree' | 'Bachelor' | 'PhD' | not specified",
  "pub_date":    "YYYY-MM-DD or not specified or in forme february or février in french ",
  "description": "COPY the full job description text EXACTLY as written on the page — do NOT summarize, do NOT shorten. Include responsibilities, requirements, about the company, everything. Preserve line breaks with \\n.",
  "all_skills":  ["skill1", "skill2", "skill3", ...],
  "skills_req":  ["skill1", "skill2"],
  "skills_bon":  ["skill1", "skill2"],
  "tags":        ["tag1", "tag2"]
}}

REMEMBER: "all_skills" must include EVERY technology/skill/tool found ANYWHERE in the text above.
If you see Python mentioned in the description → add Python.
If you see AWS in a tag → add AWS.
If you see React in requirements → add React.
Be comprehensive — this is used to compute skills gap."""


# ── Date utilities ─────────────────────────────────────────────────────────────

def _parse_date_string(text: str) -> Optional[datetime]:
    if not text:
        return None
    s = text.strip().lower()
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    now = datetime.now()
    m = re.search(r"(\d+)\s+days?\s+ago",   s)
    if m: return now - timedelta(days=int(m.group(1)))
    m = re.search(r"(\d+)\s+weeks?\s+ago",  s)
    if m: return now - timedelta(weeks=int(m.group(1)))
    m = re.search(r"(\d+)\s+months?\s+ago", s)
    if m: return now - timedelta(days=int(m.group(1)) * 30)
    if "today"     in s: return now
    if "yesterday" in s: return now - timedelta(days=1)
    if "just now"  in s: return now
    return None


def _quick_date_check(raw_html: str) -> Optional[datetime]:
    patterns = [
        r"(?:job\s+posted|date\s+posted|posted\s+on|published)[:\s]+([^\n<]{3,50})",
        r"\b(\d{1,3})\s+days?\s+ago\b",
        r"\b(\d{1,2})\s+weeks?\s+ago\b",
        r"\b(\d{1,2})\s+months?\s+ago\b",
        r"\b(20\d{2}-\d{2}-\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, raw_html, re.I)
        if m:
            candidate = m.group(1) if m.lastindex else m.group(0)
            parsed    = _parse_date_string(candidate.strip())
            if parsed:
                return parsed
    return None


# ── HTML cleaning ──────────────────────────────────────────────────────────────

def _clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    text  = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    clean = "\n".join(lines)
    if len(clean) > MAX_PAGE_CHARS:
        clean = clean[:MAX_PAGE_CHARS] + "\n[... content truncated ...]"
    return clean


def _make_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key        = AZURE_API_KEY,
        azure_endpoint = AZURE_ENDPOINT,
        api_version    = AZURE_API_VERSION,
    )


# ── Main extraction entry point ────────────────────────────────────────────────

async def extract_with_llm(
    url:     str,
    session: aiohttp.ClientSession,
    cutoff:  datetime,
) -> Optional[dict]:
    # ── 1. Fetch page ─────────────────────────────────────────────
    try:
        async with session.get(
            url,
            headers         = SCRAPE_HEADERS,
            timeout         = FETCH_TIMEOUT,
            allow_redirects = True,
        ) as resp:
            if resp.status in (404, 410):
                print(f"  [SKIP] {resp.status} — {url[:70]}")
                return None
            if resp.status == 429:
                print(f"  [SKIP] 429 rate-limited — {url[:70]}")
                return None
            if resp.status != 200:
                print(f"  [SKIP] HTTP {resp.status} — {url[:70]}")
                return None
            raw_html = await resp.text()
    except aiohttp.ClientError as e:
        print(f"  [ERROR] Fetch failed: {str(e)[:60]}")
        return None
    except Exception as e:
        print(f"  [ERROR] Unexpected: {str(e)[:60]}")
        return None

    # ── 2. Quick date check ───────────────────────────────────────
    pub_datetime = _quick_date_check(raw_html)
    if pub_datetime and pub_datetime < cutoff:
        age = (datetime.now() - pub_datetime).days
        print(f"  [SKIP] Too old ({age} days) — {url[:70]}")
        return None

    # ── 3. Clean HTML → text ──────────────────────────────────────
    page_text = _clean_html_to_text(raw_html)
    if len(page_text) < 100:
        print(f"  [SKIP] Page too short (blocked?) — {url[:70]}")
        return None

    # ── 4. Call Azure OpenAI ──────────────────────────────────────
    try:
        client = _make_client()
        async with client as azure:
            response = await azure.chat.completions.create(
                model           = AZURE_DEPLOYMENT,
                max_tokens      = MAX_TOKENS,
                temperature     = 0,
                response_format = {"type": "json_object"},
                messages        = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": _build_prompt(page_text, url)},
                ],
            )
        raw_response = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM ERROR] {type(e).__name__}: {str(e)[:300]}")
        return None

    raw_response = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw_response, flags=re.MULTILINE
    ).strip()

    # ── 5. Parse JSON ─────────────────────────────────────────────
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"  [JSON ERROR] {e} — {url[:70]}")
        return None

    # ── 6. Normalize ──────────────────────────────────────────────
    result = _normalize(data, cutoff)

    # Debug: show extracted skills
    if result.get("all_skills"):
        print(f"  [skills] {len(result['all_skills_list'])} skills found: "
              f"{result['all_skills_list'][:6]}")
    else:
        print(f"  [skills] ⚠ No skills extracted for {url[:50]}")

    return result


def _normalize(data: dict, cutoff: datetime) -> dict:
    def _csv(value) -> str:
        if isinstance(value, list):
            return ", ".join(str(v).strip() for v in value if v)
        if isinstance(value, str):
            return value
        return ""

    def _list(value) -> list:
        """Always return a list of clean strings."""
        if isinstance(value, list):
            return [str(v).strip() for v in value if v and str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [s.strip() for s in value.split(",") if s.strip()]
        return []

    def _str(value, default="not specified") -> str:
        if value and str(value).strip() and str(value).strip().lower() != "not specified":
            return str(value).strip()
        return default

    pub_date_str  = _str(data.get("pub_date"))
    expired_label = "No ✅"
    if pub_date_str != "not specified":
        try:
            pub_dt = datetime.strptime(pub_date_str, "%Y-%m-%d")
            if pub_dt < cutoff:
                expired_label = "Yes ⚠️"
        except ValueError:
            pass

    # Merge all_skills + skills_req + skills_bon into a unified list
    all_skills_raw = _list(data.get("all_skills"))
    skills_req_raw = _list(data.get("skills_req"))
    skills_bon_raw = _list(data.get("skills_bon"))
    tags_raw       = _list(data.get("tags"))

    # Deduplicated union — all_skills is the source of truth for gap analysis
    seen    = set()
    unified = []
    for s in (all_skills_raw + skills_req_raw + skills_bon_raw):
        norm = s.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            unified.append(s)

    # skills_req fallback: if empty but all_skills has content, use all_skills
    skills_req_final = skills_req_raw if skills_req_raw else unified

    return {
        # Core identity
        "title":     _str(data.get("title")),
        "company":   _str(data.get("company")),
        # Location & work mode
        "location":  _str(data.get("location")),
        "remote":    _str(data.get("remote")),
        # Contract & compensation
        "contract":  _str(data.get("contract")),
        "salary":    _str(data.get("salary")),
        # Requirements
        "experience": _str(data.get("experience")),
        "education":  _str(data.get("education")),
        # Dates
        "pub_date":  pub_date_str,
        "expired":   expired_label,
        # Content
        "description": _str(data.get("description")),
        # Skills — THREE levels:
        # all_skills     : every skill found anywhere in the job text (gap analysis)
        # skills_req      : explicitly required (or all_skills if empty)
        # skills_bon      : nice-to-have
        "all_skills":      _csv(unified),          # string for build_job_text()
        "all_skills_list": unified,                # list for compute_skills_gap()
        "skills_req":      _csv(skills_req_final), # string for frontend pills
        "skills_bon":      _csv(skills_bon_raw),
        "tags":            _csv(tags_raw),
    }