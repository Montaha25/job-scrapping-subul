"""
llm_extractor.py — Azure OpenAI Job Detail Extractor

KEY CHANGES vs previous version:
  1. Added "industry" field: LLM infers the company sector/domain
     (e.g. "Cybersecurity", "FinTech", "Cloud Infrastructure", "HealthTech")
     from the company name + job description.
     This fixes: industry = "" in the jobs table.

  2. Added "all_skills" field: the LLM scans the ENTIRE job page text
     and returns ALL technical skills/competencies it finds.

  3. "industry" is used as the company name/sector stored in jobs.industry column.
     Priority: LLM-extracted company name → inferred sector.

Pipeline per job URL:
  1. Fetch the job detail page (async HTTP)
  2. Quick date check — skip if publication date > MAX_AGE_DAYS
  3. Clean HTML → readable plain text
  4. Send to Azure OpenAI GPT → structured JSON with all_skills + industry
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
#  LLM System Prompt
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
INDUSTRY FIELD (IMPORTANT — NEW)
----------------------------------------
Extract the company name AND infer the business sector/domain.

Format: "CompanyName — Sector"
Examples:
  "Okta — Cybersecurity / Identity"
  "Alpaca — FinTech / Trading"
  "Anthropic — Artificial Intelligence"
  "Stripe — FinTech / Payments"
  "Airbnb — Travel / Marketplace"
  "Databricks — Data / Cloud"
  "Doctolib — HealthTech"

Rules:
- ALWAYS include the company name (e.g. "Okta", "Alpaca", "Stripe")
- Then add " — " and the sector you infer from the description
- If company is not findable → just put the inferred sector
- Never return empty string or "not specified" for industry

----------------------------------------
EXPERIENCE FIELD (VERY IMPORTANT)
----------------------------------------
Scan the ENTIRE job text carefully.
Extract ANY mention of years or seniority level.
Combine ALL findings into ONE concise string.
Examples: "3+ years; Senior level", "5 years software engineering; Lead"
CRITICAL: If ANY experience hint exists → NEVER return "not specified".

----------------------------------------
SALARY FIELD (VERY IMPORTANT)
----------------------------------------
- Extract full salary range with currency.
- Keep full numbers (e.g. "$300,000 - $485,000 USD").
- If missing → return "not specified".

----------------------------------------
ALL_SKILLS FIELD (MOST IMPORTANT FIELD)
----------------------------------------
Scan the ENTIRE job text thoroughly. Extract EVERY technical skill:
- Programming languages, Frameworks, Libraries
- Cloud (AWS, Azure, GCP), DevOps tools
- Databases, Data tools, AI/ML tools
- Concepts (REST API, Microservices, etc.)

Rules:
- Include ALL mentioned technologies, even if optional.
- Return a clean list of short strings (1-3 words each).
- Remove duplicates. Preserve correct casing (Python not python).

----------------------------------------
SKILLS_REQ / SKILLS_BON FIELDS
----------------------------------------
- skills_req: ONLY skills explicitly marked as required/mandatory/must-have
- skills_bon: ONLY skills explicitly marked as preferred/nice-to-have/bonus

----------------------------------------
TAGS FIELD
----------------------------------------
Category or domain tags shown on the page. If none → return [].
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
  "industry":    "CompanyName — Sector (e.g. 'Okta — Cybersecurity', 'Alpaca — FinTech'). NEVER empty.",
  "location":    "City, Country or not specified",
  "remote":      "Remote 🌍 | Hybrid 🏠🏢 | On-site 🏢 | Full Remote — Worldwide 🌍 | not specified",
  "contract":    "CDI | CDD | Internship | Alternance | Freelance | Full-time | Part-time | not specified",
  "salary":      "e.g. '$90,000 – $120,000 / yr' or not specified",
  "experience":  "ALL experience requirements combined. e.g. '3+ years engineering; Senior'. Never 'not specified' if any hint exists.",
  "education":   "e.g. 'Master degree' | 'Bachelor' | 'PhD' | not specified",
  "pub_date":    "YYYY-MM-DD or not specified",
  "description": "COPY the full job description text EXACTLY as written — do NOT summarize. Preserve line breaks with \\n.",
  "all_skills":  ["skill1", "skill2", "skill3"],
  "skills_req":  ["skill1", "skill2"],
  "skills_bon":  ["skill1", "skill2"],
  "tags":        ["tag1", "tag2"]
}}

CRITICAL for "industry": always include the company name.
Examples of correct "industry" values:
  "Okta — Cybersecurity / Identity Management"
  "Alpaca — FinTech / Algorithmic Trading"
  "Anthropic — Artificial Intelligence / Safety"
  "Stripe — FinTech / Payment Infrastructure"
  "Airbnb — Travel / Marketplace"
  "Kodify Media Group — Media / Entertainment"

REMEMBER: "all_skills" must include EVERY technology/skill/tool found ANYWHERE in the text."""


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

    # Debug: show extracted industry + skills
    industry_val = result.get("industry", "")
    print(f"  [industry] {industry_val[:60] if industry_val else '⚠ EMPTY'}")
    if result.get("all_skills"):
        print(f"  [skills] {len(result['all_skills_list'])} skills: "
              f"{result['all_skills_list'][:5]}")
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
        if isinstance(value, list):
            return [str(v).strip() for v in value if v and str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [s.strip() for s in value.split(",") if s.strip()]
        return []

    def _str(value, default="not specified") -> str:
        if value and str(value).strip() and str(value).strip().lower() not in ("not specified", ""):
            return str(value).strip()
        return default

    # ── industry : FIX PRINCIPAL ──────────────────────────────────────────────
    # Le LLM retourne "CompanyName — Sector"
    # On stocke l'intégralité dans industry (= colonne jobs.industry = nom société + secteur)
    raw_industry = data.get("industry") or data.get("company") or ""
    industry_val = _str(raw_industry, default="")

    # Fallback: si le LLM a mis le company dans "company" et rien dans "industry"
    if not industry_val:
        company_val  = _str(data.get("company") or "", default="")
        industry_val = company_val

    pub_date_str  = _str(data.get("pub_date"))
    expired_label = "No ✅"
    if pub_date_str != "not specified":
        try:
            pub_dt = datetime.strptime(pub_date_str, "%Y-%m-%d")
            if pub_dt < cutoff:
                expired_label = "Yes ⚠️"
        except ValueError:
            pass

    # Merge all_skills + skills_req + skills_bon into unified list
    all_skills_raw = _list(data.get("all_skills"))
    skills_req_raw = _list(data.get("skills_req"))
    skills_bon_raw = _list(data.get("skills_bon"))
    tags_raw       = _list(data.get("tags"))

    # Deduplicated union
    seen    = set()
    unified = []
    for s in (all_skills_raw + skills_req_raw + skills_bon_raw):
        norm = s.lower().strip()
        if norm and norm not in seen:
            seen.add(norm)
            unified.append(s)

    # skills_req fallback: if empty but all_skills has content → use all_skills
    skills_req_final = skills_req_raw if skills_req_raw else unified

    return {
        # ── Core identity ─────────────────────────────────────────────────────
        "title":     _str(data.get("title")),
        # FIX : industry = company name + sector (jamais vide)
        "industry":  industry_val,
        # ── Location & work mode ──────────────────────────────────────────────
        "location":  _str(data.get("location")),
        "remote":    _str(data.get("remote")),
        # ── Contract & compensation ───────────────────────────────────────────
        "contract":  _str(data.get("contract")),
        "salary":    _str(data.get("salary")),
        # ── Requirements ──────────────────────────────────────────────────────
        "experience": _str(data.get("experience")),
        "education":  _str(data.get("education")),
        # ── Dates ─────────────────────────────────────────────────────────────
        "pub_date":  pub_date_str,
        "expired":   expired_label,
        # ── Content ───────────────────────────────────────────────────────────
        "description": _str(data.get("description")),
        # ── Skills ────────────────────────────────────────────────────────────
        "all_skills":      _csv(unified),
        "all_skills_list": unified,
        "skills_req":      _csv(skills_req_final),
        "skills_bon":      _csv(skills_bon_raw),
        "tags":            _csv(tags_raw),
    }