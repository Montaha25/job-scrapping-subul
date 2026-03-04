"""
app.py — IPython frontend for aijobs.ai
3 cards per row, expandable Learn More with all fields
Score affiché : cosine * 100 avec virgule  ex: 72,45%
"""

import queue
import threading
from IPython.display import display, HTML

_job_queue  = queue.Queue()
_jobs_cache = []
_display_id = None


def push_job(job: dict):
    _jobs_cache.append(job)
    _job_queue.put(("job", job))


def push_done():
    _job_queue.put(("done", {}))


# ── Score formatter ───────────────────────────────────────────────────────────

def _fmt_score(score_raw) -> str:
    """
    Accepte float (0.7245) ou string ('72,45' / '72.45').
    Retourne toujours une string 'XX,XX' avec virgule.
    """
    try:
        if isinstance(score_raw, str):
            # Déjà formaté avec virgule : '72,45' → garder
            if ',' in score_raw:
                return score_raw
            # Formaté avec point : '72.45' → remplacer
            return score_raw.replace('.', ',')
        # Float cosine 0..1
        val = float(score_raw)
        if val <= 1.0:
            val = val * 100
        return f"{val:.2f}".replace('.', ',')
    except Exception:
        return "—"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pill(text, bg, color, border):
    return (f'<span style="font-size:10px;padding:2px 7px;border-radius:4px;'
            f'background:{bg};color:{color};border:1px solid {border};'
            f'white-space:nowrap">{text}</span>')


def _row(icon, label, value, color="#c9d1e8"):
    if not value or value in ("Not specified", ""):
        return ""
    return (
        f'<div style="display:flex;gap:8px;align-items:flex-start;'
        f'padding:5px 0;border-bottom:1px solid #1a1d26">'
        f'<span style="font-size:12px;min-width:18px;margin-top:1px">{icon}</span>'
        f'<div style="flex:1">'
        f'<span style="font-size:10px;color:#5a607a;text-transform:uppercase;'
        f'letter-spacing:.06em;display:block;margin-bottom:2px">{label}</span>'
        f'<span style="font-size:12px;color:{color};font-weight:500;'
        f'line-height:1.4">{value}</span>'
        f'</div></div>'
    )


def _skills_block(label, skills_str, bg, color, border):
    items = [s.strip() for s in skills_str.split(",") if s.strip()]
    if not items:
        return ""
    pills = "".join(_pill(s, bg, color, border) for s in items)
    return (
        f'<div style="padding:5px 0;border-bottom:1px solid #1a1d26">'
        f'<span style="font-size:10px;color:#5a607a;text-transform:uppercase;'
        f'letter-spacing:.06em;display:block;margin-bottom:5px">{label}</span>'
        f'<div style="display:flex;gap:4px;flex-wrap:wrap">{pills}</div></div>'
    )


# ── Card ──────────────────────────────────────────────────────────────────────

def _card_html(job, idx):
    score_raw = job.get("score", job.get("score_display", 0))
    score_str = _fmt_score(score_raw)        # "72,45"
    fill_pct  = int(float(str(score_str).replace(',', '.')) or 0)

    title    = (job.get("title")   or "—").replace("<","&lt;").replace(">","&gt;")
    industry = (job.get("industry") or job.get("company") or "").replace("<","&lt;").replace(">","&gt;")
    url      = job.get("url",      "#")
    time_ago = job.get("time_ago", job.get("pub_date", ""))
    pub_date = job.get("pub_date", "")

    salary        = job.get("salary",          "Not specified")
    location      = job.get("location",        "")
    remote        = job.get("remote",          "Remote 🌍")
    contract_type = job.get("contract",        job.get("contract_type", "Full-time"))
    experience    = job.get("experience",      "Not specified")
    education     = job.get("education",       "Not specified")
    expired       = job.get("expired",         "No ✅")
    skills_req    = job.get("skills_req",      job.get("skills_required", "")) or job.get("skills", "")
    skills_bonus  = job.get("skills_bon",      job.get("skills_bonus", ""))
    tags_raw      = job.get("tags",            "")
    description   = (job.get("description",   "") or "").replace("<","&lt;").replace(">","&gt;")

    # Score color
    score_float = float(str(score_str).replace(',', '.')) if score_str != '—' else 0
    if score_float >= 80:   color = "#00e5a0"
    elif score_float >= 65: color = "#7b61ff"
    else:                   color = "#ff6b35"

    # Expired display
    if expired in ("Yes", "Yes ⚠️"):
        expired_display = "Yes ⚠️"
        expired_color   = "#ff6b35"
    else:
        expired_display = "No ✅"
        expired_color   = "#00e5a0"

    # Company initial badge
    initial   = industry[:1].upper() if industry else "A"
    logo_html = (
        f'<div style="width:34px;height:34px;border-radius:7px;flex-shrink:0;'
        f'background:linear-gradient(135deg,#7b61ff,#00e5a0);'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-weight:800;font-size:13px;color:#0b0c10">{initial}</div>'
    )

    # Tags mini (max 3 in header)
    tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()][:3]
    tags_mini = "".join(_pill(t, f"{color}11", color, f"{color}55") for t in tags_list)

    # All tags for learn more
    all_tags  = [t.strip() for t in tags_raw.split(",") if t.strip()]
    tags_full = "".join(_pill(t, f"{color}11", color, f"{color}55") for t in all_tags)
    tags_block = (
        f'<div style="padding:5px 0;border-bottom:1px solid #1a1d26">'
        f'<span style="font-size:10px;color:#5a607a;text-transform:uppercase;'
        f'letter-spacing:.06em;display:block;margin-bottom:5px">Tags</span>'
        f'<div style="display:flex;gap:4px;flex-wrap:wrap">{tags_full}</div></div>'
    ) if tags_full else ""

    desc_block = (
        f'<div style="padding:5px 0;border-bottom:1px solid #1a1d26">'
        f'<span style="font-size:10px;color:#5a607a;text-transform:uppercase;'
        f'letter-spacing:.06em;display:block;margin-bottom:5px">Description</span>'
        f'<div style="font-size:11px;color:#8892b0;line-height:1.6;'
        f'background:#0b0c10;border-radius:7px;padding:9px;'
        f'max-height:200px;overflow-y:auto">{description}</div></div>'
    ) if description else ""

    card_id = f"aj{idx}"

    return f"""
<div id="{card_id}" style="
    background:#161922;border:1px solid #1e2130;border-top:3px solid {color};
    border-radius:12px;padding:13px;display:flex;flex-direction:column;gap:9px;
    animation:slideIn .4s ease forwards;opacity:0;transform:translateY(12px);
    transition:box-shadow .2s;"
  onmouseover="this.style.boxShadow='0 4px 24px {color}18'"
  onmouseout="this.style.boxShadow='none'">

  <!-- HEADER -->
  <div style="display:flex;gap:9px;align-items:flex-start">
    {logo_html}
    <div style="flex:1;min-width:0">
      <div style="font-size:12px;font-weight:600;line-height:1.35;color:#e8eaf0;
                  overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;
                  -webkit-box-orient:vertical" title="{title}">{title}</div>
      <div style="font-size:10px;color:#8892b0;margin-top:2px">at {industry}</div>
    </div>
    <!-- SCORE avec virgule ex: 72,45% -->
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;flex-shrink:0">
      <span style="font-size:18px;font-weight:800;font-family:monospace;
                   color:{color};line-height:1">{score_str}%</span>
      <div style="width:48px;height:3px;background:#1e2130;border-radius:2px;overflow:hidden">
        <div style="height:100%;width:{fill_pct}%;background:{color};border-radius:2px"></div>
      </div>
      <span style="font-size:9px;color:#5a607a;font-family:monospace">{time_ago}</span>
    </div>
  </div>

  <!-- QUICK INFO -->
  <div style="display:flex;flex-direction:column;gap:3px">
    <div style="font-size:11px;color:#f59e0b;font-weight:500">
      💰 {salary if salary != "Not specified"
          else "<span style='color:#5a607a;font-weight:400'>Salary not disclosed</span>"}
    </div>
    <div style="font-size:10px;color:#8892b0">
      📍 {location if location else "—"} &nbsp;·&nbsp;
      <span style="color:#00e5a0">{remote}</span> &nbsp;·&nbsp;
      <span style="color:#7b61ff">{contract_type}</span>
    </div>
  </div>

  <!-- TAGS MINI -->
  <div style="display:flex;gap:4px;flex-wrap:wrap">{tags_mini}</div>

  <!-- LEARN MORE PANEL -->
  <div id="{card_id}-panel" style="display:none;flex-direction:column;gap:0;
       border-top:1px solid #1e2130;padding-top:10px;margin-top:2px">

    {_row("💰", "Salary",       salary,          "#f59e0b")}
    {_row("📍", "Location",     location)}
    {_row("🌍", "Remote",       remote,           "#00e5a0")}
    {_row("📄", "Contract",     contract_type,    "#7b61ff")}
    {_row("⏱️", "Experience",  experience,        "#5a8fff")}
    {_row("🎓", "Education",    education,         "#5a8fff")}
    {_row("📅", "Published",    pub_date)}
    {_row("🔴", "Expired",      expired_display,  expired_color)}

    {tags_block}
    {_skills_block("✅ Required Skills", skills_req,   "#00e5a011","#00e5a0","#00e5a044")}
    {_skills_block("⭐ Bonus Skills",    skills_bonus, "#7b61ff11","#7b61ff","#7b61ff44")}
    {desc_block}

    <!-- CTA -->
    <a href="{url}" target="_blank" style="
      display:block;text-align:center;margin-top:8px;
      padding:8px;background:linear-gradient(135deg,#7b61ff,#00e5a0);
      border-radius:8px;font-size:12px;font-weight:700;
      color:#0b0c10;text-decoration:none;transition:opacity .2s"
      onmouseover="this.style.opacity='.85'"
      onmouseout="this.style.opacity='1'">
      View Full Job Offer →
    </a>
  </div>

  <!-- LEARN MORE BUTTON -->
  <button id="{card_id}-btn"
    onclick="(function(){{
      var p=document.getElementById('{card_id}-panel');
      var b=document.getElementById('{card_id}-btn');
      if(p.style.display==='none'){{
        p.style.display='flex';b.innerHTML='&#9650; Show less';
      }}else{{
        p.style.display='none';b.innerHTML='&#9660; Learn more';
      }}
    }})()"
    style="background:transparent;border:1px solid #1e2130;border-radius:6px;
           color:#5a607a;font-size:10px;padding:5px 8px;cursor:pointer;
           width:100%;font-family:monospace;transition:all .2s"
    onmouseover="this.style.borderColor='{color}';this.style.color='{color}'"
    onmouseout="this.style.borderColor='#1e2130';this.style.color='#5a607a'">
    &#9660; Learn more
  </button>

</div>"""


# ── Full page ─────────────────────────────────────────────────────────────────

def _full_html(jobs, status="live"):
    count = len(jobs)

    raw_scores = []
    for j in jobs:
        s = j.get("score", j.get("score_display", 0))
        try:
            v = float(str(s).replace(',', '.'))
            raw_scores.append(v if v > 1 else v * 100)
        except Exception:
            pass

    best = max(raw_scores, default=0)
    avg  = (sum(raw_scores) / len(raw_scores)) if raw_scores else 0

    # Format best/avg with comma
    best_str = f"{best:.2f}".replace('.', ',') if count else "—"
    avg_str  = f"{avg:.2f}".replace('.', ',')  if count else "—"

    if status == "live":
        pill_color = "#00e5a0"; pill_txt = "● LIVE"; dot_anim = "animation:pulse 1.4s infinite;"
    else:
        pill_color = "#7b61ff"; pill_txt = "✦ DONE"; dot_anim = ""

    cards_html = "".join(_card_html(j, i) for i, j in enumerate(jobs))

    empty = """
    <div style="text-align:center;padding:4rem 2rem;color:#5a607a;grid-column:1/-1">
      <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
      <p style="font-family:monospace;font-size:13px">
        Waiting for pipeline…<br>Run the scan to start.
      </p>
    </div>""" if not jobs else ""

    stats = "".join(f"""
    <div style="flex:1;min-width:90px;background:#13151c;border:1px solid #1e2130;
                border-radius:10px;padding:10px 14px">
      <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                  color:#00e5a0;line-height:1">{v}</div>
      <div style="font-size:10px;color:#5a607a;margin-top:3px;
                  text-transform:uppercase;letter-spacing:.06em">{lbl}</div>
    </div>""" for v, lbl in [
        (count,    "Matched"),
        (best_str, "Best Score %"),
        (avg_str,  "Avg Score %"),
    ])

    done_banner = ("""
    <div style="text-align:center;padding:12px;margin-top:4px;
                background:linear-gradient(135deg,#7b61ff22,#00e5a011);
                border:1px solid #7b61ff55;border-radius:10px;
                font-family:monospace;font-size:12px;color:#7b61ff;
                grid-column:1/-1">
      ✦ Pipeline complete — all jobs have been processed
    </div>""" if status == "done" else "")

    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
  @keyframes slideIn {{ to {{ opacity:1; transform:translateY(0) }} }}
  @keyframes pulse   {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.4;transform:scale(1.5)}} }}
  .jp-OutputArea-output {{ overflow-anchor: none !important; }}
  * {{ overflow-anchor: none; }}
</style>
<div style="background:#0b0c10;border-radius:16px;padding:20px 22px;
            font-family:'DM Sans',sans-serif;
            border:1px solid #1e2130;position:relative;overflow:hidden">

  <div style="position:absolute;inset:0;
    background-image:linear-gradient(#1e213030 1px,transparent 1px),
                     linear-gradient(90deg,#1e213030 1px,transparent 1px);
    background-size:40px 40px;pointer-events:none;z-index:0"></div>

  <div style="position:relative;z-index:1">

    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
      <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:#e8eaf0">
        AI<span style="color:#00e5a0">Jobs</span><span style="color:#7b61ff">Scan</span>
        <span style="font-size:11px;font-weight:400;color:#5a607a;
                     font-family:monospace;margin-left:8px">aijobs.ai + remoteok</span>
      </div>
      <div style="display:flex;align-items:center;gap:7px;padding:4px 12px;
                  background:#13151c;border:1px solid {pill_color}55;border-radius:99px;
                  font-family:monospace;font-size:11px;color:{pill_color}">
        <div style="width:6px;height:6px;border-radius:50%;
                    background:{pill_color};{dot_anim}"></div>
        {pill_txt}
      </div>
    </div>

    <!-- Stats -->
    <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap">{stats}</div>

    <!-- Label -->
    <div style="font-family:monospace;font-size:10px;color:#5a607a;
                text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px">
      // real-time feed · 3 columns · score = cosine × 100 (comma decimal)
    </div>

    <!-- 3-column grid -->
    <div style="display:grid;grid-template-columns:repeat(3,1fr);
                gap:12px;align-items:start">
      {empty}
      {cards_html}
      {done_banner}
    </div>

  </div>
</div>"""


# ── Display ───────────────────────────────────────────────────────────────────

def display_front():
    """Call in a cell BEFORE running the pipeline."""
    global _display_id, _jobs_cache
    _jobs_cache = []
    _display_id = display(HTML(_full_html([])), display_id=True)

    def _updater():
        while True:
            try:
                kind, job = _job_queue.get(timeout=300)
                if kind == "done":
                    _display_id.update(HTML(_full_html(_jobs_cache, status="done")))
                    break
                _display_id.update(HTML(_full_html(_jobs_cache, status="live")))
            except queue.Empty:
                break

    threading.Thread(target=_updater, daemon=True).start()
    print("✅ AIJobsScan ready — run the pipeline to start")