# JobScan — AI Job Scanner

Real-time job scanner for **aijobs.ai** + **remoteok.com** in parallel.
BERT semantic similarity matching, SSE streaming, modal with full job details.

---

## Project Structure

```
job-scanner/
├── main.py              ← FastAPI backend + SSE
├── extraction.py        ← aijobs.ai HTML parser
├── rok_extract.py       ← remoteok.com JSON parser
├── frontend/
│   └── index.html       ← Complete SPA (no build required)
├── requirements.txt
└── README.md
```

---

## Setup — Step by Step

### Step 1 — Open in VS Code

```bash
# Clone or copy the folder
cd job-scanner
code .
```

### Step 2 — Create Python virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> First install downloads the BERT model (~90 MB). Takes 1-2 min.

### Step 4 — Copy your scraper files

Make sure these files are in the same folder as `main.py`:
- `extraction.py`   (aijobs.ai parser)
- `rok_extract.py`  (remoteok parser)

### Step 5 — Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 6 — Open in browser

```
http://localhost:8000
```

---

## How It Works

1. Type your CV title (e.g. `Data Engineer`, `ML Engineer`)
2. Click **Scan Jobs** — both pipelines launch simultaneously
3. Jobs appear in real-time as cards (3 per row)
4. Click any card → modal opens with all details
5. Modal shows score recommendation:
   - **≥ 0.80** → ✦ I recommend applying (X% compatible)
   - **< 0.80** → ⚠️ I do not recommend (only X% similar)
6. Click **Apply Now** → opens job URL in new tab
7. Click **Ignore** → closes modal

---

## Architecture — SSE Streaming

```
Browser                     FastAPI
  |                            |
  |--- GET /scan?cv_title=X -->|
  |                            |--- asyncio.gather(pipeline_aijobs, pipeline_remoteok)
  |                            |
  |<-- data: {event:"job"} ----|  ← matched job found
  |<-- data: {event:"job_update"} |  ← after extraction (aijobs only)
  |<-- data: {event:"done"} ---|  ← scan complete
```

**Why fast?**
- remoteok → single API call, instant results
- aijobs → basic card shown immediately, enriched after extraction
- Both sources run concurrently (asyncio.gather)
- No polling — SSE pushes data as soon as available

---

## Test

### Verify server is running
```bash
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":false}
```

### Test SSE endpoint
```bash
curl -N "http://localhost:8000/scan?cv_title=Data+Engineer"
# → data: {"event":"start",...}
# → data: {"event":"job","title":"...","score":0.72,...}
# → data: {"event":"done",...}
```

### Test in browser
1. Open http://localhost:8000
2. Type: `Data Engineer`
3. Click Scan Jobs
4. Wait for cards to appear (remoteok < 5s, aijobs ~20-60s)

---

## Configuration (main.py)

| Variable | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.45` | Min cosine score to show a job |
| `MAX_DAYS` | `45` | Max age of job posting in days |
| `HTTP_TIMEOUT` | `20` | Request timeout in seconds |

---

## Troubleshooting

**ModuleNotFoundError: extraction**
→ Make sure `extraction.py` and `rok_extract.py` are in the same folder as `main.py`

**No jobs appearing**
→ Try a lower threshold: change `SIMILARITY_THRESHOLD = 0.35` in `main.py`

**aijobs returns empty**
→ The site may have changed its HTML. Check `extraction.py` parser.

**Port 8000 already in use**
→ Use `uvicorn main:app --port 8001` and open http://localhost:8001
