"""
matcher.py — Fine-tuned CV/Job matching model + Skills gap analysis

FORMAT D'ENTRAI NEMENT EXACT (notebook cell 9) :

  resume_to_text() :
    "Role: {role} | Seniority: {seniority} | Experience: {years} years |
     Industry: {industry} | Education: {education} | Skills: {skills} |
     Summary: {summary} | {bullets}"

  job_to_text() :
    "Title: {title} | Seniority: {seniority} | Industry: {industry} |
     Must have: {must_have_skills} | Nice to have: {nice_to_have} |
     {description} | Responsibilities: {resp} | Requirements: {reqs}"

AMELIORATIONS v4 (Idée 1 + Idée 2) :
  Idée 2 — Normalisation canonique AVANT envoi au modèle
    "sklearn" → "scikit-learn", "k8s" → "Kubernetes", "torch" → "PyTorch"
    Appliqué côté CV ET côté Job → mêmes tokens des deux côtés.

  Idée 1 — Expansion des abréviations
    "ML" → "ML Machine Learning", "NLP" → "NLP Natural Language Processing"
    Plus de tokens en commun → score monte.

IMPORTANT :
  - predict() est 100% SYNCHRONE (appelé via asyncio.to_thread dans main.py)
  - JAMAIS de asyncio / event loop dans ce fichier
  - Format texte identique au notebook → le modèle reconnaît les entrées

Model (notebook cell 11, best val_r2=0.9271, test MAE ~5.8 pts) :
  BiEncoderRegressorFineTuned
    encoder   = all-MiniLM-L6-v2  (hidden=384)
    regressor = Linear(1536→768) BatchNorm ReLU Dropout
                Linear(768→256)  BatchNorm ReLU Dropout
                Linear(256→64)   ReLU
                Linear(64→1)     Sigmoid → [0,1]
"""

from __future__ import annotations
import logging
import math
import re
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

BASE_DIR      = Path(__file__).parent
MODEL_DIR     = BASE_DIR / "jobscan_model"
MODEL_PATH    = MODEL_DIR / "finetuned_model.pt"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"

BACKBONE_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH    = 256   # même que MAX_LEN dans le notebook
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
#  IDEE 2 — Vocabulaire canonique (normalise CV + Job identiquement)
# =============================================================================

CANONICAL: dict[str, str] = {
    # Python ecosystem
    "sklearn":                      "scikit-learn",
    "scikit learn":                 "scikit-learn",
    "scikitlearn":                  "scikit-learn",
    "torch":                        "PyTorch",
    "pytorch":                      "PyTorch",
    "tf":                           "TensorFlow",
    "tensorflow":                   "TensorFlow",
    "keras":                        "Keras",
    "numpy":                        "NumPy",
    "pandas":                       "Pandas",
    "fastapi":                      "FastAPI",
    # Cloud
    "amazon web services":          "AWS",
    "amazonwebservices":            "AWS",
    "aws":                          "AWS",
    "google cloud":                 "GCP",
    "google cloud platform":        "GCP",
    "gcp":                          "GCP",
    "microsoft azure":              "Azure",
    "azure":                        "Azure",
    # Containers / DevOps
    "k8s":                          "Kubernetes",
    "kubernetes":                   "Kubernetes",
    "docker":                       "Docker",
    "terraform":                    "Terraform",
    "cloudformation":               "CloudFormation",
    "cloud formation":              "CloudFormation",
    "ansible":                      "Ansible",
    "ci/cd":                        "CI/CD",
    "cicd":                         "CI/CD",
    "ci cd":                        "CI/CD",
    "github actions":               "GitHub Actions",
    "gitlab ci":                    "GitLab CI",
    "gitlab":                       "GitLab",
    "jenkins":                      "Jenkins",
    "eks":                          "Amazon EKS",
    "amazon eks":                   "Amazon EKS",
    # Databases
    "postgres":                     "PostgreSQL",
    "postgresql":                   "PostgreSQL",
    "mongo":                        "MongoDB",
    "mongodb":                      "MongoDB",
    "mysql":                        "MySQL",
    "redis":                        "Redis",
    "elasticsearch":                "Elasticsearch",
    "elastic search":               "Elasticsearch",
    "elk":                          "ELK Stack",
    "elk stack":                    "ELK Stack",
    "opensearch":                   "OpenSearch",
    # Monitoring
    "prometheus":                   "Prometheus",
    "grafana":                      "Grafana",
    "cloudwatch":                   "CloudWatch",
    "cloud watch":                  "CloudWatch",
    # Languages
    "js":                           "JavaScript",
    "javascript":                   "JavaScript",
    "ts":                           "TypeScript",
    "typescript":                   "TypeScript",
    "python":                       "Python",
    "golang":                       "Go",
    "go lang":                      "Go",
    "c#":                           "C#",
    "csharp":                       "C#",
    "dotnet":                       ".NET",
    ".net":                         ".NET",
    "node":                         "Node.js",
    "nodejs":                       "Node.js",
    "node.js":                      "Node.js",
    "shell":                        "Shell",
    "bash":                         "Bash",
    # Frontend
    "reactjs":                      "React",
    "react.js":                     "React",
    "react":                        "React",
    "vuejs":                        "Vue.js",
    "vue.js":                       "Vue.js",
    "vue":                          "Vue.js",
    "angular":                      "Angular",
    # ML / AI
    "machine learning":             "Machine Learning",
    "ml":                           "Machine Learning",
    "deep learning":                "Deep Learning",
    "dl":                           "Deep Learning",
    "nlp":                          "Natural Language Processing",
    "natural language processing":  "Natural Language Processing",
    "computer vision":              "Computer Vision",
    "llm":                          "Large Language Models",
    "large language model":         "Large Language Models",
    "large language models":        "Large Language Models",
    "rag":                          "Retrieval Augmented Generation",
    "mlops":                        "MLOps",
    "ml ops":                       "MLOps",
    "hugging face":                 "Hugging Face",
    "huggingface":                  "Hugging Face",
    "langchain":                    "LangChain",
    # Data
    "spark":                        "Apache Spark",
    "apache spark":                 "Apache Spark",
    "kafka":                        "Apache Kafka",
    "apache kafka":                 "Apache Kafka",
    "airflow":                      "Apache Airflow",
    "power bi":                     "Power BI",
    "powerbi":                      "Power BI",
    # Methods
    "rest":                         "REST API",
    "rest api":                     "REST API",
    "restful":                      "REST API",
    "graphql":                      "GraphQL",
    "microservices":                "Microservices",
    "micro services":               "Microservices",
    "git":                          "Git",
    "linux":                        "Linux",
    "iac":                          "Infrastructure as Code",
    "infrastructure as code":       "Infrastructure as Code",
    # Networking / Infra
    "istio":                        "Istio",
    "service mesh":                 "Service Mesh",
    "cdn":                          "CDN",
    "dns":                          "DNS",
    "cloud native":                 "Cloud Native",
    "envoy":                        "Envoy Proxy",
    "nginx":                        "NGINX",
    "vpc":                          "VPC networking",
    "lambda":                       "AWS Lambda",
    "aws lambda":                   "AWS Lambda",
    "serverless":                   "Serverless",
}


# =============================================================================
#  IDEE 1 — Expansions des abréviations
# =============================================================================

EXPANSIONS: dict[str, str] = {
    "ML":       "Machine Learning",
    "NLP":      "Natural Language Processing",
    "CV":       "Computer Vision",
    "DL":       "Deep Learning",
    "AI":       "Artificial Intelligence",
    "RAG":      "Retrieval Augmented Generation",
    "LLM":      "Large Language Models",
    "RL":       "Reinforcement Learning",
    "AWS":      "Amazon Web Services cloud",
    "GCP":      "Google Cloud Platform",
    "EKS":      "Amazon EKS Kubernetes",
    "K8s":      "Kubernetes container orchestration",
    "CI/CD":    "continuous integration continuous deployment pipeline",
    "IaC":      "Infrastructure as Code",
    "SQL":      "Structured Query Language database",
    "API":      "Application Programming Interface REST",
    "CDN":      "Content Delivery Network",
    "DNS":      "Domain Name System networking",
    "SRE":      "Site Reliability Engineering DevOps",
    "JS":       "JavaScript web frontend",
    "TS":       "TypeScript typed JavaScript",
    "VPC":      "Virtual Private Cloud networking",
    "ELK":      "Elasticsearch Logstash Kibana monitoring",
}


def _normalize_skill(skill: str) -> str:
    """Idée 2 : normalise vers le nom canonique."""
    return CANONICAL.get(skill.lower().strip(), skill.strip())


def normalize_skills_text(text: str) -> str:
    """Idée 2 : normalise tous les skills d'un texte CSV."""
    if not text:
        return text
    parts = [s.strip() for s in text.split(",") if s.strip()]
    return ", ".join(_normalize_skill(p) for p in parts)


def expand_abbreviations(text: str) -> str:
    """Idée 1 : ajoute les expansions des abréviations."""
    if not text:
        return text
    result = text
    for abbr, expansion in EXPANSIONS.items():
        pattern = r'\b' + re.escape(abbr) + r'\b'
        result  = re.sub(pattern, f"{abbr} {expansion}", result, flags=re.IGNORECASE)
    return result


def enrich_skills_text(text: str) -> str:
    """Pipeline : normalisation (Idée 2) → expansion (Idée 1)."""
    return expand_abbreviations(normalize_skills_text(text))


# =============================================================================
#  Model architecture — copie exacte du notebook cell 11
# =============================================================================

class BiEncoderRegressorFineTuned(nn.Module):
    def __init__(self, model_name: str = BACKBONE_NAME,
                 dropout: float = 0.2, freeze_layers: int = 4):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 384 pour MiniLM

        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Linear(hidden * 4, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def mean_pool(self, token_embeds: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeds    = (token_embeds * mask_expanded).sum(dim=1)
        sum_mask      = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeds / sum_mask

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(output.last_hidden_state, attention_mask)

    def forward(self, r_input_ids, r_attention_mask,
                j_input_ids, j_attention_mask) -> torch.Tensor:
        r_emb    = self.encode_text(r_input_ids, r_attention_mask)
        j_emb    = self.encode_text(j_input_ids, j_attention_mask)
        features = torch.cat([r_emb, j_emb, torch.abs(r_emb - j_emb), r_emb * j_emb], dim=1)
        return self.regressor(features).squeeze()


# =============================================================================
#  Load model + tokenizer
# =============================================================================

def load_model() -> tuple:
    if not MODEL_PATH.exists():
        logger.error(
            f"[matcher] Manquant : {MODEL_PATH}\n"
            f"  Attendu :\n"
            f"    {MODEL_DIR}/finetuned_model.pt\n"
            f"    {MODEL_DIR}/tokenizer/\n"
        )
        return None, None

    if not TOKENIZER_DIR.exists():
        logger.error(f"[matcher] Tokenizer dir manquant : {TOKENIZER_DIR}")
        return None, None

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        logger.info("[matcher] Tokenizer chargé OK")
    except Exception as e:
        logger.error(f"[matcher] Tokenizer error: {e}")
        return None, None

    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except Exception as e:
        logger.error(f"[matcher] Impossible de lire le checkpoint : {e}")
        return None, None

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        cfg        = ckpt.get("config", {})
        dropout    = cfg.get("dropout", 0.2)
        freeze     = cfg.get("freeze_layers", 4)
    elif isinstance(ckpt, dict):
        state_dict      = ckpt
        dropout, freeze = 0.2, 4
    elif isinstance(ckpt, BiEncoderRegressorFineTuned):
        ckpt.to(DEVICE).eval()
        return ckpt, tokenizer
    else:
        logger.error(f"[matcher] Format checkpoint inconnu : {type(ckpt)}")
        return None, None

    model = BiEncoderRegressorFineTuned(BACKBONE_NAME, dropout=dropout, freeze_layers=freeze)
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info("[matcher] Poids chargés strict=True OK")
    except RuntimeError as e:
        logger.warning(f"[matcher] strict=True échoué → strict=False : {e}")
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e2:
            logger.error(f"[matcher] Chargement échoué : {e2}")
            return None, None

    model.to(DEVICE).eval()
    logger.info(f"[matcher] Modèle prêt sur {DEVICE} OK")
    return model, tokenizer


# =============================================================================
#  Text formatting — FORMAT EXACT DU NOTEBOOK (cell 9)
#
#  resume_to_text() du notebook :
#    "Role: X | Seniority: X | Experience: X years | Industry: X |
#     Education: X | Skills: X | Summary: X | bullets"
#
#  job_to_text() du notebook :
#    "Title: X | Seniority: X | Industry: X | Must have: X |
#     Nice to have: X | description | Responsibilities: X | Requirements: X"
#
#  CRITICAL : respecter exactement ce format → le modèle reconnaît les inputs
# =============================================================================

def build_resume_text(structured: dict) -> str:
    """
    Format CV identique à resume_to_text() du notebook (cell 9).
    Idée 1+2 appliquées sur le champ Skills uniquement.

    Format :
      "Role: X | Seniority: X | Experience: X years | Industry: X |
       Education: X | Skills: X enrichis | Summary: X | bullets"
    """
    def v(key, default=""):
        val = structured.get(key, default)
        if not val or str(val).strip().lower() in ("", "not specified", "unknown"):
            return default
        return str(val).strip()

    raw_skills   = v("skills")
    # Idée 2 : normalisation canonique
    norm_skills  = normalize_skills_text(raw_skills)
    # Idée 1 : expansion des abréviations
    final_skills = expand_abbreviations(norm_skills)

    logger.debug(f"[resume] skills: '{raw_skills[:50]}' → '{final_skills[:70]}'")

    role       = v("role", "Software Engineer")
    seniority  = v("seniority", "Mid")
    experience = v("years_experience", "3")
    industry   = v("industry", "Technology")
    education  = v("education", "Bachelor")
    summary    = v("summary", role)
    bullets    = v("bullets", "")

    # Format EXACT du notebook
    text = (
        f"Role: {role} | "
        f"Seniority: {seniority} | "
        f"Experience: {experience} years | "
        f"Industry: {industry} | "
        f"Education: {education} | "
        f"Skills: {final_skills} | "
        f"Summary: {summary}"
    )
    if bullets:
        text += f" | {bullets}"

    return text


def build_job_text(details: dict) -> str:
    """
    Format job identique à job_to_text() du notebook (cell 9).
    Idée 1+2 appliquées sur Must have + Nice to have.

    Format :
      "Title: X | Seniority: X | Industry: X | Must have: X enrichis |
       Nice to have: X enrichis | description | Responsibilities: X | Requirements: X"
    """
    def v(key, default=""):
        val = details.get(key, default)
        if not val:
            return default
        s = str(val).strip()
        return default if s.lower() in ("not specified", "non renseigné", "") else s

    title       = v("title", "")
    seniority   = v("experience", "")
    industry    = v("tags", "Technology")
    description = v("description", "")
    education   = v("education", "")

    # Idée 1+2 sur les skills des deux côtés
    skills_req = enrich_skills_text(v("skills_req", ""))
    all_skills = enrich_skills_text(v("all_skills", ""))
    skills_bon = enrich_skills_text(v("skills_bon", ""))

    # must_have = skills_req si dispo, sinon all_skills
    must_have = skills_req if skills_req else all_skills

    # Format EXACT du notebook
    parts = []
    if title:       parts.append(f"Title: {title}")
    if seniority:   parts.append(f"Seniority: {seniority}")
    if industry:    parts.append(f"Industry: {industry}")
    if must_have:   parts.append(f"Must have: {must_have}")
    if skills_bon:  parts.append(f"Nice to have: {skills_bon}")
    if description: parts.append(description)
    if description: parts.append(f"Responsibilities: {description}")
    if must_have:   parts.append(f"Requirements: {must_have}")
    if education:   parts.append(f"Education: {education}")

    return " | ".join(parts)


# =============================================================================
#  Skills gap analysis (avec Idée 2 pour la comparaison)
# =============================================================================

def _norm(s: str) -> str:
    """Normalise pour comparaison : lowercase, collapse séparateurs."""
    return re.sub(r'[\s\-_./()]+', '', s.lower().strip())


def _skill_found_in_cv(job_skill: str, cv_tokens: list[str]) -> bool:
    """
    Fuzzy match après normalisation canonique (Idée 2).
    Substring dans les deux sens.
    """
    canon = _normalize_skill(job_skill)
    j     = _norm(canon)
    if not j or len(j) < 2:
        return False
    for c in cv_tokens:
        if j == c:                  return True
        if len(j) >= 3 and j in c: return True
        if len(c) >= 3 and c in j: return True
    return False


def compute_skills_gap(cv_structured: dict, job_details: dict) -> dict:
    """
    Calcule le gap de skills entre CV et job.
    Idée 2 appliquée : normalisation canonique avant comparaison.

    Source skills job (priorité) :
      1. all_skills_list (liste complète extraite par LLM)
      2. skills_req CSV (fallback)
      3. []  (rien trouvé)
    """
    job_skills: list[str] = []

    all_list = job_details.get("all_skills_list", [])
    if isinstance(all_list, list) and all_list:
        job_skills = [s.strip() for s in all_list if s.strip()]

    if not job_skills:
        req_csv    = job_details.get("skills_req", "") or ""
        job_skills = [s.strip() for s in req_csv.split(",") if s.strip()
                      and s.strip().lower() != "not specified"]

    if not job_skills:
        return {"missing": [], "matched": [], "coverage": 1.0, "total": 0}

    # Idée 2 : normaliser les skills du job
    job_skills = [_normalize_skill(s) for s in job_skills]

    # Idée 2 : normaliser les skills CV avant comparaison
    cv_raw    = cv_structured.get("skills", "") or ""
    cv_norm   = normalize_skills_text(cv_raw)
    cv_tokens = [_norm(s) for s in cv_norm.split(",") if s.strip()]
    cv_tokens = [t for t in cv_tokens if t]

    missing: list[str] = []
    matched: list[str] = []

    for skill in job_skills:
        if _skill_found_in_cv(skill, cv_tokens):
            matched.append(skill)
        else:
            missing.append(skill)

    total    = len(job_skills)
    coverage = round(len(matched) / total, 3) if total > 0 else 1.0

    logger.info(
        f"[gap] {len(matched)}/{total} matchés ({coverage:.0%}) | "
        f"manquants : {missing[:4]}"
    )

    return {
        "missing":  missing,
        "matched":  matched,
        "coverage": coverage,
        "total":    total,
    }


# =============================================================================
#  Score combiné
# =============================================================================

def compute_combined_score(ai_match: float, gap: dict) -> float:
    """
    Score final = AI_match × √(coverage)

    Notebook stats de référence (cell 18, benchmark) :
      ✅ Match Parfait       → ~75-100
      🟡 Bon match industrie → ~50-75
      🟡 Match partiel       → ~30-55
      ❌ Mauvais match       → ~0-20
      ❌ Aucun rapport       → ~0-10
    """
    coverage = gap.get("coverage", 1.0)
    total    = gap.get("total",    0)

    if total == 0:
        return round(max(0.0, min(1.0, ai_match)), 4)

    if ai_match < 0:
        return round(max(0.0, min(1.0, coverage)), 4)

    combined = ai_match * math.sqrt(max(0.0, coverage))
    return round(max(0.0, min(1.0, combined)), 4)


# =============================================================================
#  predict() — 100% SYNCHRONE
#
#  POURQUOI SYNCHRONE :
#    main.py : await asyncio.to_thread(lambda: mtch.predict(...))
#    → exécuté dans un thread séparé
#    → JAMAIS créer de event loop ici (plante le scraping)
#    → JAMAIS d'async/await dans predict()
#
#  Idée 1+2 : appliquées dans build_resume_text() et build_job_text()
#  → aucun appel réseau, aucun I/O → 100% compatible thread
# =============================================================================

def predict(model, tokenizer,
            cv_structured: dict,
            job_details:   dict) -> float:
    """
    Score AI [0,1] entre CV et job via le modèle fine-tuné.
    Retourne -1.0 si modèle indisponible ou inférence échouée.

    PIPELINE :
      1. build_resume_text() → texte CV format notebook + Idée 1+2
      2. build_job_text()    → texte job format notebook + Idée 1+2
      3. tokenizer()         → input_ids, attention_mask
      4. model()             → score [0,1]

    Signature identique à l'original → aucun changement dans main.py.
    """
    if model is None or tokenizer is None:
        return -1.0

    resume_text = build_resume_text(cv_structured)
    job_text    = build_job_text(job_details)

    if not resume_text.strip() or not job_text.strip():
        return -1.0

    logger.debug(f"[predict] resume: {resume_text[:120]}")
    logger.debug(f"[predict] job:    {job_text[:120]}")

    try:
        def tok(text: str) -> dict:
            return tokenizer(
                text,
                max_length     = MAX_LENGTH,
                padding        = "max_length",
                truncation     = True,
                return_tensors = "pt",
            )

        r_enc = tok(resume_text)
        j_enc = tok(job_text)

        with torch.no_grad():
            score = model(
                r_enc["input_ids"].to(DEVICE),
                r_enc["attention_mask"].to(DEVICE),
                j_enc["input_ids"].to(DEVICE),
                j_enc["attention_mask"].to(DEVICE),
            )

        value = float(score.item() if hasattr(score, "item") else score)
        logger.info(f"[predict] raw score = {value:.4f} ({value*100:.1f}%)")
        return max(0.0, min(1.0, value))

    except Exception as e:
        logger.error(f"[matcher] predict() failed: {e}")
        return -1.0