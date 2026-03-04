"""
extraction.py — aijobs.ai
Structure réelle confirmée par debug HTML complet :

  [Title]
  [Navbar garbage]
  Job Details
  [Title again]
  [Company]          ← juste avant "Full Time"
  Full Time          ← job_type badge
  Apply Now
  [Description...]   ← tout ce bloc
  Location
  [City, Country]    ← valeur location
  [Tag]              ← ex: "Engineer", "Finance"
  Job Overview
  Job Posted:
  [X days ago]
  Job Expires:
  [date or empty]
  Job Type
  Full Time
  Share This Job:
  Related Jobs       ← STOP tout extraction ici
  [autres jobs avec leurs propres salaires — NE PAS extraire]
"""

import re
import time
import asyncio
import aiohttp
from bs4 import BeautifulSoup

EXTRACT_TIMEOUT = 20

SKILLS_REQUIRED_KW = [
    'Python', 'SQL', 'Spark', 'Kafka', 'Airflow', 'dbt', 'AWS', 'GCP',
    'Azure', 'Docker', 'Kubernetes', 'Terraform', 'Scala', 'Java',
    'Hadoop', 'BigQuery', 'Redshift', 'Snowflake', 'Databricks',
    'pandas', 'NumPy', 'TensorFlow', 'PyTorch', 'scikit-learn',
    'React', 'Node.js', 'PostgreSQL', 'MongoDB', 'Redis', 'FastAPI',
    'JavaScript', 'TypeScript', 'Go', 'Rust', 'R', 'Tableau',
    'Power BI', 'Looker', 'Fivetran', 'Airbyte', 'Flink', 'Hive',
    'Linux', 'Git', 'CI/CD', 'GraphQL', 'REST', 'gRPC',
    'Elasticsearch', 'Cassandra', 'MySQL', 'Oracle', 'MLflow',
    'DVC', 'Streamlit', 'Flask', 'Django', 'Spring', 'Kotlin',
    'Swift', 'C++', 'C#', '.NET', 'PHP', 'Ruby', 'Bash', 'MATLAB',
    'Next.js', 'Tailwind', 'Figma', 'Playwright', 'Jest', 'Cypress',
]

SKILLS_BONUS_KW = [
    'Tableau', 'Power BI', 'Looker', 'Metabase', 'Grafana',
    'Ansible', 'Jenkins', 'GitHub Actions', 'CircleCI', 'ArgoCD',
    'Prometheus', 'Datadog', 'Splunk', 'New Relic',
    'Delta Lake', 'Iceberg', 'LangChain', 'OpenAI', 'Hugging Face',
    'LLM', 'RAG', 'Vertex AI', 'SageMaker', 'Azure ML',
]

EXPERIENCE_PATTERNS = [
    (r'\b(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)\b',  '{n}+ years exp.'),
    (r'\b(\d+)\s*[-–]\s*(\d+)\s*(?:years?|yrs?)\b',                     '{n1}-{n2} years'),
    (r'\bjunior\b',                                                        'Junior (0-2 yrs)'),
    (r'\bmid[- ]?level\b',                                                 'Mid-level (2-5 yrs)'),
    (r'\bsenior\b|\bsr\.\b',                                               'Senior (5+ yrs)'),
    (r'\blead\b|\bprincipal\b|\bstaff\b',                                  'Lead / Principal'),
    (r'\bentry[- ]?level\b|\bgraduate\b',                                  'Entry level'),
    (r'\bintern\b|\binternship\b',                                         'Intern'),
]

EDUCATION_PATTERNS = [
    (r"\b(?:master'?s?|msc|m\.s\.?)\b",                                   "Master's degree"),
    (r"\b(?:bachelor'?s?|bsc|b\.s\.?)\b",                                 "Bachelor's degree"),
    (r'\b(?:phd|ph\.d\.?|doctorate)\b',                                    'PhD / Doctorate'),
    (r"\b(?:engineer'?s? degree|engineering degree)\b",                    "Engineering degree"),
    (r'\bself[- ]?taught\b|\bno degree required\b|\bbootcamp\b',           'Self-taught OK'),
]

CONTRACT_PATTERNS = [
    (r'\bfull[- ]?time\b',             'Full-time'),
    (r'\bpart[- ]?time\b',             'Part-time'),
    (r'\bcontract\b',                  'Contract'),
    (r'\bfreelance\b',                 'Freelance'),
    (r'\binternship\b|\bintern\b',     'Internship'),
]

REMOTE_PATTERNS = [
    (r'\bfully[- ]?remote\b|\b100\s*%\s*remote\b',  'Full Remote 🌍'),
    (r'\bremote[- ]?first\b',                         'Remote First 🌍'),
    (r'\bhybrid\b',                                   'Hybrid 🏠🏢'),
    (r'\bon[- ]?site\b|\bin[- ]?office\b',           'On-site 🏢'),
    (r'\bremote\b',                                   'Remote 🌍'),
]


def _scan(text, patterns):
    for pat, label in patterns:
        m = re.search(pat, text, re.I)
        if m:
            if '{n}' in label:
                label = label.replace('{n}', m.group(1))
            elif '{n1}' in label:
                label = label.replace('{n1}', m.group(1)).replace('{n2}', m.group(2))
            return label
    return ''


def _get_main_block(lines):
    """
    Retourne uniquement les lignes AVANT 'Related Jobs'.
    Cela évite de capturer salary/location des autres jobs en bas de page.
    """
    cutoff = len(lines)
    for i, l in enumerate(lines):
        if re.match(r'^related jobs?$', l, re.I):
            cutoff = i
            break
    return lines[:cutoff]


def _extract_salary(main_lines):
    """
    Cherche 'Salary:' suivi de sa valeur dans les lignes principales.
    Structure réelle :  'Salary:'  puis  '\n240,000'  (lignes séparées)
    """
    text = '\n'.join(main_lines)

    # Pattern aijobs : "Salary:" sur une ligne, valeur sur la suivante
    for i, line in enumerate(main_lines):
        if re.match(r'^salary\s*:?$', line, re.I):
            for j in range(i+1, min(i+4, len(main_lines))):
                val = main_lines[j].strip().lstrip('$').replace(',','').replace('\xa0','').strip()
                if val and re.match(r'^\d+', val):
                    try:
                        n = int(float(val))
                        if n > 500:
                            return f"${n:,} / yr"
                    except Exception:
                        pass

    # Pattern inline "Salary: $90k - $120k" ou "Salary: 240,000"
    m = re.search(r'[Ss]alary\s*:\s*\$?\s*([\d,kK]+)\s*[-–]?\s*\$?\s*([\d,kK]*)', text)
    if m:
        lo = m.group(1).strip()
        hi = m.group(2).strip()
        def parse_val(v):
            v = v.replace(',','').replace('\xa0','')
            if v.lower().endswith('k'):
                return int(float(v[:-1])) * 1000
            try: return int(float(v))
            except: return 0
        lo_n = parse_val(lo)
        hi_n = parse_val(hi) if hi else 0
        if hi_n > 500:
            return f"${lo_n:,} – ${hi_n:,} / yr"
        elif lo_n > 500:
            return f"${lo_n:,} / yr"

    # Patterns libres dans le texte principal
    m = re.search(r'\$\s*(\d+)\s*[kK]\s*[-–]\s*\$?\s*(\d+)\s*[kK]', text)
    if m:
        return f"${int(m.group(1))*1000:,} – ${int(m.group(2))*1000:,} / yr"
    m = re.search(r'\$\s*([\d,]{4,})\s*[-–]\s*\$?\s*([\d,]{4,})', text)
    if m:
        return f"${m.group(1)} – ${m.group(2)} / yr"
    m = re.search(r'([\d,]{4,})\s*[-–]\s*([\d,]{4,})\s*(USD|EUR|GBP|CAD)', text)
    if m:
        return f"{m.group(1)} – {m.group(2)} {m.group(3)} / yr"
    m = re.search(r'\$\s*(\d+)\s*[kK]\+?', text)
    if m:
        return f"${int(m.group(1))*1000:,}+ / yr"
    m = re.search(r'(?:compensation|pay)[:\s]+\$?\s*([\d,kK]+)', text, re.I)
    if m:
        return m.group(1).strip()

    return 'Not specified'


def _extract_company(main_lines, title):
    """
    Structure confirmée :
      [Title] [navbar...] 'Job Details' [Title again] [Company] 'Full Time' 'Apply Now'
    Company = ligne juste AVANT 'Full Time' (première occurrence après le 2e titre)
    """
    SKIP = {'post a job','home','jobs','companies','pricing','blog',
            'sign in','post job','job details','apply now','full time',
            'part time','contract','freelance','internship'}

    title_lower = title.lower()
    title_count = 0
    for i, line in enumerate(main_lines):
        if line.lower() == title_lower:
            title_count += 1
            if title_count == 2:
                for j in range(i+1, min(i+10, len(main_lines))):
                    candidate = main_lines[j].strip()
                    if (candidate
                            and candidate.lower() not in SKIP
                            and len(candidate) > 1
                            and len(candidate) < 80):
                        return candidate
    # Fallback : ligne juste avant 'Full Time' après le titre
    for i, line in enumerate(main_lines):
        if re.match(r'^full[- ]?time$', line, re.I) and i > 0:
            candidate = main_lines[i-1].strip()
            if (candidate
                    and candidate.lower() not in SKIP
                    and candidate.lower() != title_lower
                    and len(candidate) > 1):
                return candidate
    return ''


def _extract_location(main_lines):
    """
    Structure confirmée :
      'Location'
      'São Paulo,  Brazil'
    """
    for i, line in enumerate(main_lines):
        if re.match(r'^location$', line.strip(), re.I):
            for j in range(i+1, min(i+4, len(main_lines))):
                val = main_lines[j].strip()
                if val and len(val) > 2 and not re.match(r'^(location|job overview|engineer|finance)$', val, re.I):
                    return val[:100]
    return ''


def _extract_tags(main_lines):
    """
    Structure confirmée :
      [location_value]
      'Engineer'          ← tag
      'Job Overview'
    Les tags se trouvent entre la valeur de location et 'Job Overview'.
    """
    tags = []
    in_zone = False
    for line in main_lines:
        l = line.strip()
        if re.match(r'^job overview$', l, re.I):
            break
        if in_zone:
            if (l and len(l) < 35
                    and not re.match(r'^(job posted|job expires|full time|part time|apply now|share this job)$', l, re.I)
                    and not re.match(r'^\d', l)):
                tags.append(l)
        if re.match(r'^location$', l, re.I):
            in_zone = True
    return tags[:6]


def _extract_description(main_lines):
    """
    Description = tout entre 'Apply Now' et 'Location' label.
    """
    desc_lines = []
    in_desc = False
    for line in main_lines:
        l = line.strip()
        if re.match(r'^apply now$', l, re.I):
            in_desc = True
            continue
        if re.match(r'^location$', l, re.I):
            break
        if in_desc and l:
            desc_lines.append(l)
    desc = ' '.join(desc_lines)
    desc = desc.replace('\xa0', ' ')
    desc = re.sub(r'\s{2,}', ' ', desc)
    return desc.strip()


def _extract_posted_expired(main_lines):
    posted  = ''
    expires = ''

    NOT_A_DATE = {
        'job type', 'full time', 'part time', 'contract',
        'freelance', 'internship', 'share this job', 'related jobs',
        'job overview', 'apply now', 'full-time', 'part-time',
    }

    for i, line in enumerate(main_lines):
        if re.match(r'^job posted\s*:?$', line.strip(), re.I):
            for j in range(i+1, min(i+3, len(main_lines))):
                val = main_lines[j].strip()
                if val and val.lower() not in NOT_A_DATE:
                    posted = val[:40]
                    break

        if re.match(r'^job expires\s*:?$', line.strip(), re.I):
            for j in range(i+1, min(i+3, len(main_lines))):
                val = main_lines[j].strip()
                if val:
                    if val.lower() in NOT_A_DATE:
                        expires = ''
                    else:
                        expires = val[:40]
                    break

    return posted, expires


def _required_vs_bonus(text):
    bonus_m = re.search(
        r'(?:nice[- ]to[- ]have|bonus|plus|preferred|would be (?:a )?plus)'
        r'(.{0,1200})', text, re.I | re.S)
    bonus_text = bonus_m.group(1) if bonus_m else ''

    req_m = re.search(
        r"(?:requirements?|required|must[- ]have|qualifications?|"
        r"what you.ll need|you have|you bring|things we.re looking for|we.re looking for)"
        r'(.{0,1200})', text, re.I | re.S)
    req_text = req_m.group(1) if req_m else ''

    all_kw   = list(dict.fromkeys(SKILLS_REQUIRED_KW + SKILLS_BONUS_KW))
    required, bonus = [], []

    for sk in all_kw:
        pat = r'\b' + re.escape(sk) + r'\b'
        if not re.search(pat, text, re.I):
            continue
        in_bonus = bool(bonus_text and re.search(pat, bonus_text, re.I))
        in_req   = bool(req_text   and re.search(pat, req_text,   re.I))
        if in_bonus and not in_req:
            bonus.append(sk)
        else:
            required.append(sk)

    return required, bonus


def parse_job_detail(html, url):
    soup  = BeautifulSoup(html, 'html.parser')
    full_text = soup.get_text(separator='\n', strip=True)

    all_lines  = [l.strip() for l in full_text.split('\n') if l.strip()]
    main_lines = _get_main_block(all_lines)
    main_text  = '\n'.join(main_lines)

    # Title
    title = ''
    h1 = soup.find('h1')
    if h1:
        title = h1.get_text(strip=True)
    if not title and main_lines:
        title = main_lines[0]

    # Company
    company = _extract_company(main_lines, title)

    # Job type
    job_type = ''
    m = re.search(r'\b(full[- ]?time|part[- ]?time|contract|freelance|internship)\b',
                  main_text, re.I)
    if m:
        job_type = m.group(0)

    # Location
    location = _extract_location(main_lines)

    # Tags
    page_tags = _extract_tags(main_lines)

    # Description — complète, sans troncature
    description = _extract_description(main_lines)

    # Posted / Expired
    posted, expires = _extract_posted_expired(main_lines)
    expired_label = 'Yes ⚠️' if (expires and len(expires) > 2) else 'No ✅'

    # Salary
    salary = _extract_salary(main_lines)

    # Remote
    remote = _scan(main_text, REMOTE_PATTERNS) or 'Not specified'

    # Contract
    contract_type = _scan(main_text, CONTRACT_PATTERNS) or job_type or 'Full-time'

    # Experience / Education
    experience = _scan(main_text, EXPERIENCE_PATTERNS) or 'Not specified'
    education  = _scan(main_text, EDUCATION_PATTERNS)  or 'Not specified'

    # Skills
    skills_required, skills_bonus = _required_vs_bonus(main_text)

    return {
        'title'           : title,
        'company'         : company,
        'location'        : location,
        'job_type'        : job_type,
        'salary'          : salary,
        'remote'          : remote,
        'contract_type'   : contract_type,
        'experience'      : experience,
        'education'       : education,
        'expired'         : expired_label,
        'posted'          : posted,
        'skills_required' : ', '.join(skills_required),
        'skills_bonus'    : ', '.join(skills_bonus),
        'tags'            : ', '.join(page_tags),
        'description'     : description,
        'url'             : url,
    }


async def extract_details(urls, session):
    if not urls:
        return None
    url = urls[0]
    _t  = time.perf_counter()

    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=EXTRACT_TIMEOUT),
            allow_redirects=True,
        ) as resp:
            if resp.status == 429:
                print(f'  [429 skip] {url}')
                return None
            if resp.status == 404:
                print(f'  [404 expired] {url}')
                return None
            resp.raise_for_status()
            html    = await resp.text()
            details = parse_job_detail(html, url)
            elapsed = (time.perf_counter() - _t) * 1000

    except asyncio.TimeoutError:
        print(f'  [TIMEOUT] {url}')
        return None
    except Exception as e:
        print(f'  [ERROR] {str(e)[:60]}')
        return None

    print('  ' + '-' * 60)
    print(f'  {elapsed:.0f}ms  |  {details["title"]}')
    if details['company']:        print(f'  Company  : {details["company"]}')
    if details['location']:       print(f'  Location : {details["location"]}')
    if details['salary'] != 'Not specified':
                                  print(f'  Salary   : {details["salary"]}')
    if details['contract_type']:  print(f'  Contract : {details["contract_type"]}')
    if details['experience'] != 'Not specified':
                                  print(f'  Exp      : {details["experience"]}')
    if details['tags']:           print(f'  Tags     : {details["tags"]}')
    if details['skills_required']:print(f'  Skills   : {details["skills_required"][:80]}')
    print('  ' + '-' * 60)
    print()

    return details