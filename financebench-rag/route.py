"""Route a question to the filing (or company) it is about, using only the question text.

The router sees the question and the catalog of filings/companies in the corpus (which a
real system knows) -- never FinanceBench's gold doc_name. Routes are cached so reruns
are free and deterministic.
"""
import json

from dotenv import load_dotenv
from openai import OpenAI

from config import (DOC_INFO_PATH, PDF_DIR, ROUTER_MODEL, ROUTES_COMPANY_PATH, ROUTES_PATH,
                    ROUTES_TOP_PATH)

load_dotenv()
_client = OpenAI()

CATALOG = sorted(p.stem for p in PDF_DIR.glob("*.pdf"))

_DOC_COMPANY = {
    d["doc_name"]: d["company"]
    for d in map(json.loads, open(DOC_INFO_PATH))
    if d["doc_name"] in CATALOG
}
COMPANIES = sorted(set(_DOC_COMPANY.values()))
NO_COMPANY = "NONE"

_FILING_PROMPT = """You pick which SEC filing a financial question should be answered from.

Filing names look like COMPANY_PERIOD_TYPE:
- COMPANY_2022_10K = annual report for fiscal year 2022
- COMPANY_2023Q2_10Q = quarterly report for Q2 2023
- COMPANY_2022_8K_dated-2022-07-01 = current report filed on that date
- COMPANY_2023Q4_EARNINGS = earnings release for that quarter

Choose the one filing most likely to contain the answer. Forecasts or guidance for a
year usually appear in the previous year's annual report.

Available filings:
{catalog}

Question: {question}"""

_TOP_PROMPT = _FILING_PROMPT.replace(
    "Choose the one filing most likely to contain the answer.",
    "Rank the three filings most likely to contain the answer, best first. "
    "The three must be different filings.",
)

_TOP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "route_top3",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {f: {"type": "string", "enum": CATALOG} for f in ("first", "second", "third")},
            "required": ["first", "second", "third"],
            "additionalProperties": False,
        },
    },
}

_COMPANY_PROMPT = """You pick which company a financial question is about.

Answer with one company from the list. If the question does not name or clearly refer
to a specific company, answer NONE. Do not guess.

Companies:
{companies}

Question: {question}"""


def _schema(name, field, options):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {field: {"type": "string", "enum": options}},
                "required": [field],
                "additionalProperties": False,
            },
        },
    }


def _load_cache(path=ROUTES_PATH):
    return json.loads(path.read_text()) if path.exists() else {}


def _ask(question, cache, path, prompt, schema, field):
    """field=None returns the whole JSON answer."""
    if question in cache:
        return cache[question]

    resp = _client.chat.completions.create(
        model=ROUTER_MODEL,
        temperature=0,
        response_format=schema,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = json.loads(resp.choices[0].message.content)
    answer = answer if field is None else answer[field]

    cache[question] = answer
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))
    return answer


def route(question, cache=None):
    """Return the catalog doc_name the question is most likely about."""
    cache = _load_cache(ROUTES_PATH) if cache is None else cache
    prompt = _FILING_PROMPT.format(catalog="\n".join(CATALOG), question=question)
    return _ask(question, cache, ROUTES_PATH, prompt,
                _schema("route", "doc_name", CATALOG), "doc_name")


def route_top(question, n, cache=None):
    """Return the router's n most likely filings (n <= 3), best first, de-duplicated."""
    cache = _load_cache(ROUTES_TOP_PATH) if cache is None else cache
    prompt = _TOP_PROMPT.format(catalog="\n".join(CATALOG), question=question)
    ranked = _ask(question, cache, ROUTES_TOP_PATH, prompt, _TOP_SCHEMA, None)
    filings = list(dict.fromkeys([ranked["first"], ranked["second"], ranked["third"]]))
    return filings[:n]


def route_company(question, cache=None):
    """Return the company the question is about, or NO_COMPANY if it names none."""
    cache = _load_cache(ROUTES_COMPANY_PATH) if cache is None else cache
    prompt = _COMPANY_PROMPT.format(companies="\n".join(COMPANIES), question=question)
    return _ask(question, cache, ROUTES_COMPANY_PATH, prompt,
                _schema("route_company", "company", COMPANIES + [NO_COMPANY]), "company")


def company_filings(company):
    """All filings of a company, or None (= search everything) for NO_COMPANY."""
    if company == NO_COMPANY:
        return None
    return {doc for doc, c in _DOC_COMPANY.items() if c == company}


def doc_company(doc_name):
    return _DOC_COMPANY[doc_name]
