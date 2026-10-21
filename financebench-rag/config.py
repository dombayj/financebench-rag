from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "pdfs"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"

QUESTIONS_PATH = DATA_DIR / "financebench_open_source.jsonl"

PAGES_CACHE_DIR = CACHE_DIR / "pages"
CHUNKS_PATH = CACHE_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"

EMBED_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 60
EMBED_BATCH_SIZE = 100

TOP_KS = [1, 3, 5, 10]

DOC_INFO_PATH = DATA_DIR / "financebench_document_information.jsonl"
ROUTES_PATH = CACHE_DIR / "routes.json"
ROUTES_COMPANY_PATH = CACHE_DIR / "routes_company.json"
ROUTES_TOP_PATH = CACHE_DIR / "routes_top3.json"
ROUTER_MODEL = "gpt-4o-mini"

RRF_K = 60  # standard Reciprocal Rank Fusion constant
RRF_POOL = 100  # how deep each ranking contributes to the fusion

RERANK_MODEL = "Qwen/Qwen3-Reranker-4B"
RERANK_INSTRUCTION = ("Given a question about a company's SEC filing, judge whether the passage "
                      "contains information needed to answer the question")
RERANK_BATCH_SIZE = 4
COHERE_RERANK_MODEL = "rerank-v4.0-pro"  # `--rerank cohere`; needs COHERE_API_KEY in .env
