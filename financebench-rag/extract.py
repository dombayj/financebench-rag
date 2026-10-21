import json

import pymupdf
import tiktoken

from config import PAGES_CACHE_DIR

_ENC = tiktoken.get_encoding("cl100k_base")


def extract_pages(pdf_path):
    """Return per-page text, 0-indexed to match FinanceBench's evidence_page_num."""
    cache_path = PAGES_CACHE_DIR / f"{pdf_path.stem}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    with pymupdf.open(pdf_path) as pdf:
        pages = [page.get_text() for page in pdf]

    PAGES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(pages))
    return pages


def chunk_page(text, chunk_tokens, overlap_tokens):
    """Split one page's text into overlapping fixed-size token chunks."""
    tokens = _ENC.encode(text)
    if not tokens:
        return []
    if len(tokens) <= chunk_tokens:
        return [text]

    step = chunk_tokens - overlap_tokens
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunks.append(_ENC.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return chunks


def chunk_document(doc_name, pages, chunk_tokens, overlap_tokens):
    """Chunk every page of a document, tagging each chunk with its source page.

    Chunks never span a page boundary, so scoring against FinanceBench's
    page-level evidence stays exact.
    """
    chunks = []
    for page_num, text in enumerate(pages):
        if not text.strip():
            continue
        for i, chunk_text in enumerate(chunk_page(text, chunk_tokens, overlap_tokens)):
            chunks.append({
                "chunk_id": f"{doc_name}::p{page_num}::c{i}",
                "doc_name": doc_name,
                "page_num": page_num,
                "text": chunk_text,
            })
    return chunks
