"""Build the naive RAG index: extract PDFs -> chunk -> embed -> cache to disk."""
import json

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from config import (
    PDF_DIR, CACHE_DIR, CHUNKS_PATH, EMBEDDINGS_PATH,
    EMBED_MODEL, CHUNK_TOKENS, OVERLAP_TOKENS, EMBED_BATCH_SIZE,
)
from extract import extract_pages, chunk_document

load_dotenv()


def build_chunks():
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    all_chunks = []
    for pdf_path in tqdm(pdf_paths, desc="extracting + chunking"):
        pages = extract_pages(pdf_path)
        all_chunks.extend(chunk_document(pdf_path.stem, pages, CHUNK_TOKENS, OVERLAP_TOKENS))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"{len(pdf_paths)} docs -> {len(all_chunks)} chunks -> {CHUNKS_PATH}")
    return all_chunks


def embed_chunks(chunks):
    client = OpenAI()
    vectors = []
    for i in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc="embedding"):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=[c["text"] for c in batch])
        vectors.extend(d.embedding for d in resp.data)

    matrix = np.array(vectors, dtype=np.float32)
    np.save(EMBEDDINGS_PATH, matrix)
    print(f"embedded {len(chunks)} chunks -> {EMBEDDINGS_PATH} shape={matrix.shape}")


def main():
    chunks = build_chunks()
    embed_chunks(chunks)


if __name__ == "__main__":
    main()
