"""Load the cached naive RAG index and run top-k retrieval for a query."""
import json
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config import CHUNKS_PATH, EMBEDDINGS_PATH, EMBED_MODEL, RRF_K, RRF_POOL

load_dotenv()
_client = OpenAI()


def tokenize(text):
    """Lowercase words and digit runs, split apart so "FY2022" -> ["fy", "2022"]."""
    return re.findall(r"[a-z]+|\d+", text.lower())


class Index:
    def __init__(self):
        self.chunks = [json.loads(line) for line in open(CHUNKS_PATH)]
        matrix = np.load(EMBEDDINGS_PATH)
        self.matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        self.doc_names = np.array([c["doc_name"] for c in self.chunks])
        self._bm25 = {}  # filing set -> (chunk indices, BM25 index), built on first use

    def embed_query(self, text):
        vec = np.array(
            _client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding,
            dtype=np.float32,
        )
        return vec / np.linalg.norm(vec)

    def _allowed(self, doc_names):
        if doc_names is None:
            return np.arange(len(self.chunks))
        return np.flatnonzero(np.isin(self.doc_names, list(doc_names)))

    def dense_scores(self, query, idx):
        return self.matrix[idx] @ self.embed_query(query)

    def bm25_scores(self, query, doc_names):
        """BM25 over only the allowed filings, so IDF reflects that search space."""
        key = None if doc_names is None else frozenset(doc_names)
        if key not in self._bm25:
            idx = self._allowed(doc_names)
            self._bm25[key] = (idx, BM25Okapi([tokenize(self.chunks[i]["text"]) for i in idx]))
        idx, bm25 = self._bm25[key]
        return idx, bm25.get_scores(tokenize(query))

    def search(self, query, k=10, doc_names=None, method="dense"):
        """Top-k chunks, optionally restricted to a set of filings.

        method: "dense" (cosine similarity), "bm25" (keywords), or "hybrid"
        (Reciprocal Rank Fusion of the two rankings).
        """
        if method == "bm25":
            idx, scores = self.bm25_scores(query, doc_names)
        else:
            idx = self._allowed(doc_names)
            scores = self.dense_scores(query, idx)
            if method == "hybrid":
                bm25_idx, bm25 = self.bm25_scores(query, doc_names)
                assert np.array_equal(idx, bm25_idx)
                scores = rrf(scores, bm25)
        order = np.argsort(-scores)[:k]
        return [{**self.chunks[idx[j]], "score": float(scores[j])} for j in order]


def rrf(*score_lists):
    """Reciprocal Rank Fusion: each ranking adds 1 / (RRF_K + rank) for its top RRF_POOL items."""
    fused = np.zeros(len(score_lists[0]))
    for scores in score_lists:
        top = np.argsort(-scores)[:RRF_POOL]
        fused[top] += 1.0 / (RRF_K + np.arange(1, len(top) + 1))
    return fused
