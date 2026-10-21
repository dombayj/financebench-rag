"""Rerank retrieved chunks with a cross-encoder: local Qwen3-Reranker or the Cohere Rerank API.

The reranker reads (question, chunk) pairs together and outputs P(relevant) in [0, 1];
it never generates text. Scores are cached per (question, hash of chunk text), so re-chunking
never reuses a stale score. Each backend has its own cache file: the Qwen file name includes
the model and a hash of the instruction + prompt, so changing any of them rescores from scratch.
"""
import hashlib
import json
import os
import time

from config import (CACHE_DIR, COHERE_RERANK_MODEL, RERANK_BATCH_SIZE, RERANK_INSTRUCTION,
                    RERANK_MODEL)

# Prompt format from the Qwen3-Reranker model card.
_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
           "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
           "\"no\".<|im_end|>\n<|im_start|>user\n")
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_MAX_LENGTH = 8192


def _cache_path(backend):
    if backend == "cohere":
        return CACHE_DIR / f"rerank_cohere_{COHERE_RERANK_MODEL}.json"
    tag = hashlib.sha1((RERANK_INSTRUCTION + _PREFIX + _SUFFIX).encode()).hexdigest()[:8]
    return CACHE_DIR / f"rerank_{RERANK_MODEL.split('/')[-1]}_{tag}.json"


def _key(question, text):
    return f"{question}||{hashlib.sha1(text.encode()).hexdigest()[:12]}"


class Reranker:
    def __init__(self, backend="qwen", header=False):
        self.backend = backend
        self.header = header  # prepend the chunk's filing name, e.g. "Filing: 3M 2018 10K"
        self.model_name = COHERE_RERANK_MODEL if backend == "cohere" else RERANK_MODEL
        self.cache_path = _cache_path(backend)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.search_units = 0  # Cohere billing units used by this run (cached scores cost 0)
        self._model = None
        self._client = None

    def _load_qwen(self):
        import torch  # only the local backend needs torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(
            RERANK_MODEL, dtype=torch.bfloat16).to(device).eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(_PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(_SUFFIX, add_special_tokens=False)

    def _score_qwen(self, question, texts):
        import torch

        if self._model is None:
            self._load_qwen()
        pairs = [f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {question}\n<Document>: {t}"
                 for t in texts]
        enc = self.tokenizer(pairs, padding=False, truncation="longest_first",
                             return_attention_mask=False,
                             max_length=_MAX_LENGTH - len(self.prefix_ids) - len(self.suffix_ids))
        enc["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in enc["input_ids"]]
        enc = self.tokenizer.pad(enc, padding=True, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            logits = self._model(**enc).logits[:, -1, :]
        yes_no = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
        return torch.log_softmax(yes_no, dim=1)[:, 1].exp().tolist()

    def _score_cohere(self, question, texts):
        import cohere
        from cohere.errors import TooManyRequestsError

        if self._client is None:
            self._client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        for attempt in range(6):
            try:
                resp = self._client.rerank(model=COHERE_RERANK_MODEL, query=question,
                                           documents=texts)
                break
            except TooManyRequestsError:  # trial keys are rate limited per minute
                if attempt == 5:
                    raise
                time.sleep(15)
        billed = resp.meta.billed_units if resp.meta else None
        self.search_units += int(billed.search_units or 0) if billed else 0
        scores = [0.0] * len(texts)
        for r in resp.results:
            scores[r.index] = r.relevance_score
        return scores

    def _doc_text(self, chunk):
        if not self.header:
            return chunk["text"]
        return f"Filing: {chunk['doc_name'].replace('_', ' ')}\n\n{chunk['text']}"

    def rerank(self, question, chunks):
        """Return chunks sorted by reranker score (best first), each with "rerank_score"."""
        # Cache keys hash the exact text the model sees, so header/no-header never mix.
        todo = [c for c in chunks if _key(question, self._doc_text(c)) not in self.cache]
        # Cohere scores a whole question's pool in one billed search; Qwen goes in GPU batches.
        step = max(len(todo), 1) if self.backend == "cohere" else RERANK_BATCH_SIZE
        score = self._score_cohere if self.backend == "cohere" else self._score_qwen
        for i in range(0, len(todo), step):
            batch = todo[i:i + step]
            texts = [self._doc_text(c) for c in batch]
            for t, s in zip(texts, score(question, texts)):
                self.cache[_key(question, t)] = s
        if todo:
            # Write-then-rename so a Ctrl-C mid-save can't leave broken JSON behind.
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache))
            os.replace(tmp, self.cache_path)

        scored = [{**c, "rerank_score": self.cache[_key(question, self._doc_text(c))]}
                  for c in chunks]
        return sorted(scored, key=lambda c: -c["rerank_score"])
