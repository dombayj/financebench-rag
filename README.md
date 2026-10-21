# FinanceBench RAG: retrieval experiments

Page-level retrieval experiments on [FinanceBench](https://github.com/patronus-ai/financebench)
(150 open-source questions over 84 SEC filings: 10-Ks, 10-Qs, 8-Ks and earnings releases).
The goal is to measure how well different retrieval setups find the **exact page** that holds
the evidence for each question, starting from a naive RAG baseline and improving it one step at
a time.

All numbers are strict page-level metrics. The gold filing name is **never** used at retrieval
time: the system has to find the right filing on its own.

## Baseline

Every improvement below is measured against the same plain dense-RAG baseline:

- **Chunking:** fixed 400-token chunks with 60-token (15%) overlap; chunks never cross a page
  boundary.
- **Embedding model:** OpenAI `text-embedding-3-small`.
- **Search scope:** one index of 32,469 chunks across all 84 filings, with no metadata filtering.
- **Retrieval:** cosine similarity, top-50 chunks merged into a ranked list of unique pages, with
  no reranking, no BM25 and no query rewriting.

The improved setups never use the gold filing name either: filing selection is done by an LLM
router that sees only the question and the list of filing names.

## Results

| Setup | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| Naive RAG (dense, all 84 filings) | – | 0.373 | 0.513 | 0.248 |
| + LLM filing routing (1 filing) | – | 0.460 | 0.580 | 0.329 |
| + Cohere rerank of top-50 | – | 0.727 | 0.780 | 0.603 |
| **Top-3 filing routing + Cohere rerank of top-100 + filing-name header** | **0.520** | **0.820** | **0.887** | **0.647** |

What didn't help:
- **BM25 / hybrid (RRF)** on routed filings did worse than dense retrieval (hit@10 0.24 / 0.35 vs
  0.58). FinanceBench's boilerplate question text gets high IDF, and question wording often
  differs from filing wording ("capital expenditure" vs "purchases of PP&E").
- **Company-level routing** reached 0.98 company accuracy, but it confuses fiscal years within
  the same company.
- **Top-N routing without a reranker**: near-identical pages from other years push the right
  page down.

The full run log is in [`financebench-rag/results/history.md`](financebench-rag/results/history.md),
and each run's per-question output is in `financebench-rag/results/*.json`.

## Pipeline

1. **Extract**: PyMuPDF pulls per-page text (`extract.py`).
2. **Chunk**: 400-token chunks with 60-token overlap that never cross a page boundary, so
   page-level scoring stays exact.
3. **Embed**: OpenAI `text-embedding-3-small` (`build_index.py`).
4. **Route**: `gpt-4o-mini` picks the most likely filing(s) from the question and the filing
   catalog (`route.py`).
5. **Retrieve**: dense, BM25 or hybrid search within the routed filings (`retrieve.py`).
6. **Rerank** (optional): Cohere `rerank-v4.0-pro` or local `Qwen3-Reranker-4B` (`rerank.py`).
7. **Evaluate**: hit@k, recall@k, MRR and nDCG at page level (`evaluate.py`), plus error
   analysis (`analyze.py`).

## Setup

```bash
cd financebench-rag
pip install -r requirements.txt
```

Create a `.env` file in `financebench-rag/` or any parent folder:

```
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...            # only needed for --rerank cohere
```

### Data (not included in this repo)

The PDFs and the dataset files are not checked in. Get them from the
[FinanceBench repo](https://github.com/patronus-ai/financebench):

- `pdfs/*.pdf` → `financebench-rag/pdfs/`
- `data/financebench_open_source.jsonl` and `data/financebench_document_information.jsonl`
  → `financebench-rag/data/`

The `data/cache/` folder (page text, chunks, embeddings, router and reranker caches) is rebuilt
automatically.

## Usage

```bash
cd financebench-rag
python main.py build                                   # extract, chunk and embed all PDFs
python main.py eval                                    # naive RAG baseline
python main.py eval-routed                             # 1-filing routing
python main.py eval-routed --search hybrid             # dense + BM25 with RRF
python main.py eval-routed --rerank cohere             # + Cohere rerank
python main.py eval-routed-top3 --rerank cohere --rerank-header --pool 100   # best setup
python analyze.py                                      # error analysis of the best setup
```

Each eval run writes a timestamped JSON to `results/` and appends a row to `results/history.md`.
