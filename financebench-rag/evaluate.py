"""Score naive RAG retrieval against FinanceBench's gold evidence pages."""
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone

from config import (COHERE_RERANK_MODEL, QUESTIONS_PATH, RESULTS_DIR, ROUTER_MODEL,
                    ROUTES_COMPANY_PATH, RERANK_MODEL, ROUTES_PATH, ROUTES_TOP_PATH, RRF_K, TOP_KS)
from retrieve import Index
from route import _load_cache, company_filings, doc_company, route, route_company, route_top

POOL_K = 50  # retrieve a deeper pool so MRR isn't truncated by the largest TOP_KS


def dedupe_pages(chunks):
    """Collapse retrieved chunks to a ranked, deduped list of (doc_name, page_num)."""
    seen = set()
    pages = []
    for c in chunks:
        key = (c["doc_name"], c["page_num"])
        if key not in seen:
            seen.add(key)
            pages.append(key)
    return pages


def score_question(ranked_pages, gold_pages):
    metrics = {}
    for k in TOP_KS:
        topk = ranked_pages[:k]
        hits = gold_pages & set(topk)
        dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(topk) if p in gold_pages)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold_pages), k)))
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
        metrics[f"recall@{k}"] = len(hits) / len(gold_pages)
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    rr = 0.0
    for i, p in enumerate(ranked_pages):
        if p in gold_pages:
            rr = 1 / (i + 1)
            break
    metrics["mrr"] = rr
    return metrics


def run(mode="naive", method="dense", rerank=None, pool=POOL_K, header=False):
    """mode: "naive" (all filings), "filing" (router picks one filing),
    "company" (router picks a company; all its filings, or everything if NONE),
    "top2"/"top3" (router ranks filings; search its 2 or 3 most likely together).
    rerank: None, "qwen" or "cohere"; header: show the reranker each chunk's filing name."""
    index = Index()
    if rerank:
        from rerank import Reranker  # only needed when reranking
    reranker = Reranker(rerank, header) if rerank else None
    questions = [json.loads(line) for line in open(QUESTIONS_PATH)]
    route_cache = _load_cache(ROUTES_PATH)
    company_cache = _load_cache(ROUTES_COMPANY_PATH)
    top_cache = _load_cache(ROUTES_TOP_PATH)

    per_question = []
    for q in questions:
        gold_pages = {(e["doc_name"], e["evidence_page_num"]) for e in q["evidence"]}
        # The router sees only the question text; gold doc_name is used just to score routing.
        doc_names = None
        if mode == "filing":
            doc_name = route(q["question"], route_cache)
            doc_names = {doc_name}
        elif mode == "company":
            company = route_company(q["question"], company_cache)
            doc_names = company_filings(company)
        elif mode in ("top2", "top3"):
            doc_names = set(route_top(q["question"], int(mode[-1]), top_cache))
        chunks = index.search(q["question"], k=pool, doc_names=doc_names, method=method)
        if reranker:
            chunks = reranker.rerank(q["question"], chunks)
        ranked_pages = dedupe_pages(chunks)
        m = score_question(ranked_pages, gold_pages)
        if mode == "filing":
            m["route_correct"] = 1.0 if doc_name == q["doc_name"] else 0.0
        elif mode == "company":
            m["company_correct"] = 1.0 if company == doc_company(q["doc_name"]) else 0.0
        elif mode in ("top2", "top3"):
            m["route_correct"] = 1.0 if q["doc_name"] in doc_names else 0.0
        m["financebench_id"] = q["financebench_id"]
        m["question_type"] = q["question_type"]
        per_question.append(m)
    if reranker and reranker.backend == "cohere":
        print(f"Cohere search units billed this run: {reranker.search_units}")
    return per_question


def aggregate(per_question, group_key=None):
    groups = defaultdict(list)
    for m in per_question:
        groups[m[group_key] if group_key else "all"].append(m)

    metric_names = [k for k in per_question[0] if k not in ("financebench_id", "question_type")]
    summary = {}
    for key, rows in groups.items():
        summary[key] = {name: round(statistics.mean(r[name] for r in rows), 4) for name in metric_names}
        summary[key]["n"] = len(rows)
    return summary


def print_table(summary, title):
    print(f"\n== {title} ==")
    for key, m in summary.items():
        line = ", ".join(f"{k}={v}" for k, v in m.items() if k != "n")
        print(f"{key} (n={m['n']}): {line}")


def main(mode="naive", method="dense", rerank=None, pool=POOL_K, header=False):
    per_question = run(mode, method, rerank, pool, header)
    overall = aggregate(per_question)
    by_type = aggregate(per_question, group_key="question_type")

    print_table(overall, "Overall")
    print_table(by_type, "By question_type")

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = {"naive": "baseline", "filing": "routed", "company": "routed_company",
            "top2": "routed_top2", "top3": "routed_top3"}[mode]
    if method != "dense":
        name += f"_{method}"
    if rerank == "qwen":
        name += "_rerank"
    elif rerank:
        name += f"_rerank_{rerank}"
    if header:
        name += "_header"
    if pool != POOL_K:
        name += f"_pool{pool}"
    out_path = RESULTS_DIR / f"{name}_{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp": ts,
        "overall": overall,
        "by_question_type": by_type,
        "per_question": per_question,
    }, indent=2))
    print(f"\nsaved -> {out_path}")

    history_path = RESULTS_DIR / "history.md"
    header_needed = not history_path.exists()
    with open(history_path, "a") as f:
        if header_needed:
            f.write("| timestamp | n | hit@5 | recall@5 | mrr | ndcg@5 | notes |\n")
            f.write("|---|---|---|---|---|---|---|\n")
        o = overall["all"]
        notes = {
            "naive": "naive RAG baseline (400tok/15% overlap, text-embedding-3-small)",
            "filing": f"filing routing ({ROUTER_MODEL}, route acc {o.get('route_correct')}) + naive RAG",
            "top2": f"top-2 filing routing ({ROUTER_MODEL}, gold filing in top-2 {o.get('route_correct')}) + naive RAG",
            "top3": f"top-3 filing routing ({ROUTER_MODEL}, gold filing in top-3 {o.get('route_correct')}) + naive RAG",
            "company": f"company routing + NONE fallback ({ROUTER_MODEL}, "
                       f"company acc {o.get('company_correct')}) + naive RAG",
        }[mode]
        if method != "dense":
            search = {"bm25": "BM25 only", "hybrid": f"hybrid dense+BM25 (RRF k={RRF_K})"}[method]
            notes = notes.replace("naive RAG", search)
        if rerank:
            model = COHERE_RERANK_MODEL if rerank == "cohere" else RERANK_MODEL.split('/')[-1]
            notes += f" + rerank top-{pool} ({model}{', filing-name header' if header else ''})"
        f.write(
            f"| {ts} | {o['n']} | {o['hit@5']} | {o['recall@5']} | {o['mrr']} | {o['ndcg@5']} "
            f"| {notes} |\n"
        )
    print(f"appended -> {history_path}")


if __name__ == "__main__":
    main()
