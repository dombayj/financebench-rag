"""Error analysis for the top-3 routing + rerank setup (config B).

Rebuilds each question's ranking from the cached routes and rerank scores (never calls the
reranker API; cache misses are reported), then looks at two failure groups:
  1. questions whose gold filing is outside the router's top 3;
  2. questions that hit@10 but miss hit@1 -- what took rank 1 instead?
"""
import argparse
import json
import re
from collections import Counter

from config import QUESTIONS_PATH, RESULTS_DIR
from evaluate import dedupe_pages
from rerank import Reranker, _key
from retrieve import Index
from route import _load_cache, route_top
from config import ROUTES_TOP_PATH


def answer_numbers(answer):
    """Number strings from the gold answer, in the spellings a filing might use."""
    variants = set()
    for raw in re.findall(r"\d[\d,]*\.?\d*", answer):
        num = raw.replace(",", "").rstrip(".")
        if len(num.replace(".", "")) < 3:  # "2", "15" match almost any page
            continue
        whole, _, frac = num.partition(".")
        frac = frac.rstrip("0")
        plain = whole + ("." + frac if frac else "")
        variants |= {plain, f"{int(whole):,}" + ("." + frac if frac else "")}
    return variants


def main(header=True, pool=100):
    index = Index()
    reranker = Reranker("cohere", header)
    pages_text = {}
    for c in index.chunks:
        pages_text.setdefault((c["doc_name"], c["page_num"]), []).append(c["text"])
    questions = [json.loads(line) for line in open(QUESTIONS_PATH)]
    top_cache = _load_cache(ROUTES_TOP_PATH)

    rows, misses = [], 0
    for q in questions:
        gold = {(e["doc_name"], e["evidence_page_num"]) for e in q["evidence"]}
        filings = route_top(q["question"], 3, top_cache)
        chunks = index.search(q["question"], k=pool, doc_names=set(filings))
        keys = [_key(q["question"], reranker._doc_text(c)) for c in chunks]
        missing = sum(k not in reranker.cache for k in keys)
        misses += missing
        scored = sorted(({**c, "rerank_score": reranker.cache.get(k, 0.0)}
                         for c, k in zip(chunks, keys)), key=lambda c: -c["rerank_score"])
        page_score = {}
        for c in scored:
            page_score.setdefault((c["doc_name"], c["page_num"]), c["rerank_score"])
        ranked = dedupe_pages(scored)
        gold_ranks = [i + 1 for i, p in enumerate(ranked) if p in gold]
        rows.append({"q": q, "gold": gold, "filings": filings, "ranked": ranked,
                     "page_score": page_score, "best_gold": gold_ranks[0] if gold_ranks else None,
                     "gold_in_pool": any(p in gold for p in ranked), "missing": missing})

    print(f"rerank cache misses: {misses} pairs (scored as 0; should be 0)\n")

    # 1. Gold filing outside the router's top 3.
    out = [r for r in rows if r["q"]["doc_name"] not in r["filings"]]
    print(f"== 1. gold filing outside top-3: {len(out)} questions ==")
    for r in out:
        q = r["q"]
        print(f"- [{q['question_type']}] {q['question'][:160]}")
        print(f"    gold: {q['doc_name']}   router: {', '.join(r['filings'])}")

    # 2. hit@10 but not hit@1.
    routed = [r for r in rows if r["q"]["doc_name"] in r["filings"]]
    hit1 = [r for r in routed if r["best_gold"] == 1]
    near = [r for r in routed if r["best_gold"] and 1 < r["best_gold"] <= 10]
    far = [r for r in routed if r["best_gold"] and r["best_gold"] > 10]
    lost = [r for r in routed if not r["gold_in_pool"]]
    print(f"\n== 2. correctly routed: {len(routed)} ==")
    print(f"gold page rank 1: {len(hit1)}, rank 2-10: {len(near)}, rank >10: {len(far)}, "
          f"not in dense pool-{pool}: {len(lost)}")
    print("hit@1 by question_type:", {t: f"{sum(r['best_gold'] == 1 for r in routed if r['q']['question_type'] == t)}"
                                       f"/{sum(r['q']['question_type'] == t for r in routed)}"
                                       for t in sorted({r['q']['question_type'] for r in routed})})

    kinds = Counter()
    print(f"\n-- rank 2-10 cases: what is at rank 1? --")
    for r in sorted(near, key=lambda r: r["best_gold"]):
        q, top = r["q"], r["ranked"][0]
        gold_pages = {p for d, p in r["gold"] if d == top[0]}
        if top[0] != q["doc_name"]:
            kind = "other filing"
        elif any(abs(top[1] - p) <= 1 for p in gold_pages):
            kind = "adjacent page"
        else:
            kind = "same filing, other page"
        nums = answer_numbers(q["answer"])
        text = " ".join(pages_text.get(top, []))
        has_answer = bool(nums) and any(n in text for n in nums)
        kinds[kind] += 1
        kinds["rank-1 page contains answer number"] += has_answer
        gold_desc = ", ".join(f"{d} p{p}" for d, p in sorted(r["gold"]))
        print(f"- gold rank {r['best_gold']} [{q['question_type']}] {q['question'][:140]}")
        print(f"    rank 1: {top[0]} p{top[1]} (score {r['page_score'][top]:.3f}) -> {kind}"
              f"{', contains answer number' if has_answer else ''}")
        best = r["ranked"][r["best_gold"] - 1]
        print(f"    gold:   {gold_desc} (best score {r['page_score'][best]:.3f})  answer: {q['answer'][:60]!r}")
    print("\nsummary of rank-1 pages in rank 2-10 cases:", dict(kinds))

    print(f"\n-- gold rank >10 or not in pool --")
    for r in far + lost:
        q = r["q"]
        print(f"- gold rank {r['best_gold'] or 'not in pool'} [{q['question_type']}] {q['question'][:140]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--pool", type=int, default=100)
    args = parser.parse_args()
    main(header=not args.no_header, pool=args.pool)
