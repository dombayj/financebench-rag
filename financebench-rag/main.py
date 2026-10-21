"""CLI entry point: build the naive RAG index, then evaluate retrieval against it."""
import argparse

import build_index
import evaluate


def main():
    parser = argparse.ArgumentParser(description="Naive RAG retrieval baseline for FinanceBench")
    parser.add_argument("command", choices=["build", "eval", "eval-routed", "eval-routed-company",
                                                "eval-routed-top2", "eval-routed-top3", "all"])
    parser.add_argument("--search", choices=["dense", "bm25", "hybrid"], default="dense",
                        help="retrieval method for the eval commands")
    parser.add_argument("--rerank", nargs="?", const="qwen", choices=["qwen", "cohere"],
                        help="rerank the retrieved pool: local Qwen3 (default, RERANK_MODEL) "
                             "or the Cohere API (COHERE_RERANK_MODEL)")
    parser.add_argument("--rerank-header", action="store_true",
                        help="prepend each chunk's filing name to the text the reranker sees")
    parser.add_argument("--pool", type=int, default=evaluate.POOL_K,
                        help="how many chunks to retrieve (and rerank) per question")
    args = parser.parse_args()

    if args.command in ("build", "all"):
        build_index.main()
    if args.command in ("eval", "all"):
        evaluate.main(method=args.search, rerank=args.rerank,
                      pool=args.pool, header=args.rerank_header)
    if args.command == "eval-routed":
        evaluate.main(mode="filing", method=args.search, rerank=args.rerank,
                      pool=args.pool, header=args.rerank_header)
    if args.command == "eval-routed-company":
        evaluate.main(mode="company", method=args.search, rerank=args.rerank,
                      pool=args.pool, header=args.rerank_header)
    if args.command in ("eval-routed-top2", "eval-routed-top3"):
        evaluate.main(mode=args.command[-4:], method=args.search, rerank=args.rerank,
                      pool=args.pool, header=args.rerank_header)


if __name__ == "__main__":
    main()
