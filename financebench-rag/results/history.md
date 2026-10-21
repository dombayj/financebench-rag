| timestamp | n | hit@5 | recall@5 | mrr | ndcg@5 | notes |
|---|---|---|---|---|---|---|
| 20261016T121318Z | 150 | 0.3733 | 0.3433 | 0.2475 | 0.2402 | naive RAG baseline (400tok/15% overlap, text-embedding-3-small) |
| 20261016T220458Z | 150 | 0.46 | 0.4233 | 0.3287 | 0.3205 | filing routing (gpt-4o-mini, route acc 0.84) + naive RAG |
| 20261016T222259Z | 150 | 0.4133 | 0.38 | 0.2746 | 0.2672 | company routing + NONE fallback (gpt-4o-mini, company acc 0.98) + naive RAG |
| 20261016T223118Z | 150 | 0.4333 | 0.3967 | 0.2808 | 0.2761 | top-2 filing routing (gpt-4o-mini, gold filing in top-2 0.9267) + naive RAG |
| 20261016T223139Z | 150 | 0.4067 | 0.3733 | 0.2687 | 0.2607 | top-3 filing routing (gpt-4o-mini, gold filing in top-3 0.96) + naive RAG |
| 20261017T001026Z | 150 | 0.2133 | 0.2022 | 0.1639 | 0.1642 | filing routing (gpt-4o-mini, route acc 0.84) + BM25 only |
| 20261017T001048Z | 150 | 0.3067 | 0.2856 | 0.2112 | 0.2101 | filing routing (gpt-4o-mini, route acc 0.84) + hybrid dense+BM25 (RRF k=60) |
| 20261017T112144Z | 150 | 0.7267 | 0.6967 | 0.6034 | 0.6109 | filing routing (gpt-4o-mini, route acc 0.84) + naive RAG + rerank top-50 (rerank-v4.0-pro) |
| 20261017T123658Z | 150 | 0.7333 | 0.7067 | 0.5987 | 0.6069 | filing routing (gpt-4o-mini, route acc 0.84) + naive RAG + rerank top-50 (rerank-v4.0-pro, filing-name header) |
| 20261017T125405Z | 150 | 0.82 | 0.7933 | 0.6465 | 0.6646 | top-3 filing routing (gpt-4o-mini, gold filing in top-3 0.96) + naive RAG + rerank top-100 (rerank-v4.0-pro, filing-name header) |
| 20261018T090736Z | 150 | 0.7867 | 0.7567 | 0.6205 | 0.6353 | top-3 filing routing (gpt-4o-mini, gold filing in top-3 0.96) + naive RAG + rerank top-100 (rerank-v4.0-pro) |
