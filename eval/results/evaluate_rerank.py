"""Step 5b evaluation: tuned dense retriever -> top-POOL -> cross-encoder rerank. Prints dense vs reranked side by side.
  python -m eval.evaluate_rerank --split val --index all --retriever <tuned bi-encoder dir> --reranker <reranker dir>
"""
import argparse
import pandas as pd
from configs import settings as S
from src.data_prep import loaders as L
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, rerank_orders
from eval.evaluate_retrieval import metrics, summary, table

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--index", choices=["all", "split"], default="all")
    ap.add_argument("--retriever", required=True); ap.add_argument("--reranker", required=True)
    ap.add_argument("--pool", type=int, default=30); ap.add_argument("--include-vague", action="store_true")
    a = ap.parse_args()
    docs = L.card_frame()
    if a.index == "split": docs = docs[docs.split == a.split]
    q = L.load_queries(a.split); q = q[q.slug.notna() & q.slug.isin(docs.slug)].copy()
    if not a.include_vague: q = q[q.row_type != "vague"].copy()
    print(f"index={a.index} docs={len(docs)} | scored queries={len(q)} | pool={a.pool}")
    ret = HybridRetriever(docs, use_dense=True, model_name=a.retriever)
    orders = [[i for i, _ in lst] for lst in ret.rank_lists(q["query"].tolist(), "dense", pool=max(100, a.pool))]
    slugs, texts = ret.df["slug"].tolist(), ret.df["text"].tolist()
    base = metrics(orders, slugs, q)
    rr = CrossEncoderReranker(a.reranker)
    new = metrics(rerank_orders(orders, texts, q["query"].tolist(), rr.score, a.pool), slugs, q)
    print("\n######## dense (no rerank)  ", summary(base)); print("######## dense + reranker   ", summary(new))
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    for by in ("language", "noise", "row_type"):
        t = table(q, new, by); print(f"\n-- reranked, by {by}\n{t.to_string()}")
        t.to_csv(S.RESULTS / f"rerank_{a.split}_{a.index}_pool{a.pool}_by_{by}.csv")