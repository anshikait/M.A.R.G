"""Tuned dense retriever -> top-POOL -> cross-encoder rerank. Prints dense vs reranked side by side. (Self-contained.)
  python -m eval.evaluate_rerank --split val --index all --retriever <bi-encoder dir> --reranker <reranker dir>
  add  --limit 300 --pool 20  for a quick CPU-friendly check."""
import argparse
import numpy as np, pandas as pd
from configs import settings as S
from src.data_prep import loaders as L
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, rerank_orders


def metrics(orders, slugs, q):
    pos = {s: i for i, s in enumerate(slugs)}
    ranks, anyrel, prel = [], [], []
    for gold, rel, order in zip(q["slug"], q["relevant"], orders):
        order = list(order)
        ranks.append(order.index(pos[gold]) + 1 if pos.get(gold) in order else 10**6)
        top3 = {slugs[i] for i in order[:3]}; rel = set(rel) | {gold}
        anyrel.append(bool(top3 & rel)); prel.append(len(top3 & rel) / 3)
    r = np.array(ranks)
    return pd.DataFrame({"hit3": r <= 3, "hit10": r <= 10, "hit20": r <= 20, "hit30": r <= 30,
                         "rr": np.where(r < 10**6, 1 / r, 0.0), "anyrel3": anyrel, "prel3": prel}, index=q.index)


def summary(m):
    return (f"Recall@3={m.hit3.mean():.3f}  Recall@10={m.hit10.mean():.3f}  Recall@20={m.hit20.mean():.3f}  "
            f"Recall@30={m.hit30.mean():.3f}  MRR={m.rr.mean():.3f}  | any-relevant@3={m.anyrel3.mean():.3f}  "
            f"relevant-precision@3={m.prel3.mean():.3f}")


def table(q, m, by):
    g = pd.concat([q[[by]], m], axis=1).groupby(by)
    t = g.agg(n=("hit3", "size"), R3=("hit3", "mean"), R10=("hit10", "mean"), R30=("hit30", "mean"),
              MRR=("rr", "mean"), anyrel3=("anyrel3", "mean"))
    return t.sort_values("n", ascending=False).round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--index", choices=["all", "split"], default="all")
    ap.add_argument("--retriever", required=True); ap.add_argument("--reranker", required=True)
    ap.add_argument("--pool", type=int, default=30); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-vague", action="store_true")
    a = ap.parse_args()
    docs = L.card_frame()
    if a.index == "split": docs = docs[docs.split == a.split]
    q = L.load_queries(a.split); q = q[q.slug.notna() & q.slug.isin(docs.slug)].copy()
    if not a.include_vague: q = q[q.row_type != "vague"].copy()
    if a.limit and a.limit < len(q): q = q.sample(a.limit, random_state=0).copy()
    print(f"index={a.index} docs={len(docs)} | scored queries={len(q)} | pool={a.pool}", flush=True)
    ret = HybridRetriever(docs, use_dense=True, model_name=a.retriever)
    orders = [[i for i, _ in lst] for lst in ret.rank_lists(q["query"].tolist(), "dense", pool=max(100, a.pool))]
    slugs, texts = ret.df["slug"].tolist(), ret.df["text"].tolist()
    base = metrics(orders, slugs, q)
    print("\n######## dense (no rerank)  ", summary(base), flush=True)
    rr = CrossEncoderReranker(a.reranker)
    new = metrics(rerank_orders(orders, texts, q["query"].tolist(), rr.score, a.pool), slugs, q)
    print("######## dense + reranker   ", summary(new))
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    for by in ("language", "noise", "row_type"):
        t = table(q, new, by); print(f"\n-- reranked, by {by}\n{t.to_string()}")
        t.to_csv(S.RESULTS / f"rerank_{a.split}_{a.index}_pool{a.pool}_by_{by}.csv")