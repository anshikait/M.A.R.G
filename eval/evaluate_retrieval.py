"""Step 4 baselines. Vague rows are excluded by default (as in the plan baseline); add --include-vague to score them.  Metrics: Recall@3, Recall@10, MRR, overall and by language / noise / row type.
  python -m eval.evaluate_retrieval --split val --modes bm25                      # reproduce your 32.1% baseline first
  python -m eval.evaluate_retrieval --split val --modes bm25 dense hybrid         # GPU/Colab for BGE-M3
  python -m eval.evaluate_retrieval --split val --index split --modes bm25        # if the baseline only matches this way
Rows without a target scheme (offtopic) are not scored here."""
import argparse
import numpy as np, pandas as pd
from configs import settings as S
from src.data_prep import loaders as L
from src.retrieval.hybrid_retriever import HybridRetriever


def score(ret, q: pd.DataFrame, mode, state_filter="none", bm25_weight=1.0):
    mask = ret.state_masks([p.get("state") for p in q["profile"]]) if state_filter == "oracle" else None
    ranked = ret.rank_lists(q["query"].tolist(), mode, mask=mask, bm25_weight=bm25_weight)
    pos = {s: i for i, s in enumerate(ret.df["slug"])}
    ranks = []
    for gold, lst in zip(q["slug"], ranked):
        order = [i for i, _ in lst]
        ranks.append(order.index(pos[gold]) + 1 if pos.get(gold) in order else 10**6)
    r = np.array(ranks)
    return pd.DataFrame({"hit3": r <= 3, "hit10": r <= 10, "rr": np.where(r < 10**6, 1 / r, 0.0)}, index=q.index)


def table(q, m, by):
    g = pd.concat([q[[by]], m], axis=1).groupby(by)
    t = g.agg(n=("hit3", "size"), R3=("hit3", "mean"), R10=("hit10", "mean"), MRR=("rr", "mean"))
    return t.sort_values("n", ascending=False).round(3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--modes", nargs="+", default=["bm25"])
    ap.add_argument("--index", choices=["all", "split"], default="all"); ap.add_argument("--include-vague", action="store_true"); ap.add_argument("--bm25-weights", type=float, nargs="+", default=[1.0], help="hybrid only: BM25 weight(s) vs dense=1.0; e.g. 1 0.5 0.25 0.1 0")
    ap.add_argument("--state-filter", choices=["none", "oracle"], default="none"); ap.add_argument("--model", default=None)
    a = ap.parse_args()
    docs = L.card_frame()
    if a.index == "split": docs = docs[docs.split == a.split]
    q = L.load_queries(a.split); q = q[q.slug.notna() & q.slug.isin(docs.slug)].copy()
    if not a.include_vague:                      # your plan's baseline (1,610 queries) excludes vague rows
        q = q[q.row_type != "vague"].copy()
    print(f"index={a.index} docs={len(docs)} | scored queries={len(q)} | state_filter={a.state_filter}")
    print("doc state values:", docs.state.astype(str).value_counts().head(4).to_dict(), "| levels:", docs.level.astype(str).value_counts().to_dict())
    ret = HybridRetriever(docs, use_dense=any(m != "bm25" for m in a.modes), model_name=a.model)
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    runs = []
    for mode in a.modes:
        for w in (a.bm25_weights if mode == "hybrid" else [None]):
            runs.append((mode, w))
    for mode, w in runs:
        tag = mode if w is None else f"{mode}_w{w:g}"
        m = score(ret, q, mode, a.state_filter, 1.0 if w is None else w)
        print(f"\n######## {tag}   Recall@3={m.hit3.mean():.3f}  Recall@10={m.hit10.mean():.3f}  MRR={m.rr.mean():.3f}")
        if len(runs) > 3 and w is not None:      # weight sweep: one line each, details only for the best later
            continue
        for by in ("language", "noise", "row_type"):
            t = table(q, m, by); print(f"\n-- by {by}\n{t.to_string()}")
            t.to_csv(S.RESULTS / f"retrieval_{a.split}_{a.index}_{a.state_filter}_{tag}_by_{by}.csv")