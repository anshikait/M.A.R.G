"""Step 7 check: rule engine vs the dataset's own eligibility ground truth.
Gold = eligible_same_category_slugs (eligible schemes inside the query's category) and relevant_slugs.
Only schemes that have a rule in rag_schemes.json can be judged.
  python -m eval.evaluate_eligibility --split val
NOTE: this ground truth is produced FROM rag_schemes.json rules, so a high score proves the engine is
implemented consistently; it does not measure accuracy of rules extracted from raw eligibility text (see evaluate_constraints)."""
import argparse
import pandas as pd
from src.data_prep import loaders as L
from src.reasoning.rag_rules import all_constraints
from src.reasoning.rule_engine import evaluate_scheme


def run(split, limit=None):
    cons = all_constraints(L.load_rag_rules())
    cat = {k: v["category"] for k, v in L.load_rag_meta().items()}
    q = L.load_queries(split); q = q[(q.row_type == "standard") & q.profile.map(bool)]
    if limit: q = q.head(limit)
    res = {}
    for gold_col, scoped in (("elig_same", True), ("relevant", True), ("relevant", False)):
        for strict in (False, True):                 # lenient: not-unsatisfied counts as eligible; strict: satisfied only
            tp = fp = fn = 0
            for r in q.itertuples():
                gold = set(getattr(r, gold_col)) & set(cons)
                rcat = r.category
                for slug, c in cons.items():
                    if scoped and cat.get(slug) != rcat: continue
                    o = evaluate_scheme(c, r.profile)["overall"]
                    pred = (o == "satisfied") if strict else (o != "unsatisfied")
                    tp += pred and slug in gold; fp += pred and slug not in gold; fn += (not pred) and slug in gold
            p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            res[(gold_col, "category-scoped" if scoped else "all schemes", "strict" if strict else "lenient")] = \
                {"TP": tp, "FP": fp, "FN": fn, "precision": round(p, 3), "recall": round(rc, 3),
                 "F1": round(2 * p * rc / max(p + rc, 1e-9), 3)}
    print(pd.DataFrame(res).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); run(a.split, a.limit)
