"""Plan step 7.5: score = retrieval relevance x share of conditions met. Unsatisfied schemes are excluded.
Schemes without a rule in rag_schemes.json (the ~3,900 non-RAG schemes) are kept but flagged 'no_rule'."""
from collections import Counter
from typing import Dict, List, Optional, Tuple
from configs.constraint_schema import ATTRIBUTE_QUESTIONS, Constraint
from src.reasoning.rule_engine import evaluate_scheme


def rank_candidates(profile: Dict, candidates: List[Tuple[str, float]], constraints: Dict[str, List[Constraint]],
                    top_k=5, no_rule_factor=0.3) -> Dict:
    """candidates: [(slug, relevance in 0..1)] from the retriever."""
    kept, excluded = [], []
    for slug, rel in candidates:
        cons = constraints.get(slug)
        if cons is None:
            kept.append({"slug": slug, "bucket": "no_rule", "score": rel * no_rule_factor, "details": [], "missing": [],
                         "overall": "unknown"}); continue
        ev = evaluate_scheme(cons, profile)
        if ev["overall"] == "unsatisfied":
            excluded.append({"slug": slug, **ev}); continue
        n = ev["n_constraints"]; sat = sum(d["status"] == "satisfied" for d in ev["details"]); unk = n - sat
        share = 1.0 if n == 0 else (sat + 0.5 * unk) / n          # rule with no conditions = open to all
        bucket = "likely_eligible" if ev["overall"] == "satisfied" else "needs_more_info"
        kept.append({"slug": slug, "bucket": bucket, "score": round(rel * share, 4), **{k: ev[k] for k in ("overall", "details", "missing")}})
    kept.sort(key=lambda r: r["score"], reverse=True)
    return {"ranked": kept[:top_k], "n_excluded": len(excluded), "excluded": excluded,
            "next_question": next_question([r for r in kept if r["bucket"] == "needs_more_info"])}


def next_question(candidates: List[Dict]) -> Optional[Dict]:
    """Information-gain heuristic: ask for the attribute that is unknown in the most still-possible schemes."""
    cnt = Counter(a for r in candidates for a in r["missing"])
    if not cnt: return None
    attr, n = cnt.most_common(1)[0]
    return {"attribute": attr, "question": ATTRIBUTE_QUESTIONS.get(attr, f"Please share your {attr}."), "schemes_affected": n}
