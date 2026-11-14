"""Step 7.2 check: regex extractor (from raw eligibility text) vs the structured rules in rag_schemes.json (silver labels).
  python -m eval.evaluate_constraints
Compared on schemes present in BOTH en/schemes_detail.jsonl and rag_schemes.json. Atom-level P/R per attribute.
This is the number to beat with the LLM silver-labeller and your hand-verified 300-scheme gold set."""
import json
from collections import defaultdict
import pandas as pd
from src.data_prep import loaders as L
from src.reasoning.constraint_extractor import extract, scheme_state_constraint
from src.reasoning.rag_rules import rule_to_constraints


def atoms(cons):
    out = set()
    for c in cons:
        if c.attribute == "age":
            lo, hi = (c.value if c.operator == "between" else (c.value if c.operator == ">=" else None, c.value if c.operator == "<=" else None))
            if lo is not None: out.add(("min_age", float(lo)))
            if hi is not None: out.add(("max_age", float(hi)))
        elif c.attribute == "annual_income": out.add(("max_income", float(c.value)))
        elif isinstance(c.value, list): out.update((c.attribute, str(v)) for v in c.value)
        else: out.add((c.attribute, str(c.value)))
    return out


if __name__ == "__main__":
    det, rules, meta = L.load_details(), L.load_rag_rules(), L.load_rag_meta(); both = sorted(set(det) & set(rules))
    print(f"schemes with both eligibility text and a rule: {len(both)}")
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int); exact = 0
    for s in both:
        cons = extract(det[s]["eligibility"]); sc = scheme_state_constraint(meta[s]["level"], meta[s]["state"])
        if sc and not any(c.attribute == "state" for c in cons): cons.append(sc)
        pred, gold = atoms(cons), atoms(rule_to_constraints(rules[s]))
        exact += pred == gold
        for a in pred & gold: tp[a[0]] += 1
        for a in pred - gold: fp[a[0]] += 1
        for a in gold - pred: fn[a[0]] += 1
    rows = [{"attr": a, "tp": tp[a], "fp": fp[a], "fn": fn[a], "P": round(tp[a] / max(tp[a] + fp[a], 1), 3),
             "R": round(tp[a] / max(tp[a] + fn[a], 1), 3)} for a in sorted(set(tp) | set(fp) | set(fn))]
    print(pd.DataFrame(rows).to_string(index=False)); print(f"scheme-level exact match: {exact}/{len(both)}")
