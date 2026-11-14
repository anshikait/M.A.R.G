"""Run FIRST:  python -m src.data_prep.inspect_data      (prints the real schema of every data file)
Then fix configs/settings.py FIELDS if a field name was not recognised."""
import json, sys
import pandas as pd
from configs import settings as S
from src.data_prep import loaders as L


def show(title, path):
    print(f"\n=== {title}  ({path.relative_to(S.ROOT) if path.exists() else path})")
    if not path.exists(): print("  MISSING"); return
    data = L.read_any(path)
    if isinstance(data, dict): print("  dict keys:", list(data)[:12]); data = list(data.values())[:1] or data
    n = len(data) if hasattr(data, "__len__") else "?"
    rec = data[0] if isinstance(data, list) and data else {}
    print("  records:", n); print("  keys   :", list(rec)[:25] if isinstance(rec, dict) else type(rec))
    if isinstance(rec, dict): print("  sample :", json.dumps(rec, ensure_ascii=False, default=str)[:350])


if __name__ == "__main__":
    show("registry", S.REGISTRY_CSV); show("split json", S.SPLIT_JSON); show("cards", S.CARDS)
    for sp, p in S.QUERY_FILES.items(): show(f"queries/{sp}", p)
    show("details", S.DETAILS); show("rag rules", S.RAG_RULES); show("hindi list", S.HI_LIST)
    print("\n=== recognised fields (query file)")
    if S.QUERY_FILES["val"].exists():
        rec = L.read_jsonl(S.QUERY_FILES["val"])[0]
        for k in S.FIELDS:
            if k in ("slug","name","card","eligibility","benefits","application","rule"): continue
            print(f"  {k:10s} -> {'OK ' if L.pick(rec, k) is not None else 'NOT FOUND'}")
        q = L.load_queries("val")
        for c in ("language", "noise", "row_type", "asker"): print(f"\n{c}:\n", q[c].value_counts().head(20).to_string())
