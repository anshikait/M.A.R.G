"""Step 6 baselines on your query files.
  python -m eval.evaluate_nlu --split val
Intent: rules vs TF-IDF+LR (trained on train.jsonl) -> accuracy, macro-F1.   Slots: regex baseline vs true_profile.
Slots are scored on ENGLISH rows only (other languages need the translation layer first) and only for attributes the
query was meant to contain (true_profile minus missing_fields). Recall is a lower bound: some profile fields
(e.g. residence, BPL) are never stated in text."""
import argparse
from collections import defaultdict
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from src.data_prep import loaders as L
from src.nlu import intent as I
from src.nlu.slots import extract_slots
from src.translation.translator import normalise_numbers

ATTRS = ["age", "gender", "state", "caste_category", "annual_income", "occupation", "disability", "bpl", "residence"]
MISS = {"caste": "caste_category", "income": "annual_income"}


def predicted(row, states):
    text = normalise_numbers(" ".join(map(str, row.turns)) if len(row.turns) > 1 else row.query)
    ext = extract_slots(text, states)
    prof = {}
    for s in ext["slots"]:
        if s["subject"] == "self" or ext["asker"] == "relative": prof[s["attribute"]] = s["value"]
    return prof, ext, text


def run(split):
    states = sorted({str(s).lower() for s in L.load_registry()["state"] if str(s).strip() and str(s).lower() != "all"}) \
        if L.S.REGISTRY_CSV.exists() else None
    q = L.load_queries(split); tr = L.load_queries("train")
    names = L.load_registry()["name"] if L.S.REGISTRY_CSV.exists() else []
    nidx = I.build_name_index(names)
    I.MODEL_PATH.unlink(missing_ok=True)
    pred_rule, tp, fp, fn = [], defaultdict(int), defaultdict(int), defaultdict(int)
    for r in q.itertuples():
        prof, ext, text = predicted(r, states)
        ncore = sum(a in prof for a in I.CORE)
        pred_rule.append(I.rule_intent(r.query, ncore, nidx))
        if r.language != "english" or not r.profile: continue
        missing = {MISS.get(m, m) for m in r.missing}
        for a in ATTRS:
            g = r.profile.get(a)
            if g is None or a in missing or g is False: continue
            if prof.get(a) == g or (a == "annual_income" and a in prof and abs(prof[a] - g) <= 0.01 * g): tp[a] += 1
            else: fn[a] += 1
        for a, v in prof.items():
            if a in r.profile and r.profile[a] != v and not (a == "annual_income" and abs(v - r.profile[a]) <= 0.01 * r.profile[a]):
                fp[a] += 1
    print(f"INTENT (rules)  acc={accuracy_score(q.row_type, pred_rule):.3f}  macroF1={f1_score(q.row_type, pred_rule, average='macro'):.3f}")
    if len(tr) >= 20:
        pipe = I.train(tr["query"], tr["row_type"]); pred_ml = pipe.predict(q["query"])
        print(f"INTENT (tfidf+LR) acc={accuracy_score(q.row_type, pred_ml):.3f}  macroF1={f1_score(q.row_type, pred_ml, average='macro'):.3f}")
        print(classification_report(q.row_type, pred_ml, zero_division=0))
    rows = []
    for a in ATTRS:
        p_, r_ = tp[a] / max(tp[a] + fp[a], 1), tp[a] / max(tp[a] + fn[a], 1)
        rows.append({"attr": a, "tp": tp[a], "fp": fp[a], "fn": fn[a], "P": round(p_, 3), "R": round(r_, 3),
                     "F1": round(2 * p_ * r_ / max(p_ + r_, 1e-9), 3)})
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    print("\nSLOTS (English rows, regex baseline)"); print(pd.DataFrame(rows).to_string(index=False))
    print(f"micro  P={T/max(T+F,1):.3f} R={T/max(T+N,1):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="val"); run(ap.parse_args().split)
