"""NLI for fuzzy ("text") conditions + synthetic NLI training-data generator.

1) generate_pairs(): synthetic profiles x extracted constraints, auto-labelled by the rule engine
   (premise = verbalised profile, hypothesis = verbalised condition, label = entailment / contradiction / neutral).
   Use ONLY train/val schemes for training; keep a human-verified sample.
2) NLIMatcher: wraps a HF NLI model (e.g. fine-tuned DeBERTa-v3) for conditions the rules cannot handle.
   Neutral maps to UNKNOWN - this is what makes the three-valued logic work.
"""
import json, random
from typing import Dict, List
from configs import settings as S
from configs.constraint_schema import Constraint, Tri, STATES
from src.reasoning.rule_engine import evaluate_constraint

LABEL = {Tri.SATISFIED: "entailment", Tri.UNSATISFIED: "contradiction", Tri.UNKNOWN: "neutral"}


def random_profile(rng: random.Random, p_missing=0.3) -> Dict:
    prof = {"age": rng.randint(5, 80), "annual_income": rng.choice([50000, 100000, 180000, 250000, 400000, 800000, 1500000]),
            "gender": rng.choice(["male", "female"]), "caste_category": rng.choice(["general", "sc", "st", "obc", "ews"]),
            "state": rng.choice(STATES), "occupation": rng.choice(["farmer", "student", "labourer", "artisan", "entrepreneur", "unemployed"]),
            "marital_status": rng.choice(["married", "unmarried", "widow"]), "residence": rng.choice(["rural", "urban"]),
            "disability": rng.random() < 0.15, "bpl": rng.random() < 0.3}
    return {k: v for k, v in prof.items() if rng.random() > p_missing}


def verbalise_profile(p: Dict) -> str:
    parts = []
    if "age" in p: parts.append(f"I am {p['age']} years old")
    if "gender" in p: parts.append(f"I am {p['gender']}")
    if "annual_income" in p: parts.append(f"my annual family income is Rs {int(p['annual_income'])}")
    if "caste_category" in p: parts.append(f"I belong to the {p['caste_category'].upper()} category")
    if "state" in p: parts.append(f"I live in {p['state'].title()}")
    if "occupation" in p: parts.append(f"my occupation is {p['occupation'].replace('_', ' ')}")
    if "marital_status" in p: parts.append(f"my marital status is {p['marital_status']}")
    if "residence" in p: parts.append(f"I live in a {p['residence']} area")
    if "disability" in p: parts.append("I have a disability" if p["disability"] else "I do not have a disability")
    if "bpl" in p: parts.append("I have a BPL card" if p["bpl"] else "I do not have a BPL card")
    return ". ".join(parts) + "." if parts else "No details provided."


def _lst(v): return " or ".join(str(x).upper() if len(str(x)) <= 3 else str(x) for x in v)


def verbalise_condition(c: Constraint) -> str:
    """ATOMIC hypothesis for ONE constraint (a full source sentence may hold several conditions,
    which would make the auto-label wrong). Add paraphrases / Hindi variants later for robustness."""
    a, op, v = c.attribute, c.operator, c.value
    if a == "age":
        return {"between": f"The applicant is between {v[0] if isinstance(v, list) else v} and {v[1] if isinstance(v, list) else v} years old.",
                ">=": f"The applicant is at least {v} years old.", "<=": f"The applicant is at most {v} years old."}.get(op, f"Age {op} {v}.")
    if a == "annual_income":
        return f"The annual family income is at most Rs {int(v)}." if op in ("<=", "<") else f"The annual family income is at least Rs {int(v)}."
    if a == "gender": return f"The applicant is {_lst(v)}."
    if a == "caste_category": return f"The applicant belongs to the {_lst(v)} category."
    if a == "state": return f"The applicant is a resident of {_lst(v).title()}."
    if a == "occupation": return f"The applicant is a {_lst(v).replace('_', ' ')}."
    if a == "marital_status": return f"The applicant is a {_lst(v)}."
    if a == "residence": return f"The applicant lives in a {_lst(v)} area."
    if a == "disability": return "The applicant has a disability."
    if a == "bpl": return "The applicant holds a Below Poverty Line (BPL) card."
    return c.source_sentence


def generate_pairs(constraints_by_scheme: Dict[str, List[Constraint]], scheme_ids: List[str],
                   per_constraint=4, seed=S.SEED) -> List[Dict]:
    rng = random.Random(seed)
    rows = []
    for sid in scheme_ids:
        for c in constraints_by_scheme.get(sid, []):
            if c.attribute == "text": continue
            for _ in range(per_constraint):
                p = random_profile(rng)
                st, _ = evaluate_constraint(c, p)
                rows.append({"slug": sid, "premise": verbalise_profile(p),
                             "hypothesis": verbalise_condition(c), "label": LABEL[st]})
    return rows


class NLIMatcher:
    def __init__(self, model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", entail_t=0.7, contra_t=0.7):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model, top_k=None)
        self.et, self.ct = entail_t, contra_t

    def judge(self, premise: str, hypothesis: str) -> Tri:
        out = self.pipe({"text": premise, "text_pair": hypothesis})
        out = out[0] if isinstance(out[0], list) else out
        sc = {o["label"].lower(): o["score"] for o in out}
        e = sc.get("entailment", 0); c = sc.get("contradiction", 0)
        if e >= self.et: return Tri.SATISFIED
        if c >= self.ct: return Tri.UNSATISFIED
        return Tri.UNKNOWN


if __name__ == "__main__":
    """NLI pairs from rag rules; train+val schemes only (the frozen split decides, test schemes are never used)."""
    from src.data_prep import loaders as L
    from src.reasoning.rag_rules import all_constraints
    cons, sm = all_constraints(L.load_rag_rules()), L.load_split_map()
    for name in ("train", "val"):
        ids = [slug for slug in cons if sm.get(slug) == name]
        rows = generate_pairs(cons, ids)
        path = S.CACHE / f"nli_{name}.jsonl"; S.CACHE.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(name, len(ids), "schemes ->", len(rows), "pairs ->", path)
