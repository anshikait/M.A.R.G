"""Eligibility text -> list[Constraint].  Regex baseline + LLM silver-labelling prompt.

  python -m src.reasoning.constraint_extractor            # runs regex baseline over all schemes
Output: data/processed/constraints.jsonl   {"scheme_id":..., "constraints":[...]}
"""
import json, re
from typing import Callable, List, Optional
from configs import settings as S
from configs.constraint_schema import Constraint, STATES

MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7,
        "thousand": 1e3, "k": 1e3, "hazaar": 1e3, "hazar": 1e3, "hajar": 1e3, "l": 1e5}
CASTE_PAT = {
    "sc": r"\bscheduled castes?\b|\bsc\b", "st": r"\bscheduled tribes?\b|\bst\b",
    "obc": r"\bother backward class(?:es)?\b|\bobc\b", "ews": r"\bews\b|economically weaker",
    "general": r"\bgeneral (?:category|caste)\b",
}
OCC_PAT = {
    "farmer": r"\bfarmers?\b|\bfarming\b|\bagricultur", "student": r"\bstudents?\b",
    "fisherman": r"\bfisher", "artisan": r"\bartisans?\b|\bweavers?\b|\bcraftsm",
    "worker": r"\blabou?rers?\b|\bwage earners?\b|\bunorganised workers?\b",
    "entrepreneur": r"\bentrepreneurs?\b|\bself[- ]employed\b|\bstart[- ]?ups?\b",
    "vendor": r"\bstreet vendors?\b|\bvendors?\b|\bhawkers?\b",
    "unemployed": r"\bunemployed\b|\bjobseekers?\b",
}
MARITAL_PAT = {"widow": r"\bwidows?\b", "unmarried": r"\bunmarried\b", "divorced": r"\bdivorced\b|\bdeserted\b"}


def parse_amount(num: str, unit: Optional[str]) -> float:
    v = float(num.replace(",", ""))
    return v * MULT.get((unit or "").lower(), 1)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\b(Rs|Re|No|Dr|Sh|Smt)\.", r"\1", str(text))      # keep abbreviations from splitting
    parts = re.split(r"(?<=[.;])\s+|\n+|•|\u2022|\s-\s", str(text))
    return [p.strip(" -*") for p in parts if len(p.strip()) > 8]


def _find(patterns, s):
    return [k for k, p in patterns.items() if re.search(p, s, re.I)]


def extract_sentence(s: str) -> List[Constraint]:
    out: List[Constraint] = []
    low = s.lower()

    # ---- age
    if re.search(r"\bage\b|\byears?\b|\byrs\b", low):
        m = re.search(r"(\d{1,2})\s*(?:to|and|-|–)\s*(\d{1,2})\s*(?:years?|yrs?)", low)
        if m:
            out.append(Constraint("age", "between", [int(m[1]), int(m[2])], s, "years"))
        else:
            m = re.search(r"(?:minimum age(?: of)?|at least|not less than|above|over|aged)\s*(\d{1,2})\s*(?:years?|yrs?)?", low) \
                or re.search(r"(\d{1,2})\s*(?:years?|yrs?)\s*(?:of age\s*)?(?:or|and)\s*(?:above|older|more)", low)
            if m:
                out.append(Constraint("age", ">=", int(m[1]), s, "years"))
            m = re.search(r"(?:maximum age(?: of)?|not (?:more|older|above) than|below|under|up to|upto|not exceed(?:ing)?)\s*(\d{1,2})\s*(?:years?|yrs?)", low) \
                or re.search(r"(\d{1,2})\s*(?:years?|yrs?)\s*(?:of age\s*)?(?:or|and)\s*(?:below|younger|less)", low)
            if m:
                out.append(Constraint("age", "<=", int(m[1]), s, "years"))

    # ---- income
    m = re.search(r"income[^.;]{0,80}?(?:not (?:exceed(?:ing)?|more than)|less than|below|under|up to|upto|"
                  r"does not exceed|maximum(?: of)?|within|should be)\s*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|thousand)?", low)
    if m:
        c = Constraint("annual_income", "<=", parse_amount(m[1], m[2]), s, "INR")
        # exception clause: "except SC/ST", "excluding ..."
        ex = re.search(r"(?:except|excluding|other than|exempt(?:ed)? for)\s+([^.;]+)", low)
        if ex:
            cats = _find(CASTE_PAT, ex[1])
            if cats:
                c.exceptions.append({"attribute": "caste_category", "operator": "in", "value": cats})
        out.append(c)

    # ---- gender
    f = re.search(r"\bwomen\b|\bwoman\b|\bfemale\b|\bgirls?\b", low)
    mm = re.search(r"\bmen\b|\bmale\b|\bboys?\b", low) and not re.search(r"\bfemale\b|\bwomen\b", low)
    if f and not re.search(r"\bmen and women\b|\bwomen and men\b|\bmale and female\b", low):
        out.append(Constraint("gender", "in", ["female"], s))
    elif mm:
        out.append(Constraint("gender", "in", ["male"], s))

    # ---- caste / category (OR-ed within one sentence)
    cats = _find(CASTE_PAT, low)
    if cats and re.search(r"belong|categor|caste|tribe|class|reserved|member|candidate|applicant|student", low) \
            and not re.search(r"except|excluding|other than|exempt", low):
        out.append(Constraint("caste_category", "in", cats, s))

    # ---- occupation / marital
    occ = _find(OCC_PAT, low)
    if occ:
        out.append(Constraint("occupation", "in", occ, s))
    mar = _find(MARITAL_PAT, low)
    if mar:
        out.append(Constraint("marital_status", "in", mar, s))

    # ---- residence, disability, bpl, state
    res = [r for r in ("rural", "urban") if re.search(rf"\b{r}\b", low)]
    if len(res) == 1:
        out.append(Constraint("residence", "in", res, s))
    if re.search(r"disabilit|divyang|\bpwd\b|handicap|differently[- ]abled", low):
        out.append(Constraint("disability", "==", True, s))
    if re.search(r"below poverty line|\bbpl\b", low):
        out.append(Constraint("bpl", "==", True, s))
    if re.search(r"resident of|domicile|permanent resident|native of|belong(?:s|ing)? to", low):
        sts = [st for st in STATES if re.search(rf"\b{st}\b", low)]
        if sts:
            out.append(Constraint("state", "in", sts, s))
    return out


def extract(text: str) -> List[Constraint]:
    cons: List[Constraint] = []
    for s in split_sentences(text):
        cons.extend(extract_sentence(s))
    # de-duplicate identical (attribute, operator, value)
    seen, uniq = set(), []
    for c in cons:
        key = (c.attribute, c.operator, json.dumps(c.value, sort_keys=True))
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq


# ----------------------------------------------------------------------------------
# LLM silver labelling (plug in any callable: prompt:str -> str).  Verify >=300 by hand.
SILVER_PROMPT = """Extract eligibility constraints from the scheme text as a JSON list.
Each item: {{"attribute": one of [age, annual_income, gender, caste_category, state, occupation,
education, marital_status, residence, disability, bpl, text],
"operator": one of [<=, <, >=, >, ==, between, in, not_in],
"value": number | string | boolean | list, "source_sentence": exact sentence,
"exceptions": [{{"attribute":..., "operator":"in", "value":[...]}}]}}.
Use "text" for conditions that fit none of the attributes. Income in rupees (convert lakh/crore).
Return ONLY the JSON list, nothing else.

SCHEME TEXT:
{text}
"""


def scheme_state_constraint(level, state):
    """State-level schemes are for residents of that state, even when the eligibility text never says so
    (this is how rag_schemes.json rules get their `state`). Central schemes ('All') get no state constraint."""
    from configs.constraint_schema import norm_state
    st = str(state or "").strip().lower()
    if str(level).lower() == "state" and st and st not in ("all", "nan"):
        return Constraint("state", "in", [norm_state(st)], "scheme is run by this state (registry metadata)")
    return None


def extract_with_llm(text: str, llm: Callable[[str], str]) -> List[Constraint]:
    raw = llm(SILVER_PROMPT.format(text=text))
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return [Constraint.from_dict(d) for d in json.loads(raw)]
    except Exception:
        return []


def run_all(out_path=None):
    """Regex baseline over eligibility text of the schemes that have details (1,482). Output keyed by slug."""
    from src.data_prep.loaders import load_details, load_registry
    reg = load_registry().set_index("slug")
    out_path = out_path or (S.CACHE / "constraints_regex.jsonl"); S.CACHE.mkdir(parents=True, exist_ok=True)
    det, n_with = load_details(), 0
    with open(out_path, "w") as f:
        for slug, d in det.items():
            cons = extract(d["eligibility"])
            if slug in reg.index and not any(c.attribute == "state" for c in cons):
                sc = scheme_state_constraint(reg.loc[slug, "level"], reg.loc[slug, "state"])
                if sc: cons.append(sc)
            n_with += bool(cons)
            f.write(json.dumps({"slug": slug, "constraints": [c.to_dict() for c in cons]}) + "\n")
    print(f"wrote {out_path} | with >=1 constraint: {n_with}/{len(det)}")


if __name__ == "__main__":
    run_all()
