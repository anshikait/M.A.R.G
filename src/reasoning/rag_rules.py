"""rag_schemes.json `rule` dict -> list[Constraint].  Keys seen in your data:
state, occupation (str|list), min_age, max_age, max_income, castes (list), gender, disability, bpl, residence.
null / "all" => no restriction. (A rule with no keys = open to everyone.)"""
from typing import Dict, List
from configs.constraint_schema import Constraint, ANY_VALUES, norm_state, norm_caste, GENDER_ALIASES


def _vals(v):
    if v is None: return []
    out = []
    for x in (v if isinstance(v, (list, tuple, set)) else [v]):
        s = str(x).strip().lower()
        if s not in ANY_VALUES: out.append(s)
    return out


def _num(v):
    try: return None if v is None else float(v)
    except (TypeError, ValueError): return None


def rule_to_constraints(rule: Dict) -> List[Constraint]:
    r, src, out = rule or {}, "rag_schemes.json rule", []
    lo, hi = _num(r.get("min_age")), _num(r.get("max_age"))
    if lo and hi: out.append(Constraint("age", "between", [lo, hi], src, "years"))
    elif lo: out.append(Constraint("age", ">=", lo, src, "years"))
    elif hi: out.append(Constraint("age", "<=", hi, src, "years"))
    inc = _num(r.get("max_income"))
    if inc: out.append(Constraint("annual_income", "<=", inc, src, "INR"))
    for key, attr, fn in (("state", "state", norm_state), ("occupation", "occupation", str),
                          ("castes", "caste_category", norm_caste), ("residence", "residence", str)):
        vs = [fn(v) for v in _vals(r.get(key))]
        if vs: out.append(Constraint(attr, "in", vs, src))
    g = [GENDER_ALIASES.get(v, v) for v in _vals(r.get("gender"))]
    if g: out.append(Constraint("gender", "in", g, src))
    if r.get("disability") is True: out.append(Constraint("disability", "==", True, src))
    if r.get("bpl") is True: out.append(Constraint("bpl", "==", True, src))
    return out


def all_constraints(rules: Dict[str, Dict]) -> Dict[str, List[Constraint]]:
    return {slug: rule_to_constraints(rule) for slug, rule in rules.items()}
