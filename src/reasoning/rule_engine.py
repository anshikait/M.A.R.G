"""Three-valued (satisfied / unsatisfied / unknown) evaluation of constraints against a profile."""
from typing import Dict, List, Tuple
from configs.constraint_schema import Constraint, Tri, NUMERIC


def _norm(v):
    if isinstance(v, str): return v.strip().lower()
    return v


def _as_set(v):
    if v is None: return None
    if isinstance(v, (list, tuple, set)): return {_norm(x) for x in v}
    return {_norm(v)}


def _match(attr, op, value, profile: Dict) -> Tri:
    pv = profile.get(attr)
    if pv is None:
        return Tri.UNKNOWN
    if attr in NUMERIC:
        pv = float(pv)
        if op == "<=": ok = pv <= value
        elif op == "<": ok = pv < value
        elif op == ">=": ok = pv >= value
        elif op == ">": ok = pv > value
        elif op == "==": ok = pv == value
        elif op == "between": ok = value[0] <= pv <= value[1]
        else: return Tri.UNKNOWN
        return Tri.SATISFIED if ok else Tri.UNSATISFIED
    if op == "in":
        return Tri.SATISFIED if _as_set(pv) & _as_set(value) else Tri.UNSATISFIED
    if op == "not_in":
        return Tri.UNSATISFIED if _as_set(pv) & _as_set(value) else Tri.SATISFIED
    if op == "==":
        return Tri.SATISFIED if _norm(pv) == _norm(value) else Tri.UNSATISFIED
    return Tri.UNKNOWN


def evaluate_constraint(c: Constraint, profile: Dict) -> Tuple[Tri, str]:
    if c.attribute == "text":
        return Tri.UNKNOWN, "needs NLI / manual check"
    status = _match(c.attribute, c.operator, c.value, profile)
    reason = f"{c.attribute} {c.operator} {c.value}; profile has {profile.get(c.attribute)!r}"
    if status == Tri.UNSATISFIED:
        for ex in c.exceptions:                      # exempt categories override a failed condition
            if _match(ex["attribute"], ex.get("operator", "in"), ex["value"], profile) == Tri.SATISFIED:
                return Tri.SATISFIED, reason + f" (exempt via {ex['attribute']})"
    return status, reason


def evaluate_scheme(constraints: List[Constraint], profile: Dict) -> Dict:
    """Constraints sharing a `group` are OR-ed; all other constraints are AND-ed."""
    details, groups = [], {}
    for c in constraints:
        st, why = evaluate_constraint(c, profile)
        item = {"attribute": c.attribute, "constraint": f"{c.operator} {c.value}", "status": st.value,
                "reason": why, "source": c.source_sentence}
        details.append(item)
        if c.group: groups.setdefault(c.group, []).append(st)
    solo = [Tri(d["status"]) for d, c in zip(details, constraints) if not c.group]
    for sts in groups.values():                      # OR group
        if Tri.SATISFIED in sts: solo.append(Tri.SATISFIED)
        elif Tri.UNKNOWN in sts: solo.append(Tri.UNKNOWN)
        else: solo.append(Tri.UNSATISFIED)
    if Tri.UNSATISFIED in solo: overall = Tri.UNSATISFIED
    elif Tri.UNKNOWN in solo or not solo: overall = Tri.UNKNOWN
    else: overall = Tri.SATISFIED
    missing = sorted({d["attribute"] for d in details if d["status"] == "unknown" and d["attribute"] != "text"})
    return {"overall": overall.value, "details": details, "missing": missing,
            "n_constraints": len(constraints)}
