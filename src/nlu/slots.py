"""Profile slot extraction from (English) user text.  Regex baseline - replace/compare with DeBERTa NER.

Returns {"slots":[{attribute,value,subject,span}], "hypothetical":bool, "correction":bool}
subject = "self" | "other"  (e.g. 'my father is a farmer' -> other, never written to the profile)
"""
import re
from configs.constraint_schema import STATES, STATE_ABBR
from src.reasoning.constraint_extractor import OCC_PAT, MARITAL_PAT, CASTE_PAT, parse_amount

THIRD = r"\bmy (?:father|mother|husband|wife|son|daughter|brother|sister|parents?|friend|neighbou?r|uncle|aunt|grand\w+)\b"
HYPO = r"\bwhat if\b|\bsuppose\b|\bassume\b|\bin case\b|\bif i\b|\bif my\b|\bhypothetical"
CORR = r"\bactually\b|\bcorrection\b|\bsorry\b|\bi meant\b|\bupdate\b|\bchange (?:it|my)\b|\bnot \d"


def _clauses(text):
    return [c.strip() for c in re.split(r"[.;,?!\n]|\bbut\b|\band my\b", text) if c.strip()]


def extract_slots(text: str, states=None) -> dict:
    states = states or STATES
    low = text.lower()
    slots = []
    def add(attr, val, clause, subj):
        slots.append({"attribute": attr, "value": val, "subject": subj, "span": clause})
    for cl in _clauses(low):
        subj = "other" if re.search(THIRD, cl) else "self"
        m = re.search(r"(\d{1,3})[- ]?(?:years?|yrs?)[- ]?old|\bage(?: is|:)?\s*(\d{1,3})|\baged\s*(\d{1,3})|\bi am (\d{2})\b|\bi'm (\d{2})\b", cl)
        if m:
            add("age", int(next(g for g in m.groups() if g)), cl, subj)
        m = re.search(r"(?:income|earn\w*|salary|earning)[^\d]{0,30}(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|thousand|hazaa?r|k|l)?\b(\s*(?:per month|a month|monthly|/month|pm))?", cl)
        if m:
            add("annual_income", parse_amount(m[1], m[2]) * (12 if m[3] else 1), cl, subj)
        if re.search(r"\bwoman\b|\bfemale\b|\bgirl\b|\blady\b|\bwidow\b|\bmother\b(?! of)", cl) and subj == "self" and not re.search(r"\bmy mother\b", cl):
            add("gender", "female", cl, subj)
        elif re.search(r"\bi am (?:a )?(?:man|male|boy)\b|\bi'm (?:a )?(?:man|male)\b", cl):
            add("gender", "male", cl, subj)
        for k, p in CASTE_PAT.items():
            if re.search(p, cl) and re.search(r"belong|category|caste|i am|i'm|from the", cl):
                add("caste_category", k, cl, subj)
        for st in states:
            if re.search(rf"\b{re.escape(st)}\b", cl):
                add("state", st, cl, subj)
        for ab in re.findall(r"\b([A-Z]{2})\b", text):           # 'GJ' -> gujarat (case-sensitive on the raw text)
            if ab.lower() in STATE_ABBR and ab.lower() in cl.split() and STATE_ABBR[ab.lower()] in states:
                add("state", STATE_ABBR[ab.lower()], cl, subj)
        for k, p in OCC_PAT.items():
            if re.search(p, cl) and re.search(r"\bi am\b|\bi'm\b|\bi work\b|\bmy (?:job|occupation|work|profession)\b|\b\w+ by profession\b|\bmy father\b|\bmy mother\b|\bhusband\b|\bwife\b", cl):
                add("occupation", k, cl, subj)
        for k, p in MARITAL_PAT.items():
            if re.search(p, cl): add("marital_status", k, cl, subj)
        if re.search(r"\bdisabled\b|\bdisability\b|\bdivyang\b|\bhandicap", cl) and not re.search(r"\bno disab|\bnot disab|without disab", cl):
            add("disability", True, cl, subj)
        if re.search(r"\bbpl\b|below poverty line", cl):
            add("bpl", not bool(re.search(r"\bno bpl|not bpl|don't have (?:a )?bpl|without bpl", cl)), cl, subj)
        if re.search(r"\bvillage\b|\brural\b", cl): add("residence", "rural", cl, subj)
        elif re.search(r"\bcity\b|\burban\b|\btown\b", cl): add("residence", "urban", cl, subj)
    n_other = sum(x["subject"] == "other" for x in slots)
    asker = "relative" if slots and n_other / len(slots) >= 0.5 else "self"
    return {"asker": asker, "slots": slots, "hypothetical": bool(re.search(HYPO, low)), "correction": bool(re.search(CORR, low))}
