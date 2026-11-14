"""Constraint schema shared by extractor, rule engine, ranker and NLI data generator."""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, List, Optional


class Tri(str, Enum):
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"


NUMERIC = {"age", "annual_income"}
CATEGORICAL = {"gender", "caste_category", "state", "occupation", "education",
               "marital_status", "residence"}
BOOLEAN = {"disability", "bpl"}
ATTRIBUTES = NUMERIC | CATEGORICAL | BOOLEAN
OPERATORS = {"<=", "<", ">=", ">", "==", "between", "in", "not_in"}

# Question to ask the user when an attribute is missing
ATTRIBUTE_QUESTIONS = {
    "age": "What is your age?",
    "annual_income": "What is your annual family income (in rupees)?",
    "gender": "What is your gender?",
    "caste_category": "Which category do you belong to (General / SC / ST / OBC / EWS)?",
    "state": "Which state do you live in?",
    "occupation": "What is your occupation (farmer, student, worker, etc.)?",
    "education": "What is your highest education level?",
    "marital_status": "What is your marital status?",
    "residence": "Do you live in a rural or urban area?",
    "disability": "Do you have a disability?",
    "bpl": "Do you hold a Below Poverty Line (BPL) card?",
}

STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "jammu and kashmir", "ladakh", "chandigarh",
    "puducherry", "andaman and nicobar", "lakshadweep",
]


@dataclass
class Constraint:
    attribute: str                      # one of ATTRIBUTES, or "text" for fuzzy conditions (-> NLI)
    operator: str = "=="
    value: Any = None                   # number | str | bool | [list] | [lo, hi]
    source_sentence: str = ""
    unit: Optional[str] = None
    exceptions: List[dict] = field(default_factory=list)  # [{"attribute":..,"operator":"in","value":[..]}]
    group: Optional[str] = None         # constraints sharing a group id are OR-ed; others are AND-ed

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Constraint(**{k: d[k] for k in Constraint.__dataclass_fields__ if k in d})


# ---------------- aligning with the dataset's profile / rule vocabulary ----------------
STATE_ABBR = {"an": "andaman and nicobar", "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam",
              "br": "bihar", "ch": "chandigarh", "cg": "chhattisgarh", "dl": "delhi", "ga": "goa", "gj": "gujarat",
              "hr": "haryana", "hp": "himachal pradesh", "jk": "jammu and kashmir", "jh": "jharkhand",
              "ka": "karnataka", "kl": "kerala", "la": "ladakh", "ld": "lakshadweep", "mp": "madhya pradesh",
              "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
              "od": "odisha", "or": "odisha", "py": "puducherry", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim",
              "tn": "tamil nadu", "ts": "telangana", "tg": "telangana", "tr": "tripura", "up": "uttar pradesh",
              "uk": "uttarakhand", "wb": "west bengal"}
PROFILE_ALIASES = {
    "age": "age", "income": "annual_income", "annual_income": "annual_income", "family_income": "annual_income",
    "caste": "caste_category", "category": "caste_category", "caste_category": "caste_category",
    "gender": "gender", "sex": "gender", "state": "state", "occupation": "occupation",
    "disability": "disability", "disabled": "disability", "is_disabled": "disability",
    "bpl": "bpl", "is_bpl": "bpl", "residence": "residence", "area": "residence", "location_type": "residence",
    "education": "education", "marital_status": "marital_status",
}
CASTE_ALIASES = {"scheduled caste": "sc", "scheduled castes": "sc", "scheduled tribe": "st", "scheduled tribes": "st",
                 "other backward class": "obc", "other backward classes": "obc", "gen": "general",
                 "unreserved": "general", "economically weaker section": "ews"}
GENDER_ALIASES = {"m": "male", "f": "female", "man": "male", "woman": "female", "women": "female", "girl": "female"}
ANY_VALUES = {"all", "any", "none", "null", "na", "n/a", "nan", "both", "unrestricted", "-", "", "unknown"}


def _boolish(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    s = str(v).strip().lower()
    return True if s in {"yes", "true", "y", "1"} else False if s in {"no", "false", "n", "0"} else None


def norm_state(v):
    s = str(v).strip().lower()
    return STATE_ABBR.get(s, s)


def norm_caste(v):
    s = str(v).strip().lower()
    return CASTE_ALIASES.get(s, s)


def normalise_profile(p):
    """Dataset profile (true_profile) -> engine profile. Missing / unknown values are dropped (=> 'unknown')."""
    out = {}
    for k, v in (p or {}).items():
        a = PROFILE_ALIASES.get(str(k).strip().lower())
        if a is None or v is None or (isinstance(v, str) and v.strip().lower() in ANY_VALUES):
            continue
        try:
            if a == "age": out[a] = int(float(v))
            elif a == "annual_income": out[a] = float(str(v).replace(",", "").replace("₹", ""))
            elif a in BOOLEAN:
                b = _boolish(v)
                if b is not None: out[a] = b
            elif a == "state": out[a] = norm_state(v)
            elif a == "caste_category": out[a] = norm_caste(v)
            elif a == "gender":
                g = str(v).strip().lower(); out[a] = GENDER_ALIASES.get(g, g)
            else: out[a] = str(v).strip().lower()
        except (ValueError, TypeError):
            continue
    return out
