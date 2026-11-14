"""Row-type classifier (your labels): standard | named | vague | offtopic.
  standard -> describes a situation -> retrieve + eligibility      named   -> direct scheme lookup
  vague    -> too little detail      -> ask clarifying question     offtopic-> polite refusal
Baselines: rules (below) and TF-IDF + LogisticRegression trained on data/splits/train.jsonl. Later: DeBERTa / XLM-R.
"""
import pickle, re
from pathlib import Path
from typing import Iterable, Optional
from configs import settings as S

INTENTS = ["standard", "named", "vague", "offtopic"]
MODEL_PATH = S.CACHE / "intent_tfidf_lr.pkl"
CORE = ["age", "gender", "state", "caste_category", "annual_income", "residence"]
_SCHEMEISH = r"scheme|yojana|pension|scholarship|loan|subsidy|benefit|insurance|assistance|grant|eligible|apply|sahayata|योजना|छात्रवृत्ति|पेंशन"


def _norm(s): return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


def build_name_index(names: Iterable[str]):
    return {" ".join(_norm(n)) for n in names if len(" ".join(_norm(n))) >= 12}


def rule_intent(text: str, n_core_slots: int, name_index: Optional[set] = None) -> str:
    low_tokens = " ".join(_norm(text))
    if name_index and any(n in low_tokens for n in name_index): return "named"
    if not re.search(_SCHEMEISH, text.lower()) and n_core_slots == 0: return "offtopic"
    return "vague" if n_core_slots < 4 else "standard"


def train(texts, labels, out=MODEL_PATH):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
                         LogisticRegression(max_iter=2000, C=5, class_weight="balanced"))
    pipe.fit(texts, labels)
    Path(out).parent.mkdir(parents=True, exist_ok=True); pickle.dump(pipe, open(out, "wb"))
    return pipe


def predict(text: str, n_core_slots: int = 0, name_index=None) -> str:
    """Precise rule guards first (a full scheme name => named; no scheme words and no details => offtopic),
    then the trained classifier for standard-vs-vague, falling back to rules if no model is trained yet."""
    r = rule_intent(text, n_core_slots, name_index)
    if r in ("named", "offtopic"): return r
    if MODEL_PATH.exists():
        return pickle.load(open(MODEL_PATH, "rb")).predict([text])[0]
    return r
