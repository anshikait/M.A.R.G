"""Language ID + number normalisation + glossary masking + pluggable translation backends.

Pivot design: translate_in() -> English pipeline -> translate_out().
Backends: "identity" (testing), "nllb" (facebook/nllb-200-distilled-600M via transformers).
Swap in IndicTrans2 by adding a backend class with .translate(text, src, tgt).
Romanised Hindi (Hinglish) is detected as "hi-Latn"; transliterate to Devanagari first (e.g. IndicXlit) - TODO.
"""
import re
from typing import Dict, List, Optional, Tuple

SCRIPTS = {"hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "pa": (0x0A00, 0x0A7F), "gu": (0x0A80, 0x0AFF),
           "or": (0x0B00, 0x0B7F), "ta": (0x0B80, 0x0BFF), "te": (0x0C00, 0x0C7F), "kn": (0x0C80, 0x0CFF),
           "ml": (0x0D00, 0x0D7F)}
NLLB = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "pa": "pan_Guru", "gu": "guj_Gujr",
        "or": "ory_Orya", "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda", "ml": "mal_Mlym",
        "hi-Latn": "hin_Deva"}
HINGLISH = set("mera meri mujhe mujhko hai hain nahi nahin kya kaise kaun kitna kitni ke ki ka ko se me mein "
               "aur yojana paisa sarkar chahiye milega milta hu hoon ghar kisan beti pension bhi par".split())


def detect_language(text: str) -> str:
    counts = {l: 0 for l in SCRIPTS}
    for ch in text:
        o = ord(ch)
        for l, (a, b) in SCRIPTS.items():
            if a <= o <= b: counts[l] += 1
    best = max(counts, key=counts.get)
    if counts[best] >= 2: return best
    toks = re.findall(r"[a-z]+", text.lower())
    if toks and sum(t in HINGLISH for t in toks) >= 2 and sum(t in HINGLISH for t in toks) / len(toks) >= 0.2:
        return "hi-Latn"
    return "en"


# ---------------- number normalisation ("ढाई लाख", "2.5 lakh", Devanagari digits) ----------------
_DEV = str.maketrans("०१२३४५६७८९", "0123456789")
_WORDS = {"एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8,
          "नौ": 9, "दस": 10, "ढाई": 2.5, "डेढ़": 1.5, "डेढ": 1.5, "सवा": 1.25}
_UNITS = {"लाख": 1e5, "हज़ार": 1e3, "हजार": 1e3, "करोड़": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5,
          "lacs": 1e5, "crore": 1e7, "crores": 1e7, "thousand": 1e3, "hazaar": 1e3, "hazar": 1e3, "hajar": 1e3}


_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                                     "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_NW = r"(?:(?:%s)(?:[- ](?:%s))?|%s)" % ("|".join(_TENS), "|".join(list(_ONES)[1:10]), "|".join(_ONES))


def _w2n(w: str) -> int:
    parts = re.split(r"[- ]", w.lower()); return sum(_TENS.get(p, _ONES.get(p, 0)) for p in parts)


def words_to_digits(text: str) -> str:
    """'just turned fifty three' -> 'just turned 53' (only in age-like contexts, to avoid touching 'one scheme')."""
    t = re.sub(r"\b(turned|aged?|am|is)\s+(%s)\b" % _NW, lambda m: f"{m[1]} {_w2n(m[2])}", text, flags=re.I)
    return re.sub(r"\b(%s)\s*(years?|yrs?)\b" % _NW, lambda m: f"{_w2n(m[1])} {m[2]}", t, flags=re.I)


def normalise_numbers(text: str) -> str:
    text = words_to_digits(text)
    t = text.translate(_DEV)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)            # Indian digit grouping: 1,50,000 -> 150000
    pat = r"(\d+(?:\.\d+)?|" + "|".join(map(re.escape, _WORDS)) + r")\s*(" + "|".join(map(re.escape, _UNITS)) + r")"
    def rep(m):
        n = _WORDS.get(m[1]) if m[1] in _WORDS else float(m[1])
        return str(int(n * _UNITS[m[2].lower()]))
    return re.sub(pat, rep, t, flags=re.I)


# ---------------- glossary masking (protect scheme names from the MT model) ----------------
def mask_terms(text: str, terms: List[str]) -> Tuple[str, Dict[str, str]]:
    mapping = {}
    for i, term in enumerate(sorted(set(terms), key=len, reverse=True)):
        if len(term) < 4: continue
        pat = re.compile(re.escape(term), re.I)
        if pat.search(text):
            tok = f"XSCHEMEX{i}X"
            text = pat.sub(tok, text); mapping[tok] = term
    return text, mapping


def unmask_terms(text: str, mapping: Dict[str, str]) -> str:
    for tok, term in mapping.items():
        text = text.replace(tok, term)
    return text


class IdentityBackend:
    def translate(self, text, src, tgt): return text


class NLLBBackend:
    def __init__(self, model="facebook/nllb-200-distilled-600M"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)

    def translate(self, text, src, tgt):
        self.tok.src_lang = NLLB[src]
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=400)
        out = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(NLLB[tgt]),
                                  max_new_tokens=400, num_beams=4)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0]


class Translator:
    def __init__(self, backend=None, glossary: Optional[List[str]] = None):
        self.backend = backend or IdentityBackend()
        self.glossary = glossary or []

    def translate_in(self, text: str) -> Dict:
        lang = detect_language(text)
        t = normalise_numbers(text)
        if lang == "en":
            return {"text": t, "lang": "en", "original": text}
        masked, m = mask_terms(t, self.glossary)
        en = unmask_terms(self.backend.translate(masked, lang, "en"), m)
        return {"text": normalise_numbers(en), "lang": lang, "original": text}

    def translate_out(self, text: str, lang: str) -> str:
        if lang == "en": return text
        masked, m = mask_terms(text, self.glossary)
        return unmask_terms(self.backend.translate(masked, "en", lang), m)
