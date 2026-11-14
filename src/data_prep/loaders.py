"""Readers for the marg data layout. Everything is joined on `slug`. Field names are matched tolerantly
via configs.settings.FIELDS, so a renamed column only needs one line added there."""
import ast, hashlib, html, json, re
from pathlib import Path
from typing import Dict, List
import pandas as pd
from configs import settings as S
from configs.constraint_schema import normalise_profile


def pick(rec: dict, key: str, default=None):
    low = {str(k).lower(): v for k, v in rec.items()}
    for cand in S.FIELDS[key]:
        if cand.lower() in low and low[cand.lower()] is not None:
            return low[cand.lower()]
    return default


def read_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_any(path):
    p = Path(path)
    if p.suffix == ".jsonl": return read_jsonl(p)
    if p.suffix == ".json": return json.load(open(p, encoding="utf-8"))
    if p.suffix == ".csv": return pd.read_csv(p).to_dict("records")
    raise ValueError(p)


def _as_list(v):
    if v is None or (isinstance(v, float) and pd.isna(v)): return []
    if isinstance(v, list): return v
    if isinstance(v, str):
        v = v.strip()
        if not v: return []
        for fn in (json.loads, ast.literal_eval):
            try:
                r = fn(v); return r if isinstance(r, list) else [r]
            except Exception: pass
        return [x.strip() for x in v.split(",") if x.strip()]
    return [v]


def _as_dict(v):
    if isinstance(v, dict): return v
    if isinstance(v, str):
        for fn in (json.loads, ast.literal_eval):
            try:
                r = fn(v); return r if isinstance(r, dict) else {}
            except Exception: pass
    return {}


# ---------------- registry + frozen split ----------------
def load_registry() -> pd.DataFrame:
    df = pd.read_csv(S.REGISTRY_CSV)
    df.columns = [c.strip() for c in df.columns]
    slug = next(c for c in df.columns if c.lower() in [x.lower() for x in S.FIELDS["slug"]])
    name = next((c for c in df.columns if c.lower() in [x.lower() for x in S.FIELDS["name"]]), None)
    df = df.rename(columns={slug: "slug", **({name: "name"} if name else {})})
    for c in ("name", "level", "state", "category", "tags", "split"):
        if c not in df.columns: df[c] = ""
    return df.fillna("").drop_duplicates("slug").reset_index(drop=True)


def load_split_map() -> Dict[str, str]:
    """slug -> train|val|test. Accepts {slug: split}, {split: [slugs]}, or either wrapped in a metadata dict."""
    obj = json.load(open(S.SPLIT_JSON, encoding="utf-8"))
    for _ in range(3):
        if isinstance(obj, dict):
            for key in ("splits", "split", "assignments", "assignment", "schemes", "mapping", "data"):
                if key in obj and isinstance(obj[key], (dict, list)):
                    obj = obj[key]; break
    if isinstance(obj, list):
        return {pick(r, "slug"): str(r.get("split")) for r in obj}
    names = {"train", "val", "validation", "test"}
    if set(map(str.lower, obj.keys())) & names and all(isinstance(v, list) for v in obj.values()):
        return {slug: ("val" if k.lower() == "validation" else k.lower()) for k, v in obj.items() for slug in v}
    return {k: ("val" if str(v).lower() == "validation" else str(v).lower()) for k, v in obj.items()}


def split_file_hash_matches() -> str:
    b = Path(S.SPLIT_JSON).read_bytes()
    for name in ("sha256", "sha1", "md5"):
        if getattr(hashlib, name)(b).hexdigest().startswith(S.EXPECTED_SPLIT_HASH):
            return f"hash OK ({name} prefix {S.EXPECTED_SPLIT_HASH})"
    return "hash prefix not matched by sha256/sha1/md5 - fine if you hash canonical JSON; otherwise check the file wasn't edited"


# ---------------- cards, queries, details, rules ----------------
def clean_text(t) -> str:
    """Fix HTML entities seen in the scraped data: '&#39;' and the broken form '& 39;'."""
    t = html.unescape(str(t))
    return re.sub(r"&\s*39;", "'", re.sub(r"&\s*amp;", "&", t))


def load_cards() -> pd.DataFrame:
    rows = []
    for r in read_jsonl(S.CARDS):
        text = pick(r, "card")
        if text is None:
            text = " | ".join(str(v) for k, v in r.items() if isinstance(v, str) and k.lower() not in ("slug",))
        rows.append({"slug": pick(r, "slug"), "card_name": pick(r, "name", ""), "text": clean_text(text)})
    return pd.DataFrame(rows).drop_duplicates("slug")


def card_frame() -> pd.DataFrame:
    """Registry + cards: one row per scheme -> slug,name,level,state,category,split,text (the search index)."""
    reg, cards = load_registry(), load_cards()
    df = reg.merge(cards, on="slug", how="inner")
    sm = load_split_map()
    df["split"] = df["slug"].map(sm).fillna(df["split"])
    df["name"] = df["name"].where(df["name"].astype(str) != "", df["card_name"])
    return df.drop(columns=["card_name"]).reset_index(drop=True)


def load_queries(split: str) -> pd.DataFrame:
    rows = []
    for r in read_jsonl(S.QUERY_FILES[split]):
        turns = _as_list(pick(r, "turns"))
        q = pick(r, "query")
        if q is None and turns:
            q = " ".join(t.get("content", t.get("text", "")) if isinstance(t, dict) else str(t)
                         for t in turns if not isinstance(t, dict) or t.get("role", "user") == "user")
        rt = str(pick(r, "row_type", "standard")).strip().lower().replace("-", "_").replace(" ", "_")
        tgt = pick(r, "target")
        rows.append({
            "query": q or "", "slug": tgt if isinstance(tgt, str) and tgt else None,
            "language": str(pick(r, "language", "en")), "noise": str(pick(r, "noise", "clean")).lower(),
            "category": r.get("category"), "row_type": rt, "asker": str(pick(r, "asker", "self")).lower(),
            "profile": normalise_profile(_as_dict(pick(r, "profile"))),
            "relevant": _as_list(pick(r, "relevant")), "elig_same": _as_list(pick(r, "elig_same")),
            "missing": _as_list(pick(r, "missing")), "turns": turns})
    return pd.DataFrame(rows)


def _rich(o) -> str:
    """Flatten the site's rich-text nodes ({'type':..,'children':[{'text':..}]}) into plain text."""
    if o is None: return ""
    if isinstance(o, str): return o
    if isinstance(o, list): return "\n".join(t for t in (_rich(x) for x in o) if t)
    if isinstance(o, dict):
        if "text" in o and "children" not in o: return str(o["text"])
        return _rich(o.get("children") or o.get("process") or "")
    return str(o)


def load_details() -> Dict[str, dict]:
    """slug -> {name, description, benefits, eligibility, exclusions, application, links}. Plain text, English."""
    if not S.DETAILS.exists(): return {}
    out = {}
    for r in read_jsonl(S.DETAILS):
        en = r.get("en", r)
        sc, ec = en.get("schemeContent", {}) or {}, en.get("eligibilityCriteria", {}) or {}
        out[pick(r, "slug")] = {
            "name": (en.get("basicDetails", {}) or {}).get("schemeName", ""),
            "description": clean_text(sc.get("detailedDescription_md") or sc.get("briefDescription", "")),
            "benefits": clean_text(sc.get("benefits_md") or _rich(sc.get("benefits"))),
            "eligibility": clean_text(ec.get("eligibilityDescription_md") or _rich(ec.get("eligibilityDescription"))),
            "exclusions": clean_text(sc.get("exclusions_md", "")),
            "application": clean_text(_rich(en.get("applicationProcess"))),
            "links": [x.get("url") for x in (sc.get("references") or []) if isinstance(x, dict)]}
    return out


def load_rag_meta() -> Dict[str, dict]:
    """slug -> {name, category, state, level, topics, url} for the 1,082 rule-bearing schemes."""
    obj = json.load(open(S.RAG_RULES, encoding="utf-8"))
    return {pick(r, "slug"): {k: r.get(k) for k in ("name", "category", "state", "level", "topics", "url", "description")}
            for r in (obj if isinstance(obj, list) else obj.get("schemes", []))}


def load_rag_rules() -> Dict[str, dict]:
    obj = json.load(open(S.RAG_RULES, encoding="utf-8"))
    items = obj if isinstance(obj, list) else [dict(v, slug=k) if isinstance(v, dict) else {"slug": k} for k, v in obj.items()] \
        if all(isinstance(v, dict) for v in obj.values()) else obj.get("schemes", [])
    out = {}
    for r in items:
        slug = pick(r, "slug")
        rule = _as_dict(pick(r, "rule")) or {k: v for k, v in r.items() if k not in ("slug", "name", "scheme_name", "title")}
        if slug: out[slug] = rule
    return out


# ---------------- integrity ----------------
def integrity_report() -> Dict:
    reg, cards, sm = load_registry(), load_cards(), load_split_map()
    rep = {"registry": len(reg), "cards": len(cards), "split_map": len(sm), "hash": split_file_hash_matches(),
           "split_counts": pd.Series(sm).value_counts().to_dict(),
           "registry_not_in_split": len(set(reg.slug) - set(sm)), "cards_not_in_registry": len(set(cards.slug) - set(reg.slug))}
    leaks = {}
    for sp in ("train", "val", "test"):
        if not S.QUERY_FILES[sp].exists(): continue
        q = load_queries(sp)
        mapped = q.slug.map(sm)
        bad = q[q.slug.notna() & mapped.notna() & (mapped != sp)]
        leaks[sp] = {"queries": len(q), "target_in_wrong_split": len(bad),
                     "target_not_in_split_map": int((q.slug.notna() & mapped.isna()).sum()), "no_target(offtopic)": int(q.slug.isna().sum())}
    rep["query_leaks"] = leaks
    return rep
