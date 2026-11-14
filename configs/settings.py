"""Paths and field names aligned with the marg repo layout (all joined by scheme `slug`).
If `python -m src.data_prep.inspect_data` shows different field names, add them to FIELDS below."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REGISTRY_CSV = DATA / "registry" / "scheme_registry.csv"
SPLIT_JSON = DATA / "registry" / "scheme_split_v1.json"
CARDS = DATA / "cards" / "scheme_cards.jsonl"
QUERY_FILES = {s: DATA / "splits" / f"{s}.jsonl" for s in ("train", "val", "test")}
DETAILS = DATA / "en" / "schemes_detail.jsonl"
RAG_RULES = DATA / "rag_schemes.json"
HI_LIST = DATA / "hi" / "schemes_list.jsonl"
CACHE = DATA / "cache"
RESULTS = ROOT / "eval" / "results"

EXPECTED_SPLIT_HASH = "ae21571e152abac9"       # from your plan (algorithm unknown; we try sha256/sha1/md5 prefixes)
SEED = 42
DENSE_MODEL = "BAAI/bge-m3"                     # multilingual (plan step 4). Use BAAI/bge-small-en-v1.5 for quick CPU tests
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LINK = "https://www.myscheme.gov.in/schemes/{slug}"

# candidate field names, first match wins (case-insensitive)
FIELDS = {
    "slug":      ["slug", "scheme_slug", "scheme_id", "id"],
    "name":      ["name", "scheme_name", "title"],
    "card":      ["card", "text", "card_text", "content", "document"],
    "query":     ["query", "question", "text", "utterance", "message", "user_query"],
    "target":    ["target_slug", "target_scheme", "scheme_slug", "slug", "target"],
    "language":  ["language", "lang"],
    "noise":     ["noise_level", "noise", "noise_tag"],
    "row_type":  ["row_type", "type", "query_type"],
    "asker":     ["asker", "asked_by", "who"],
    "profile":   ["true_profile", "profile"],
    "relevant":  ["relevant_slugs", "relevant"],
    "elig_same": ["eligible_same_category_slugs", "eligible_slugs"],
    "missing":   ["missing_fields", "missing"],
    "turns":     ["chat_turns", "turns", "messages", "conversation"],
    "eligibility": ["eligibility", "eligibility_text", "eligibility_criteria"],
    "benefits":  ["benefits", "benefit"],
    "application": ["application", "application_process", "how_to_apply"],
    "rule":      ["rule", "rules", "eligibility_rule", "constraints"],
}
