"""Grounded Q&A: retrieve -> (optional LLM) -> answer with citations, or abstain.
`llm` is any callable prompt:str -> str (OpenAI/Claude/local LoRA model). Without an LLM we return the best
extractive sentence, which doubles as the 'extractive baseline' for your comparison table.
"""
import re
from typing import Callable, Dict, Optional

STOP = set("the a an of to for in on and or is are what who how which can i my me do does be by with from at as it this that".split())
_tok = lambda t: [w for w in re.findall(r"\w+", t.lower()) if w not in STOP and len(w) > 2]

PROMPT = """You are JANORA, an assistant for Indian government schemes.
Answer ONLY using the CONTEXT. If the context does not contain the answer, reply exactly: NOT_FOUND.
Cite sources as [1], [2] after the facts they support. Be concise.

CONTEXT:
{context}

USER PROFILE: {profile}
QUESTION: {question}
ANSWER:"""


def _overlap(question, hits) -> float:
    q = set(_tok(question))
    if not q: return 0.0
    txt = set(_tok(" ".join(hits["text"].tolist())))
    return len(q & txt) / len(q)


def answer(question: str, retriever, llm: Optional[Callable[[str], str]] = None, profile: Optional[Dict] = None,
           k=4, min_overlap=0.25, **filters) -> Dict:
    hits = retriever.search(question, k=k, **filters)
    evidence = [{"n": i + 1, "scheme_id": r.scheme_id, "scheme": r.scheme_name, "field": r.field, "text": r.text}
                for i, r in enumerate(hits.itertuples())]
    if hits.empty or _overlap(question, hits) < min_overlap:
        return {"answer": "I could not find this in the scheme information I have.", "abstained": True, "evidence": evidence}
    if llm is None:                                   # extractive fallback
        q = set(_tok(question))
        best = max((s for e in evidence for s in re.split(r"(?<=[.!?])\s+", e["text"])),
                   key=lambda s: len(q & set(_tok(s))), default="")
        return {"answer": best, "abstained": False, "evidence": evidence, "mode": "extractive"}
    ctx = "\n".join(f"[{e['n']}] ({e['scheme']} / {e['field']}) {e['text']}" for e in evidence)
    out = llm(PROMPT.format(context=ctx, profile=profile or "not provided", question=question)).strip()
    if "NOT_FOUND" in out:
        return {"answer": "I could not find this in the scheme information I have.", "abstained": True, "evidence": evidence}
    return {"answer": out, "abstained": False, "evidence": evidence, "mode": "generative"}
