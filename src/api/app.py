"""FastAPI orchestration aligned with the marg data.
  JANORA_DENSE=0 uvicorn src.api.app:app --reload        (BM25 only, no model download)
  JANORA_DENSE=1 JANORA_MODEL=<fine-tuned dir> ...         (tuned BGE-M3, dense-only retrieval; GPU recommended)
  JANORA_MT=nllb ...                                      (translate non-English to English first = 'pivot')
  JANORA_NATIVE=1 ...                                     (retrieve with the ORIGINAL-language query = 'native')
POST /chat {"session_id":"u1","text":"..."}   GET /profile/{sid}   POST /reset/{sid}"""
import os
from typing import Dict
from fastapi import FastAPI
from pydantic import BaseModel
from configs import settings as S
from src.data_prep import loaders as L
from src.nlu import intent as intent_mod
from src.nlu.profile_manager import ProfileManager
from src.nlu.slots import extract_slots
from src.reasoning.rag_rules import all_constraints
from src.reasoning.ranker import rank_candidates, next_question
from src.retrieval.hybrid_retriever import HybridRetriever
from src.translation.translator import Translator, NLLBBackend

DISCLAIMER = "This is guidance only - please verify on the official portal before applying."
DEFAULT_ASK = ["state", "age", "occupation", "annual_income", "caste_category", "gender"]


class Pipeline:
    def __init__(self):
        docs = L.card_frame()
        self.docs = docs.set_index("slug")
        self.states = sorted({str(s).lower() for s in docs["state"] if str(s).strip() and str(s).lower() != "all"})
        self.cons = all_constraints(L.load_rag_rules()) if S.RAG_RULES.exists() else {}
        self.details = L.load_details()
        self.name_idx = intent_mod.build_name_index(docs["name"])
        self.retriever = HybridRetriever(docs, use_dense=os.getenv("JANORA_DENSE", "1") == "1",
                                         model_name=os.getenv("JANORA_MODEL"))      # path of the fine-tuned model
        mt = NLLBBackend() if os.getenv("JANORA_MT") == "nllb" else None
        self.tr = Translator(mt, glossary=list(docs["name"]))
        self.native = os.getenv("JANORA_NATIVE") == "1"
        self.sessions: Dict[str, ProfileManager] = {}

    def link(self, slug): return S.LINK.format(slug=slug)

    def _candidates(self, query, profile, k=30):
        hits = self.retriever.search(query, k=k, mode="dense" if self.retriever.use_dense else "bm25")   # dense-only won the weight sweep; state is handled by the eligibility rules
        top = float(hits["score"].max()) if len(hits) else 1.0
        return [(r.slug, r.score / top) for r in hits.itertuples()]

    def handle(self, sid: str, text: str) -> Dict:
        pm = self.sessions.setdefault(sid, ProfileManager())
        tin = self.tr.translate_in(text); en, lang = tin["text"], tin["lang"]
        ext = extract_slots(en, self.states)
        events = pm.apply(ext)
        profile = events.get("whatif_profile") if events["hypothetical"] else pm.values()
        n_core = sum(a in profile for a in intent_mod.CORE)
        intent = intent_mod.predict(en, n_core, self.name_idx)
        if events["hypothetical"] and ext["slots"]: intent = "standard"        # what-if => re-rank on temp profile
        resp: Dict = {"intent": intent, "language": lang, "asker": ext["asker"], "profile_events": events}
        q = text if self.native else en

        if intent == "named":
            top = self.retriever.search(q, k=1).iloc[0]; d = self.details.get(top.slug, {})
            resp["scheme"] = {"slug": top.slug, "name": top["name"], "link": self.link(top.slug)}
            msg = f"{top['name']}: " + (d.get("eligibility") or top.text)[:500]
            if top.slug in self.cons:
                ev = rank_candidates(profile, [(top.slug, 1.0)], self.cons)
                resp["fit"] = (ev["ranked"] or [{"bucket": "not_eligible", "details": []}])[0] if ev["ranked"] else {"bucket": "not_eligible"}
        elif intent == "offtopic":
            msg = "I can help with Indian government schemes - tell me about yourself or ask about a scheme."
        elif intent == "vague":
            cands = self._candidates(q, profile)
            res = rank_candidates(profile, cands, self.cons)
            nq = res["next_question"] or next((  # fall back to the first core attribute still missing
                {"attribute": a, "question": __import__("configs.constraint_schema", fromlist=["x"]).ATTRIBUTE_QUESTIONS[a]}
                for a in DEFAULT_ASK if a not in profile), None)
            resp["next_question"] = nq
            msg = "I need a little more detail. " + (nq["question"] if nq else "Could you tell me more about yourself?")
        else:                                                                       # standard
            cands = self._candidates(q, profile)
            res = rank_candidates(profile, cands, self.cons)
            resp["eligibility"] = {"excluded_as_ineligible": res["n_excluded"], "next_question": res["next_question"],
                "top": [{"slug": r["slug"], "name": self.docs.loc[r["slug"], "name"], "bucket": r["bucket"], "score": r["score"],
                         "details": r["details"], "missing": r["missing"], "link": self.link(r["slug"])} for r in res["ranked"]]}
            b = [r["bucket"] for r in res["ranked"]]
            msg = (f"Top matches: {b.count('likely_eligible')} look likely, {b.count('needs_more_info')} need more info, "
                   f"{b.count('no_rule')} have no structured rules yet (check the official page). "
                   f"{res['n_excluded']} candidates were ruled out for your profile.")
            if res["next_question"]: msg += " " + res["next_question"]["question"]
        resp["answer_en"] = msg
        resp["answer"] = self.tr.translate_out(msg, "hi" if lang == "hi-Latn" else lang)
        resp.update(disclaimer=DISCLAIMER, profile=pm.values())
        return resp


class ChatIn(BaseModel):
    session_id: str
    text: str


app = FastAPI(title="JANORA")
_pipe = None


def pipe() -> Pipeline:
    global _pipe
    if _pipe is None: _pipe = Pipeline()
    return _pipe


@app.post("/chat")
def chat(body: ChatIn): return pipe().handle(body.session_id, body.text)


@app.get("/profile/{session_id}")
def profile(session_id: str): return pipe().sessions.get(session_id, ProfileManager()).snapshot()


@app.post("/reset/{session_id}")
def reset(session_id: str):
    pipe().sessions.pop(session_id, None); return {"ok": True}