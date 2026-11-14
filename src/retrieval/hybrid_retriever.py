"""BM25 + dense (BGE-M3 by default) + Reciprocal Rank Fusion over scheme cards (one card per scheme).
docs = loaders.card_frame()  (columns: slug,name,state,level,category,split,text)."""
import hashlib, re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from configs import settings as S

_TOK = re.compile(r"[\w\u0900-\u0DFF]+")        # \w alone splits Indic words at vowel signs; this keeps them whole
tok = lambda t: _TOK.findall(str(t).lower())


class HybridRetriever:
    def __init__(self, docs: pd.DataFrame, use_dense=True, model_name=None, batch_size=32):
        self.df = docs.reset_index(drop=True)
        self.bm25 = BM25Okapi([tok(t) for t in self.df["text"]])
        self.use_dense, self.model_name = use_dense, model_name or S.DENSE_MODEL
        if use_dense:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            key = hashlib.md5(("".join(self.df.slug) + "".join(self.df.text)).encode()).hexdigest()[:10]
            path = S.CACHE / f"emb_{self.model_name.replace('/', '_')}_{key}.npy"
            if path.exists(): self.emb = np.load(path)
            else:
                S.CACHE.mkdir(parents=True, exist_ok=True)
                self.emb = self.model.encode(self.df["text"].tolist(), batch_size=batch_size,
                                             normalize_embeddings=True, show_progress_bar=True)
                np.save(path, self.emb)

    # ---- ranked lists of row indices, one per query
    def _bm25(self, queries, mask, pool):
        out = []
        for qi, q in enumerate(queries):
            s = self.bm25.get_scores(tok(q))
            if mask is not None: s = np.where(mask[qi] if mask.ndim == 2 else mask, s, -np.inf)
            idx = np.argsort(-s)[:pool]
            out.append([i for i in idx if np.isfinite(s[i]) and s[i] > 0])
        return out

    def _dense(self, queries, mask, pool, batch=64):
        Q = self.model.encode(list(queries), batch_size=batch, normalize_embeddings=True)
        S_ = Q @ self.emb.T
        if mask is not None: S_ = np.where(mask if mask.ndim == 2 else mask[None, :], S_, -np.inf)
        return [list(np.argsort(-row)[:pool]) for row in S_]

    def rank_lists(self, queries, mode="hybrid", pool=100, rrf_k=60, mask=None):
        lists = []
        if mode in ("bm25", "hybrid"): lists.append(self._bm25(queries, mask, pool))
        if mode in ("dense", "hybrid") and self.use_dense: lists.append(self._dense(queries, mask, pool))
        out = []
        for per_q in zip(*lists):
            fused = {}
            for lst in per_q:
                for r, i in enumerate(lst): fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
            out.append(sorted(fused.items(), key=lambda kv: -kv[1]))
        return out                                   # [[(row_idx, score), ...], ...]

    def state_masks(self, states):
        """(n_queries x n_docs) bool: keep central/all-India schemes and schemes of the user's state.
        Queries with unknown state (None) keep every scheme."""
        st = self.df["state"].astype(str).str.lower().values
        open_ = (self.df["level"].astype(str).str.lower() == "central").values | np.isin(st, ["all", "", "nan"])
        return np.stack([np.ones(len(self.df), bool) if not s else open_ | (st == str(s).lower()) for s in states])

    def mask_for(self, state=None, category=None):
        m = np.ones(len(self.df), dtype=bool)
        if state: m &= self.df["state"].astype(str).str.lower().isin(["all", "", "nan", str(state).lower()]).values
        if category: m &= (self.df["category"].astype(str) == category).values
        return m

    def search(self, query, k=5, mode="hybrid", state=None, category=None):
        mask = self.mask_for(state, category) if (state or category) else None
        res = self.rank_lists([query], mode, mask=mask)[0][:k]
        rows = self.df.iloc[[i for i, _ in res]].copy()
        rows["score"] = [s for _, s in res]
        return rows
