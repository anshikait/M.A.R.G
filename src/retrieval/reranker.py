"""Cross-encoder reranker (default BAAI/bge-reranker-v2-m3): scores (query, scheme card) pairs jointly."""
import numpy as np


def score_pairs(model, tok, queries, texts, device, max_len=320, bs=64):
    import torch
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(queries), bs):
            enc = tok(queries[i:i + bs], texts[i:i + bs], truncation="only_second", max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                out.append(model(**enc).logits.float().squeeze(-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


class CrossEncoderReranker:
    def __init__(self, path, device=None, max_len=320):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.max_len = max_len

    def score(self, queries, texts, bs=64):
        return score_pairs(self.model, self.tok, queries, texts, self.device, self.max_len, bs)

    def rerank(self, query, texts):
        """Returns candidate indices sorted best-first, and their scores."""
        s = self.score([query] * len(texts), list(texts))
        order = np.argsort(-s)
        return order, s[order]


def rerank_orders(orders, texts_by_row, queries, scorer, pool=30):
    """orders: per-query dense ranking (row indices). Rerank the first `pool`, keep the tail as is.
    scorer(list_of_queries, list_of_texts) -> np.array of scores.  Returns new orders (same length as input)."""
    pairs_q, pairs_t, spans = [], [], []
    for q, order in zip(queries, orders):
        cand = list(order[:pool]); spans.append(len(cand))
        pairs_q += [q] * len(cand); pairs_t += [texts_by_row[i] for i in cand]
    scores = np.asarray(scorer(pairs_q, pairs_t)); out, k = [], 0
    for order, n in zip(orders, spans):
        cand = list(order[:n]); sc = scores[k:k + n]; k += n
        out.append([cand[j] for j in np.argsort(-sc, kind="stable")] + list(order[n:]))
    return out