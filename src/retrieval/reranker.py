"""Step 5b: fine-tune the cross-encoder reranker on top of the TUNED retriever's candidates.

  python -m src.retrieval.finetune_reranker --retriever /content/drive/MyDrive/janora_models/bge_m3_janora \
         --out /content/drive/MyDrive/janora_models/bge_reranker_janora --freeze-layers 8
  python -m eval.evaluate_rerank --split val --index all --retriever <bi-encoder dir> --reranker <that out dir>

Listwise loss: for each query a group of 1 positive (target scheme) + (group-1) hard negatives taken from the retriever's
top-pool for TRAIN queries over TRAIN cards only; softmax cross-entropy over the group. By default negatives skip every scheme
the person is eligible for (relevant_slugs / eligible_same_category_slugs) so valid answers are not pushed down;
--keep-relevant-as-negatives teaches 'prefer the exact target' (ablate it: it can raise exact-target recall but lower relevance).
"""
import argparse, math, os, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
from configs import settings as S
from src.data_prep import loaders as L
from src.retrieval.finetune_retriever import build_training_rows, enable_checkpointing, freeze_lower


def top_pool(Q, D, k):
    sims = Q @ D.T
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    return [row[np.argsort(-sims[i, row])] for i, row in enumerate(idx)]


def make_groups(q, cand_slugs, train_slugs, group, exclude_relevant, rng, take_from=15):
    """One list per query: [positive, neg1, ... neg(group-1)]. cand_slugs[i] = retriever top-pool (best first)."""
    groups = []
    for r, cands in zip(q.itertuples(), cand_slugs):
        banned = {r.slug} | (set(r.relevant) | set(r.elig_same) if exclude_relevant else set())
        negs = [s for s in cands if s not in banned][:take_from]
        rng.shuffle(negs); negs = negs[:group - 1]
        while len(negs) < group - 1:
            s = rng.choice(train_slugs)
            if s not in banned and s not in negs: negs.append(s)
        groups.append([r.slug] + negs)
    return groups


def main():
    import torch, torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.retrieval.reranker import score_pairs
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=S.RERANK_MODEL)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--qps", type=int, default=4, help="queries per optimizer step (pairs per step = qps x group)")
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--pool", type=int, default=30); ap.add_argument("--val-n", type=int, default=300)
    ap.add_argument("--freeze-layers", type=int, default=8); ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--keep-relevant-as-negatives", action="store_true"); ap.add_argument("--seed", type=int, default=S.SEED)
    a = ap.parse_args()
    rng = random.Random(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    docs = L.card_frame(); text = dict(zip(docs.slug, docs.text))
    dtr, dva = docs[docs.split == "train"].reset_index(drop=True), docs[docs.split == "val"].reset_index(drop=True)
    qtr = build_training_rows(L.load_queries("train"), set(dtr.slug))
    qva = L.load_queries("val"); qva = qva[qva.slug.notna() & qva.slug.isin(dva.slug) & (qva.row_type != "vague")]
    qva = qva.sample(min(a.val_n, len(qva)), random_state=a.seed).reset_index(drop=True)
    print(f"train queries={len(qtr)} | val subset={len(qva)} | device={dev}")

    # candidates from the TUNED retriever (then freed from GPU before the reranker loads)
    st = SentenceTransformer(a.retriever, device=dev); st.max_seq_length = 256
    enc = lambda t: st.encode(t, batch_size=64, normalize_embeddings=True)
    Dtr, Dva = enc(dtr.text.tolist()), enc(dva.text.tolist())
    tr_top = top_pool(enc(qtr["query"].tolist()), Dtr, a.pool)
    va_top = top_pool(enc(qva["query"].tolist()), Dva, a.pool)
    del st; torch.cuda.empty_cache()
    tr_slugs = [[dtr.slug[j] for j in row] for row in tr_top]
    groups = make_groups(qtr, tr_slugs, dtr.slug.tolist(), a.group, not a.keep_relevant_as_negatives, rng)
    va_cands = [[dva.slug[j] for j in row] for row in va_top]
    va_gold = qva["slug"].tolist()
    base_r3 = float(np.mean([g in c[:3] for g, c in zip(va_gold, va_cands)]))
    print(f"retriever-only R@3 on this val subset (val-split index): {base_r3:.3f}  | pool recall: {np.mean([g in c for g, c in zip(va_gold, va_cands)]):.3f}")

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base).to(dev)
    if not a.no_grad_ckpt: print("gradient checkpointing:", "on" if enable_checkpointing(model) else "NOT available")
    print("frozen lower layers:", freeze_lower(model, a.freeze_layers))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.0f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(groups) / a.qps); warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    def val_r3():
        qs, ts, spans = [], [], []
        for q, c in zip(qva["query"], va_cands): qs += [q] * len(c); ts += [text[s] for s in c]; spans.append(len(c))
        sc = score_pairs(model, tok, qs, ts, dev, a.max_len, 64); k, hits = 0, []
        for g, c, n in zip(va_gold, va_cands, spans):
            order = np.argsort(-sc[k:k + n]); k += n; hits.append(g in [c[j] for j in order[:3]])
        return float(np.mean(hits))

    best, step, queries = -1.0, 0, qtr["query"].tolist()
    for ep in range(a.epochs):
        model.train(); order = list(range(len(groups))); rng.shuffle(order); run = 0.0
        for b in range(0, len(order), a.qps):
            ids = order[b:b + a.qps]
            qs = [queries[i] for i in ids for _ in range(a.group)]
            ts = [text[s] for i in ids for s in groups[i]]
            enc_b = tok(qs, ts, truncation="only_second", max_length=a.max_len, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(**enc_b).logits.float().view(len(ids), a.group)
            loss = F.cross_entropy(logits, torch.zeros(len(ids), dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1; run += loss.item()
            if step % 100 == 0: print(f"  ep{ep + 1} step {step}/{steps} loss={run / 100:.4f}"); run = 0.0
        r3 = val_r3(); print(f"epoch {ep + 1}: reranked val R@3={r3:.3f}  (retriever only {base_r3:.3f})")
        if r3 > best:
            best = r3; os.makedirs(a.out, exist_ok=True); model.save_pretrained(a.out); tok.save_pretrained(a.out); print("  saved ->", a.out)
    print("best reranked val R@3:", round(best, 4))


if __name__ == "__main__":
    main()