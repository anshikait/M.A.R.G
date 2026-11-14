"""Step 5a: fine-tune the dense retriever (BGE-M3) on train queries -> train-scheme cards, MNRL + mined hard negatives.

  python -m src.retrieval.finetune_retriever --out /content/drive/MyDrive/janora_models/bge_m3_janora
  python -m eval.evaluate_retrieval --split val --modes dense hybrid --index split --model <that dir>

Leak rules enforced here: only TRAIN-split queries and TRAIN-split scheme cards are used (val/test card text is never seen).
Negatives are mined from the base model's own top results, skipping the target AND (default) every scheme the person is
eligible for (relevant_slugs / eligible_same_category_slugs), because those are valid answers and would be false negatives.
Memory: the 250k-row word-embedding matrix is frozen (saves ~3 GB of optimizer state; fits a T4). If still out of memory: --freeze-layers 8, then --batch 8.
"""
import argparse, math, os, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less fragmentation on small GPUs
import numpy as np
from configs import settings as S
from src.data_prep import loaders as L


def build_training_rows(q, train_slugs, include_vague=False):
    q = q[q.slug.notna() & q.slug.isin(train_slugs)]
    if not include_vague: q = q[q.row_type != "vague"]
    return q.reset_index(drop=True)


def pick_negatives(sims, q, slugs, exclude_relevant=True, pool=10, seed=S.SEED):
    """sims: (n_queries x n_docs) similarity of every query to every TRAIN card. Returns one negative slug per query."""
    rng, out = random.Random(seed), []
    for i, r in enumerate(q.itertuples()):
        banned = {r.slug} | (set(r.relevant) | set(r.elig_same) if exclude_relevant else set())
        cand = []
        for j in np.argsort(-sims[i])[:300]:
            if slugs[j] not in banned:
                cand.append(slugs[j])
                if len(cand) == pool: break
        out.append(rng.choice(cand) if cand else rng.choice(slugs))
    return out


def duplicate_mask(q_slugs, doc_slugs):
    """mask[i][j] True where doc j is the SAME scheme as query i's target but is not query i's own positive (index i).
    Those would be false negatives from in-batch sampling, so their scores are set to -inf."""
    return [[(doc_slugs[j] == q_slugs[i]) and j != i for j in range(len(doc_slugs))] for i in range(len(q_slugs))]


def find_backbone(model):
    """The underlying HF encoder (has .embeddings and .encoder.layer) - location differs across sentence-transformers versions."""
    for m in model.modules():
        if hasattr(m, "embeddings") and hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
            return m
    return None


def enable_checkpointing(model):
    bb = find_backbone(model)
    if bb is None or not hasattr(bb, "gradient_checkpointing_enable"): return False
    try: bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError: bb.gradient_checkpointing_enable()
    return True


def freeze_lower(model, n_layers):
    """Freeze all embeddings + the first n_layers encoder layers: smaller optimizer state and shorter backward pass."""
    bb = find_backbone(model)
    if bb is None or n_layers <= 0: return 0
    for p in bb.embeddings.parameters(): p.requires_grad = False
    for layer in bb.encoder.layer[:n_layers]:
        for p in layer.parameters(): p.requires_grad = False
    return n_layers


def quick_val(model, docs_val, qval, k=3, bs=64):
    D = model.encode(docs_val.text.tolist(), batch_size=bs, normalize_embeddings=True)
    Q = model.encode(qval["query"].tolist(), batch_size=bs, normalize_embeddings=True)
    pos = {s: i for i, s in enumerate(docs_val.slug)}
    top = np.argsort(-(Q @ D.T), axis=1)[:, :10]
    gold = np.array([pos[s] for s in qval.slug])
    return {"R@3": float((top[:, :3] == gold[:, None]).any(1).mean()), "R@10": float((top == gold[:, None]).any(1).mean())}


def main():
    import torch, torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=S.DENSE_MODEL); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--scale", type=float, default=20.0); ap.add_argument("--seed", type=int, default=S.SEED)
    ap.add_argument("--include-vague", action="store_true"); ap.add_argument("--keep-relevant-as-negatives", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="gradient checkpointing is ON by default")
    ap.add_argument("--freeze-layers", type=int, default=0, help="freeze embeddings + first N layers (try 8-12 if still out of memory)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    docs = L.card_frame(); text = dict(zip(docs.slug, docs.text))
    dtr, dva = docs[docs.split == "train"].reset_index(drop=True), docs[docs.split == "val"].reset_index(drop=True)
    qtr = build_training_rows(L.load_queries("train"), set(dtr.slug), a.include_vague)
    qva = L.load_queries("val"); qva = qva[qva.slug.notna() & qva.slug.isin(dva.slug) & (qva.row_type != "vague")]
    print(f"train queries={len(qtr)} | train cards={len(dtr)} | val queries={len(qva)} | device={dev}")

    model = SentenceTransformer(a.base, device=dev); model.max_seq_length = a.max_len
    base_val = quick_val(model, dva, qva); print("base model on val (split index):", base_val)

    D = model.encode(dtr.text.tolist(), batch_size=64, normalize_embeddings=True)
    Q = model.encode(qtr["query"].tolist(), batch_size=64, normalize_embeddings=True)
    negs = pick_negatives(Q @ D.T, qtr, dtr.slug.tolist(), not a.keep_relevant_as_negatives, seed=a.seed)
    del D, Q

    for n, p in model.named_parameters():
        if "word_embeddings" in n: p.requires_grad = False
    if not a.no_grad_ckpt: print("gradient checkpointing:", "on" if enable_checkpointing(model) else "NOT available (continuing without)")
    if a.freeze_layers: print("frozen lower layers:", freeze_lower(model, a.freeze_layers))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.0f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(qtr) / a.batch); warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    def embed(texts):
        prep = getattr(model, "preprocess", None) or model.tokenize     # newer sentence-transformers renamed it
        feats = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in prep(texts).items()}   # keep non-tensor entries as-is
        return F.normalize(model(feats)["sentence_embedding"], dim=-1)

    best, step = -1.0, 0
    for ep in range(a.epochs):
        model.train(); order = list(range(len(qtr))); random.shuffle(order); run = 0.0
        for b in range(0, len(order), a.batch):
            ids = order[b:b + a.batch]
            if len(ids) < 2: continue
            rows = [qtr.iloc[i] for i in ids]
            q_slugs = [r.slug for r in rows]; n_slugs = [negs[i] for i in ids]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                qe = embed([r["query"] for r in rows])
                de = torch.cat([embed([text[s] for s in q_slugs]), embed([text[s] for s in n_slugs])])
                scores = (qe @ de.T) * a.scale
            mask = torch.tensor(duplicate_mask(q_slugs, q_slugs + n_slugs), device=dev)
            scores = scores.float().masked_fill(mask, float("-inf"))
            loss = F.cross_entropy(scores, torch.arange(len(ids), device=dev))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1; run += loss.item()
            if step % 50 == 0: print(f"  ep{ep + 1} step {step}/{steps} loss={run / 50:.4f}"); run = 0.0
        model.eval(); v = quick_val(model, dva, qva); print(f"epoch {ep + 1}: val {v}  (base {base_val})")
        if v["R@3"] > best:
            best = v["R@3"]; os.makedirs(a.out, exist_ok=True); model.save(a.out); print("  saved ->", a.out)
    print("best val R@3:", round(best, 4))


if __name__ == "__main__":
    main()