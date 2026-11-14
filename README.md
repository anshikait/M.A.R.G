# JANORA code, aligned with the marg data layout

## 1. Install into your repo
Copy `configs/ src/ eval/ requirements.txt` into the ROOT of the `marg` repo (next to `data/`). Nothing in `data/` is touched.

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## 2. Verify the data (always first)
    python -m src.data_prep.inspect_data     # every field should say OK; fix configs/settings.py FIELDS if one says NOT FOUND
    python -m src.data_prep.check_data       # must print PASS: no scheme in two splits, every query target in its own split

## 3. Experiments in plan order (each prints tables; CSVs go to eval/results/)
    # Step 4 - retrieval. FIRST reproduce your keyword baseline (target: Recall@3 = 0.321 overall):
    python -m eval.evaluate_retrieval --split val --modes bm25
    #   if it does not match, try:  --index split   (index only the val schemes)
    # then dense + hybrid (BGE-M3, ~568M params: use a Colab/Kaggle GPU; first run caches embeddings in data/cache/):
    python -m eval.evaluate_retrieval --split val --modes bm25 dense hybrid

    # Step 6 - understanding the user (intent = row_type, slots vs true_profile; English rows)
    python -m eval.evaluate_nlu --split val

    # Step 7 - eligibility
    python -m eval.evaluate_eligibility --split val      # rule engine vs eligible_same_category_slugs / relevant_slugs
    python -m src.reasoning.constraint_extractor         # regex extractor over the 1,482 detail records -> data/cache/constraints_regex.jsonl
    python -m eval.evaluate_constraints                  # regex extractor vs rag_schemes.json rules (silver labels)
    python -m src.reasoning.nli_matcher                  # NLI training pairs from train/val schemes only -> data/cache/nli_*.jsonl

Run `--split test` only ONCE, at the very end.

## 4. Run the app
    JANORA_DENSE=0 uvicorn src.api.app:app --reload      # BM25 only, no model download
    JANORA_DENSE=1 uvicorn src.api.app:app --reload      # BM25 + BGE-M3 hybrid
    JANORA_MT=nllb  ...                                   # pivot: translate non-English to English first
    JANORA_NATIVE=1 ...                                   # native: retrieve with the original-language query
    # open http://127.0.0.1:8000/docs  ->  POST /chat {"session_id":"u1","text":"..."}

## What each file does
configs/settings.py            paths + field-name candidates (edit if a field is renamed)
configs/constraint_schema.py   Constraint, 3-valued logic, profile aliases (income->annual_income, caste->caste_category), state abbreviations
src/data_prep/loaders.py       readers for every data file (slug-joined), HTML-entity cleanup, nested detail flattening, integrity report
src/retrieval/hybrid_retriever.py   BM25 + dense + RRF over scheme cards (Indic-safe tokenizer, embedding cache)
src/reasoning/rag_rules.py     rag_schemes.json rule -> Constraint list
src/reasoning/rule_engine.py   satisfied / unsatisfied / unknown per condition, with exceptions and OR groups
src/reasoning/ranker.py        score = retrieval relevance x share of conditions met; clarifying question by information gain
src/reasoning/constraint_extractor.py   regex baseline + LLM silver-label prompt; adds state from registry for state-level schemes
src/reasoning/nli_matcher.py   synthetic NLI pair generator (atomic hypotheses) + NLI model wrapper (wrapper untested)
src/nlu/slots.py, intent.py, profile_manager.py   slot regex baseline, row-type classifier, defaults/overrides/what-if/relative asker
src/translation/translator.py  language ID, number normalisation (lakh, hazaar, '1,50,000', 'fifty three'), glossary masking, NLLB backend
src/api/app.py                 FastAPI pipeline (named / standard / vague / offtopic routing)

## Notes
* The scheme_split_v1.json file in the sample upload hashes to sha256 df348a574f05cedd; your plan says ae21571e152abac9.
  `check_data` reports whether the plan hash matches. If not, check `git log` for that file (or the hash may use a canonical-JSON form).
* Models that need a download (BGE-M3, NLLB, NLI) were NOT run in the sandbox; the code paths are written but untested.

## Step 4/5 additions
    # oracle state filter (upper bound of what the slot extractor's state can give):
    python -m eval.evaluate_retrieval --split val --modes bm25 dense hybrid --index split --state-filter oracle
    # step 5a: fine-tune BGE-M3 (Colab T4, ~10-25 min); only train queries + train cards are used
    python -m src.retrieval.finetune_retriever --out /content/drive/MyDrive/janora_models/bge_m3_janora
    python -m eval.evaluate_retrieval --split val --modes dense hybrid --index split --model /content/drive/MyDrive/janora_models/bge_m3_janora
