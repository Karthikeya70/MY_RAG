---
title: RAG Based Insurance Chatbot
emoji: 🛡️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# RAG Based Insurance Chatbot

Ask questions about two Bajaj Allianz **Health Guard** policy documents (Gold and
Silver plans) and get answers that are **grounded, cited, and honest** — if the
document doesn't say it, the system says so instead of making it up.

Built **without any orchestration framework** (no LangChain / LlamaIndex): every
stage is plain, readable Python, and every design decision below was made by
measuring against an evaluation set — not by guessing.

Presented by **Karthikeya (IITGN)**.

---

## Pipeline

```
PDF ──> parse.py ──> chunk.py ──> embed.py ──> retrieve.py ──> generate.py
        Docling      section-      bge-small     3-way fused     OpenRouter LLM,
        layout +     aware,        embeddings    retrieval +     strict grounding
        tables       token-        + fact index  reranker        prompt + citations
                     budgeted      in ChromaDB
                                        │
                     evaluate.py <──────┘         app.py — Flask UI (ARC Prize theme)
                     20-question eval:            with a live scoreboard
                     hit@k, MRR, answer
                     accuracy, faithfulness
```

### How one question flows

1. **Retrieve** — the question is embedded and three "opinions" are collected:
   - chunk-level vector search (top 20 nearest chunks, plan-filtered first),
   - fact-level vector search (individual sentences vote for their parent
     chunk — "search small, read big"; top-5 facts only),
   - a cross-encoder **reranker** that reads question + chunk together.
   The opinions are combined with **Reciprocal Rank Fusion**; top 5 chunks win.
2. **Semantic cache check** — if a meaning-equivalent question was answered
   before (question distance ≤ 0.18 **and** same top-1 retrieved chunk), the
   stored answer is reused with **no LLM call** (~2s instead of 10–15s).
   Both conditions are measured, not guessed: "expenses *before*
   hospitalisation" vs "*after* discharge" embed at distance 0.106 — closer
   than many true paraphrases — but retrieve different evidence, which is
   what keeps their answers apart. The cache is partitioned by plan.
3. **Generate** — the chunks are pasted into a prompt with strict rules:
   *answer ONLY from these extracts, cite plan/section/page, say so if the
   answer isn't there*. Temperature 0. The answer is stored in the cache.
4. **Audit** (in evaluation) — a second LLM call checks every claim in the
   answer against the extracts and flags anything unsupported (faithfulness).
   Evaluation always bypasses the cache: it must measure the live system.

---

## Results (32-question eval, data/eval/questions.json)

| Metric | Score |
|---|---|
| Retrieval hit@5 | **96%** (24/25 answerable questions) |
| Retrieval hit@1 / MRR | 72% / 0.813 |
| Answer accuracy | **96%** (31/32, includes correct refusals on 7 bait questions) |
| Faithfulness (LLM-as-judge) | **93%** (30/32 — both flags manually audited as judge false positives) |

*Baseline measured end-to-end on google/gemini-2.5-flash-lite (generation and
judging). The one real answer failure is the Nepal reasoning-gap query
(below). The two faithfulness flags were audited by hand: one marks a refusal
as a hallucination (contradicting the judge's own rules), the other marks a
statement directly supported by the extract — judge noise, reported rather
than hidden. Earlier runs on free models (nemotron) scored 32/32 answers and
32/32 faithfulness on the same questions.*

*The eval was deliberately grown from 17 → 20 → 32 questions when it started
saturating at 100% — a test you always pass has stopped teaching you anything.
The growth immediately paid off: it exposed a new failure class (below).*

The eval mixes easy lookups, Gold-vs-Silver trap questions (the two plans differ
in sneaky ways), waiting-period legalese, table lookups, and **hallucination
bait** — questions that sound answerable but aren't in the document (air
ambulance, network hospital lists, premiums), where the only correct answer is
"the document doesn't say."

### Retrieval improvements, step by step

| Configuration | hit@5 |
|---|---|
| Naive vector search | 80% |
| + section headings repeated on split chunks | 80% (fixed q06, others unchanged) |
| + cross-encoder reranker (rank order used directly) | 86% |
| + Reciprocal Rank Fusion (vector ⊕ reranker) | 93% |
| + fact-level index, all 30 facts voting | 86% ← *regression, caught by eval* |
| + fact-level index, **top-5 facts voting** | **100%** |

### Decisions the numbers made

- **Reranker size:** benchmarked `BAAI/bge-reranker-base` (1.1 GB) against
  `ms-marco-MiniLM-L-6-v2` (80 MB): identical 93% at the time, same failure
  cases. **Kept the small one** — same score, ~8× less compute.
- **Fusion over trust:** letting the reranker overrule the vector search
  broke as many questions as it fixed; fusing the rankings kept both strengths.
- **Fact index scope:** indexing table rows as facts flooded the ranking with
  near-duplicates (30 "Email: …" rows) and *dropped* the score to 86%.
  Prose-only facts with capped voting reached 100%.
- **Honesty in grading:** an early "100% answers" score turned out to be a
  grading loophole (the model's "no limit is stated" accidentally matched the
  expected "no sublimit"). The grader was tightened; the honest score was 94%.
  The reverse also happened twice: correct refusals ("no mention of air
  ambulance") were graded FAIL because the phrase list was too literal.
  And once the *eval itself* was wrong: q22 expected joint replacement in the
  24-month waiting list, but the policy gives it a separate 36-month rule —
  the model read the document more carefully than the eval author.
  Lesson: when a test fails, audit the test before the system.

### Known limitations

- **Reasoning-gap queries** (the one retrieval miss): *"Can I take treatment
  in Nepal?"* fails because "Nepal" appears nowhere in the document — linking
  Nepal → outside India → the Territorial Limits section requires world
  knowledge no embedding has. The standard cure is LLM query-rewriting
  (rewrite the question into policy language before searching); identified,
  not yet implemented.

- Docling dropped 3 ombudsman office emails at PDF-extraction time (they never
  reached the index; verified against the raw parse output).
- The Silver PDF contains no UIN; the field is `null` rather than guessed.
- The Silver PDF's page-1 footer is missing, so its page-1 chunks cite page 2.
- The cross-encoder scores "…without any sublimit" as irrelevant to "is there
  a limit?" (negation blindness) — the fact index compensates.

---

## Running it

```bash
pip install -r requirements.txt          # docling, sentence-transformers, chromadb
# put an OpenRouter key in .env:  OPENROUTER_API_KEY=...

python src/parse.py gold                 # PDF -> markdown (repeat: silver)
python src/chunk.py gold                 # markdown -> chunks JSON (repeat: silver)
python src/embed.py                      # chunks + facts -> ChromaDB
python src/retrieve.py "your question" gold      # test retrieval alone
python src/generate.py "your question" gold      # full RAG answer
python src/app.py                        # web UI at http://localhost:5000
```

### Evaluation

```bash
python src/evaluate.py                   # retrieval metrics only (free, fast)
python src/evaluate.py --full            # + LLM answers, graded
python src/evaluate.py --full --judge    # + faithfulness audit (LLM-as-judge)
python src/evaluate.py --no-facts        # ablation: without the fact index
python src/evaluate.py --no-rerank       # ablation: pure vector search
```

The web UI's scoreboard reads `data/eval/results.json` live — run an eval,
refresh the page, the numbers update.

## Stack

| Component | Choice | Why |
|---|---|---|
| PDF parsing | Docling | layout-aware, real table extraction |
| Chunking | custom Python | section-aware, token-budgeted, keeps table records whole, data-integrity checked |
| Embeddings | BAAI/bge-small-en-v1.5 (local) | free, 512-token window, data stays on-machine |
| Vector store | ChromaDB (embedded) | zero ops at this scale; swap for pgvector/Qdrant when the corpus grows |
| Reranker | ms-marco-MiniLM-L-6-v2 | benchmarked equal to a 14× larger model here |
| LLM | gemini-2.5-flash-lite via OpenRouter (free models as fallback) | ~$0.0003/question, no rate-limit stalls; swap models by editing one list |
| UI | Flask + hand-written HTML | full control; themed after arcprize.org |
