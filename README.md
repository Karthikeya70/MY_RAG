# RAG Based Insurance Chatbot

**Live demo:** https://my-rag.calmhill-bfae2bec.centralindia.azurecontainerapps.io/
(The first load after a quiet period takes 30-60 seconds. The app sleeps when nobody is using it to keep hosting free.)

A chatbot that answers questions about two real insurance policy documents —
Bajaj Allianz **Health Guard Gold** and **Health Guard Silver**. Every answer:

- comes **only from the policy documents** (never from the AI's general knowledge),
- **cites the exact section and page** it came from,
- and if the document does not contain the answer, the bot **clearly says so**
  instead of making something up.

Built for learning purposes, **without LangChain or LlamaIndex**. Every step is
plain Python you can read and understand. Every design decision in this project
was made by **measuring accuracy on a test set** — not by guessing.

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. The idea is simple:

Large language models (LLMs) are great writers but they answer from memory,
and their memory contains the whole internet — including *other* insurance
companies' policies. Ask a plain LLM "does my policy cover air ambulance?"
and it may confidently describe some other company's rules.

RAG fixes this in two steps:

1. **Retrieve** — first find the exact paragraphs of *your* document that
   relate to the question.
2. **Generate** — then ask the LLM to write an answer *using only those
   paragraphs*, with citations.

The LLM becomes a careful reader of the document instead of a know-it-all.

---

## How the whole system works

```
 PDF files
    |
    v
 1. PARSE        (parse.py)    Read the PDF and convert it to clean text.
    |
    v
 2. CHUNK        (chunk.py)    Cut the text into ~250 small pieces ("chunks"),
    |                          one topic per piece, following the document's
    |                          own section headings.
    v
 3. EMBED        (embed.py)    Convert every chunk into a vector — a list of
    |                          384 numbers that captures its MEANING. Similar
    |                          meanings get similar numbers. Store everything
    |                          in a small local database (ChromaDB).
    v
 4. RETRIEVE     (retrieve.py) When a question comes in, find the 5 chunks
    |                          whose meaning is closest to the question.
    v
 5. GENERATE     (generate.py) Give those 5 chunks to an LLM with strict
    |                          rules: answer only from these, cite everything,
    |                          say "not in the document" if it isn't there.
    v
 6. EVALUATE     (evaluate.py) A 32-question exam with known answers that
                               scores every part of the system.
```

The web app (`app.py` + one HTML page) puts a face on this pipeline.

---

## How one question flows through the system

Say a user asks: *"How much will the policy pay for a road ambulance?"*

1. The question is converted into a vector (its "meaning coordinates").
2. **Three searches run and vote together:**
   - **Chunk search** — find the 20 chunks nearest in meaning.
   - **Sentence search** — every individual sentence of the document was
     also indexed separately. If one sentence matches the question sharply,
     the chunk it belongs to gets a vote. This finds facts that are buried
     in the middle of long chunks.
   - **Reranker** — a second, smaller AI model reads the question TOGETHER
     with each candidate chunk and scores how well that chunk actually
     answers it. (The first search reads them separately, which is faster
     but less accurate.)
   The three rankings are combined with a simple points system
   (Reciprocal Rank Fusion). The best 5 chunks win.
3. **Cache check** — if a very similar question was answered before AND it
   retrieved the same evidence, the saved answer is returned instantly
   (2 seconds instead of 10, and no LLM cost).
4. Otherwise, the 5 chunks + the question + the strict rules go to the LLM
   (Google Gemini 2.5 Flash Lite, via OpenRouter).
5. The answer comes back with citations like
   `[Silver, section "4. Road Ambulance", page 2]` and is saved to the cache.

If the user selected a plan (Gold or Silver), the search is **filtered before
it runs** — so a Silver question can never accidentally pull a Gold rule.
This matters because the two plans differ in tricky ways (example: Gold has
no room-rent limit, Silver caps it at 1% of sum insured per day).

---

## Results

Measured on a 32-question test set (`data/eval/questions.json`). Every
expected answer was verified against the actual document before being added.

| What we measure | Score | Meaning |
|---|---|---|
| Retrieval hit@5 | **96%** | The correct chunk was in the top 5 results |
| Retrieval hit@1 | 72% | The correct chunk was the #1 result |
| MRR | 0.813 | On average the correct chunk ranks close to #1 |
| Answer accuracy | **96%** | The final answer contained the right facts (includes 7 trick questions where refusing to answer is the correct behavior) |
| Faithfulness | **93%** | A second AI audited every answer and found no made-up facts (the 2 flags were checked by hand — both were mistakes by the auditor, not the bot) |

### How retrieval improved, step by step

Each row is a change we tried. The score decided whether it stayed.

| Change | hit@5 | Kept? |
|---|---|---|
| Basic vector search only | 80% | starting point |
| + repeat section headings on split chunks | 80% | yes (fixed one question) |
| + reranker deciding the order alone | 86% | no — fixed 2 questions, broke 2 others |
| + combine rankings with points (fusion) | 93% | yes |
| + sentence-level index, everything votes | 86% | no — table rows flooded the results |
| + sentence-level index, top-5 votes only | **96%** | yes (final) |

### Decisions the measurements made for us

- **Small reranker kept over a 14x bigger one** — we benchmarked both.
  Identical score. The big download was not wasted: now we *know*.
- **A bigger LLM was tried and rejected** — GPT-4o-mini wrote nicer answers
  but added details from its training memory (it invented a payment limit
  that is not in the document). Our faithfulness metric caught it.
- **The test set itself was wrong once** — it expected "24 months" for joint
  replacement surgery, but the policy actually says 36. The bot read the
  document more carefully than the person who wrote the test.
  Lesson: when a test fails, check the test before blaming the system.

---

## The parts in a little more detail

### 1. Parsing (`parse.py`)
Uses **Docling** (an open-source PDF understanding library) to convert the
PDFs into markdown with proper headings and tables. A PDF is basically a
photo of a page — this step turns it back into structured text.

### 2. Chunking (`chunk.py`)
Cuts the text at the document's own section headings, so each chunk covers
one topic. Extra care for tables: a multi-row entry (like one complaint
office's name + address + phone + email) is never split across two chunks,
and the table's header row is copied into every piece. Chunk size is
measured in **tokens using the embedding model's own tokenizer** — because
that model silently ignores everything past 512 tokens, and word counts
underestimate badly for tables. A built-in integrity check confirms that
no line of the document was lost during chunking.

### 3. Embedding and storage (`embed.py`)
Model: `BAAI/bge-small-en-v1.5` (small, free, runs on CPU).
Storage: **ChromaDB** in embedded mode — it is just a folder of files, like
SQLite. No server, no account. Three collections (think: tables) live in it:

| Collection | Contents | Used for |
|---|---|---|
| `insurance_chunks` | 254 chunks + plan/section/page metadata | main search + what the LLM reads |
| `insurance_facts` | 1,092 individual sentences, each pointing to its parent chunk | precise "buried fact" search |
| `answer_cache` | past questions + their answers | skipping the LLM for repeat questions |

### 4. Retrieval (`retrieve.py`)
The three-vote system described above. Ablation flags (`--no-rerank`,
`--no-facts`, `--no-fuse` in the eval) let you turn each part off and
measure what it contributes.

### 5. Generation (`generate.py`)
A plain HTTPS call to OpenRouter (no SDK). The system prompt contains the
five rules that make the bot honest. Temperature is 0, so the same question
always gets the same answer — for insurance facts, consistency is a feature.
A fallback list of models is tried in order if the first one is down.

### 6. The semantic cache (`generate.py`)
Saves every answered question with its meaning-vector. A new question reuses
a saved answer only if **three checks all pass**: the questions are close in
meaning (distance ≤ 0.18), they retrieve the same #1 chunk, and they are for
the same plan. The thresholds came from measurement: "expenses BEFORE
hospitalisation" and "expenses AFTER discharge" sound nearly identical
(distance 0.106!) but have different answers — the evidence check is what
keeps them apart.

### 7. Evaluation (`evaluate.py`)
Two separate scores: did retrieval find the right chunk (free to run, no
LLM needed), and was the final answer correct. Plus a **faithfulness audit**:
a second LLM call checks every sentence of every answer against the source
chunks and flags anything that was made up. Run it yourself:

```bash
python src/evaluate.py                   # retrieval scores only (fast, free)
python src/evaluate.py --full            # + generate and grade real answers
python src/evaluate.py --full --judge    # + faithfulness audit
python src/evaluate.py --no-facts       # example ablation: no sentence index
```

The web app's scoreboard reads the latest results file automatically —
run the eval, refresh the page, the numbers update.

---

## Running it locally

```bash
pip install -r requirements.txt
# create a .env file containing:  OPENROUTER_API_KEY=your_key_here

python src/parse.py gold        # PDF -> markdown   (also: silver)
python src/chunk.py gold        # markdown -> chunks (also: silver)
python src/embed.py             # chunks -> vector database
python src/app.py               # web app at http://localhost:5000
```

You can also test individual stages from the command line:

```bash
python src/retrieve.py "Is there any limit on room rent?" gold
python src/generate.py "How much for a road ambulance?" silver
```

---

## How it is deployed (all free)

```
push to GitHub main branch
        |
        v
GitHub Actions builds the Docker image        (free for public repos)
        |
        v
image published to GitHub Container Registry  (ghcr.io, free)
        |
        v
Azure Container Apps runs it                   (Azure for Students, no card)
```

The Docker image bakes in the two ML models and rebuilds the vector database
at build time, so the running app never downloads anything at startup. The
app scales to zero when idle (that is why the first visit is slow) and the
only secret it needs is the OpenRouter API key, set as an environment
variable in Azure — never committed to git.

---

## Tech stack

| Part | Choice | Why |
|---|---|---|
| PDF parsing | Docling | understands layout and tables |
| Chunking | own Python code | full control over tables and sizes |
| Embeddings | bge-small-en-v1.5 | free, local, 512-token window |
| Vector DB | ChromaDB (embedded) | zero setup; swap for pgvector/Qdrant at scale |
| Reranker | ms-marco-MiniLM-L-6-v2 | benchmarked equal to a 14x bigger model |
| LLM | Gemini 2.5 Flash Lite via OpenRouter | ~$0.0003 per question, reliable |
| Web app | Flask + one hand-written HTML page | no build tools, full control |
| Hosting | Azure Container Apps + GitHub Actions + ghcr | genuinely $0/month |

## Known limitations (documented, not hidden)

- **"Nepal" type questions fail.** The document never says "Nepal", so a
  question about treatment in Nepal cannot find the "India only" rule by
  meaning alone — that connection needs world knowledge. The standard fix
  (rewriting the query with an LLM before searching) was prototyped and is
  described as future work.
- The PDF parser dropped 3 email addresses from a messy table. Whatever
  parsing loses, no later stage can recover.
- The rerankers cannot understand negation ("no sublimit" as the answer to
  "is there a limit?"). The sentence-level index compensates.
- The Silver PDF has no product code (UIN) printed in it, so that field is
  empty rather than guessed.

---

Built by **Karthikeya (IITGN)** as an end-to-end learning project:
parse -> chunk -> embed -> retrieve -> generate -> evaluate -> deploy,
with every step measured and every failure documented.
