import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
from sentence_transformers import SentenceTransformer
import chromadb

# The embedding model. bge-small-en-v1.5 is free, runs locally, and can read
# up to 512 tokens per chunk (many small models stop at 256, which would
# silently ignore the second half of our bigger chunks).
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge models were trained with this prefix on the QUESTION side only.
# It tells the model "this text is a search query, find passages that
# answer it" — without it, retrieval quality drops a little.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

COLLECTION_NAME = "insurance_chunks"

# ─── Step 1: Load the chunks from both plans ─────────────────────────────────
chunks = []
for plan in ["gold", "silver"]:
    with open(f"data/processed/bajaj_{plan}_chunks.json", encoding="utf-8") as f:
        chunks.extend(json.load(f))
print(f"Loaded {len(chunks)} chunks (gold + silver)")

# ─── Step 2: Load the model and check for truncation ─────────────────────────
print(f"Loading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

# The model reads at most max_seq_length tokens; anything beyond that is
# silently cut off. Let's measure our chunks so we KNOW instead of hoping.
tokenizer = model.tokenizer
token_lengths = [len(tokenizer.encode(c["text"])) for c in chunks]
limit = model.max_seq_length
over_limit = [(c, l) for c, l in zip(chunks, token_lengths) if l > limit]
print(f"Model token limit: {limit}")
print(f"Token length: min={min(token_lengths)}, avg={sum(token_lengths)//len(token_lengths)}, max={max(token_lengths)}")
print(f"Chunks longer than the limit (their tail gets ignored): {len(over_limit)}")
for c, l in over_limit:
    print(f"  - {c['chunk_id']}: {c['section']} ({l} tokens)")

# ─── Step 3: Turn every chunk into a vector ──────────────────────────────────
texts = [c["text"] for c in chunks]
print("\nEmbedding all chunks (first run downloads the model, ~130 MB)...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,  # makes cosine similarity behave nicely
)
print(f"Done. Each chunk is now a vector of {embeddings.shape[1]} numbers.")

# ─── Step 4: Store vectors + metadata in ChromaDB ────────────────────────────
client = chromadb.PersistentClient(path=".chroma")

# Start fresh each run so re-running never creates duplicates
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass
collection = client.create_collection(
    COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # compare vectors by angle, the standard for text
)

metadatas = []
for c in chunks:
    metadatas.append({
        "plan": c["plan"],
        "source_file": c["source_file"],
        "UIN": c["UIN"] if c["UIN"] else "",  # Chroma cannot store None
        "section": c["section"],
        "page": c["page"],
    })

collection.add(
    ids=[c["chunk_id"] for c in chunks],
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas,
)
print(f"Stored {collection.count()} chunks in ChromaDB at .chroma/")

# ─── Step 4b: Fine-grained FACT index ("search small, read big") ─────────────
# Problem this solves: a fact buried inside a long multi-topic chunk is
# invisible to search. Example: "Room ... without any sublimit" is one
# clause inside the In-patient Hospitalisation chunk, which also covers
# ICU, surgeons, anesthesia, oxygen... so the whole chunk embeds as
# "generally about hospitalisation" and a room-rent question misses it.
#
# Cure: ALSO index every individual line/sentence ("fact"), each pointing
# back to its parent chunk. Small texts match questions precisely; at
# answer time we hand the LLM the full PARENT chunk, so no context is lost.
FACTS_COLLECTION_NAME = "insurance_facts"
import re

def split_into_facts(chunk):
    """One fact per bullet / sentence of PROSE, prefixed with the section
    name so the fact still knows its topic.

    Table rows are deliberately EXCLUDED: a table already is a list of
    small facts, and indexing every row floods the fact search with
    dozens of near-identical entries (30 'Email: ...' rows, 38 'Room
    Charges' items) that drown out the real matches. Prose is where
    facts hide mid-paragraph — that's the problem this index solves."""
    facts = []
    for line in chunk["text"].split("\n"):
        line = line.strip()
        if not line or line.startswith("##") or line.startswith("|"):
            continue
        # split long prose into sentences
        facts.extend(re.split(r"(?<=[.;])\s+(?=[A-Z0-9])", line))
    facts = [f.strip() for f in facts if len(f.strip()) >= 25]
    return [f'{chunk["section"]} — {f}' for f in facts]

fact_ids, fact_texts, fact_metas = [], [], []
for c in chunks:
    for i, fact in enumerate(split_into_facts(c)):
        fact_ids.append(f'{c["chunk_id"]}#f{i}')
        fact_texts.append(fact)
        fact_metas.append({"parent": c["chunk_id"], "plan": c["plan"]})

print(f"\nEmbedding {len(fact_texts)} fine-grained facts...")
fact_embeddings = model.encode(
    fact_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
)

try:
    client.delete_collection(FACTS_COLLECTION_NAME)
except Exception:
    pass
facts_collection = client.create_collection(
    FACTS_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)
facts_collection.add(
    ids=fact_ids,
    embeddings=fact_embeddings.tolist(),
    documents=fact_texts,
    metadatas=fact_metas,
)
print(f"Stored {facts_collection.count()} facts in ChromaDB")

# ─── Step 5: Sanity test — ask a real question ───────────────────────────────
test_query = "How much does the policy pay for road ambulance charges?"
print(f"\nTest query: {test_query}")
query_vector = model.encode([QUERY_PREFIX + test_query], normalize_embeddings=True)

results = collection.query(query_embeddings=query_vector.tolist(), n_results=3)
for rank, (cid, dist, meta) in enumerate(zip(
    results["ids"][0], results["distances"][0], results["metadatas"][0]
), start=1):
    print(f"\n#{rank}  {cid}  (distance={dist:.4f})")
    print(f"    plan={meta['plan']} | section={meta['section']} | page={meta['page']}")
