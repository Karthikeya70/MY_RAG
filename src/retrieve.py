import sys
sys.stdout.reconfigure(encoding="utf-8")

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# Must match what embed.py used — a question embedded with a DIFFERENT
# model would land on a different "meaning map" and find nonsense.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION_NAME = "insurance_chunks"

# Second-stage reranker. The embedding model reads the question and each
# chunk SEPARATELY (that's what makes it fast enough to scan everything),
# so it misses subtle connections — e.g. that "limit on room rent" is
# answered by a chunk saying "Room, Boarding ... without any sublimit".
# A cross-encoder reads question + chunk TOGETHER and judges the pair,
# which is far more accurate but slower — so we only apply it to the
# top candidates the fast search already found.
# Benchmarked BAAI/bge-reranker-base (1.1GB) against this 80MB model on the
# 17-question eval: both scored 93% hit@5 with fusion, and both failed the
# same question (q04 — a granularity problem no reranker can fix). Same
# score, ~8x less compute -> the small model stays. See data/eval.
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20  # how many chunks the fast search hands to the reranker

print("Loading embedding model...", file=sys.stderr)
model = SentenceTransformer(MODEL_NAME)
print("Loading reranker...", file=sys.stderr)
reranker = CrossEncoder(RERANKER_NAME)
client = chromadb.PersistentClient(path=".chroma")
collection = client.get_collection(COLLECTION_NAME)
# Fine-grained index: individual lines/sentences pointing to their parent
# chunk ("search small, read big") — built by embed.py step 4b.
facts_collection = client.get_collection("insurance_facts")


def _chunk_record(cid, meta, text, distance=None):
    return {
        "chunk_id": cid,
        "distance": distance,
        "plan": meta["plan"],
        "section": meta["section"],
        "page": meta["page"],
        "source_file": meta["source_file"],
        "text": text,
    }


def search(question, plan=None, k=5, rerank=True, fuse=True, facts=True):
    """Find the k chunks that best answer the question.

    Three independent opinions are collected and fused:

    A. CHUNK vector search — the 20 chunks nearest in meaning (plan filter
       applied FIRST, so asking about Silver can never sneak in Gold).
    B. FACT vector search — the 30 nearest individual lines/sentences;
       each fact votes for its PARENT chunk. This finds facts buried
       inside long multi-topic chunks ("search small, read big").
    C. RERANKER — a cross-encoder re-reads the question TOGETHER with
       every candidate chunk and scores how well it actually answers.

    The rankings are combined with Reciprocal Rank Fusion; the best k
    chunks survive. Flags turn stages off for ablation comparisons:
    rerank=False (A+B only), fuse=False (C's order wins outright),
    facts=False (A+C only).
    """
    query_vector = model.encode(
        [QUERY_PREFIX + question], normalize_embeddings=True
    )
    where = {"plan": plan} if plan else None

    # ── Opinion A: chunk-level vector search ────────────────────────────
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=CANDIDATES if (rerank or facts) else k,
        where=where,
    )
    pool = {}           # chunk_id -> full record (the candidate pool)
    vector_order = []   # chunk ids in opinion-A order
    for cid, dist, meta, text in zip(
        results["ids"][0], results["distances"][0],
        results["metadatas"][0], results["documents"][0],
    ):
        pool[cid] = _chunk_record(cid, meta, text, distance=dist)
        vector_order.append(cid)

    # ── Opinion B: fact-level search, facts vote for their parents ──────
    # Only the top few facts get a vote. A fact ranking is razor-sharp at
    # the top ("this exact sentence answers the question") but full of
    # weak lookalikes further down — letting all 30 vote floods the
    # fusion and drowns out the chunk-level results.
    FACT_VOTES = 5
    fact_order = []
    if facts:
        fresults = facts_collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=30,
            where=where,
        )
        for fmeta, fdist in zip(fresults["metadatas"][0], fresults["distances"][0]):
            pid = fmeta["parent"]
            if pid not in fact_order:
                fact_order.append(pid)
            if pid not in pool:
                # parent chunk wasn't in opinion A's top-20 — fetch it
                got = collection.get(ids=[pid])
                if got["ids"]:
                    pool[pid] = _chunk_record(
                        pid, got["metadatas"][0], got["documents"][0],
                        distance=fdist,  # best proxy we have for the UI bar
                    )
            if len(fact_order) >= FACT_VOTES:
                break

    hits = list(pool.values())

    # ── Opinion C: cross-encoder reranking of the whole pool ────────────
    rerank_order = []
    if rerank and hits:
        scores = reranker.predict([(question, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        rerank_order = [h["chunk_id"] for h in
                        sorted(hits, key=lambda h: h["rerank_score"], reverse=True)]

    # ── Fuse with Reciprocal Rank Fusion ────────────────────────────────
    # Each chunk earns 1/(60+rank) points per ranking it appears in.
    # A chunk several opinions like wins; a chunk only ONE opinion likes
    # survives if that opinion is confident (rank 1-2). The constant 60
    # comes from the original RRF paper.
    if rerank and not fuse:
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    else:
        rankings = [r for r in (vector_order, fact_order, rerank_order) if r]
        points = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking, start=1):
                points[cid] = points.get(cid, 0) + 1 / (60 + rank)
        for h in hits:
            h["fused_score"] = points.get(h["chunk_id"], 0)
        hits.sort(key=lambda h: h["fused_score"], reverse=True)

    return hits[:k]


if __name__ == "__main__":
    # Usage: python src/retrieve.py "your question" [gold|silver]
    if len(sys.argv) < 2:
        print('Usage: python src/retrieve.py "your question" [gold|silver]')
        sys.exit(1)

    question = sys.argv[1]
    plan = None
    if len(sys.argv) > 2:
        plan = sys.argv[2].capitalize()  # gold -> Gold

    print(f"\nQuestion: {question}")
    print(f"Plan filter: {plan or 'none (searching both plans)'}\n")

    for rank, hit in enumerate(search(question, plan=plan), start=1):
        print(f"#{rank}  {hit['chunk_id']}  distance={hit['distance']:.4f}")
        print(f"    {hit['plan']} | {hit['section']} | page {hit['page']}")
        preview = ' '.join(hit['text'].split())[:220]
        print(f"    {preview}\n")
