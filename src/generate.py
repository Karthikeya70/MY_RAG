import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import requests
from dotenv import load_dotenv

# Reuse phase 4 — search() gives us the most relevant chunks.
# (Importing retrieve also loads the embedding model + ChromaDB once.)
from retrieve import search, model as embed_model, client as chroma_client

# ─── Semantic answer cache ───────────────────────────────────────────────────
# Users ask the same things in different words ("How much for an ambulance?"
# vs "What's the road ambulance limit?"). Generating costs an LLM call every
# time; the answers would be identical anyway (temperature 0). So we cache
# answers by MEANING: embed each answered question, and when a new question
# lands close enough to a cached one, reuse that answer without calling
# the LLM at all.
#
# Distance alone is NOT safe: we measured "expenses BEFORE hospitalisation"
# vs "expenses AFTER discharge" at distance 0.106 — closer than some true
# paraphrases (0.15+), yet the answers differ (60 vs 90 days). So a cache
# hit requires BOTH:
#   1. question distance <= CACHE_MAX_DISTANCE, and
#   2. the SAME top-1 retrieved chunk — if two questions pull different
#      best evidence, the answer does not transfer, no matter how similar
#      they sound. Retrieval is local and fast; only the LLM call is skipped.
# Both limits were set by measurement: true paraphrases landed at 0.076-0.174
# with equal top-1; every danger pair we tried (before/after hospitalisation
# 0.106, cataract/hernia 0.194, room rent/ICU 0.242) differed in top-1.
CACHE_MAX_DISTANCE = 0.18
answer_cache = chroma_client.get_or_create_collection(
    "answer_cache", metadata={"hnsw:space": "cosine"}
)


def _cache_key_embedding(question):
    # No search prefix here: we compare question-to-question (symmetric),
    # not question-to-passage.
    return embed_model.encode([question], normalize_embeddings=True)


def _cache_lookup(question, plan, hits):
    """Return the cached entry that safely answers this question, or None."""
    if answer_cache.count() == 0:
        return None
    res = answer_cache.query(
        query_embeddings=_cache_key_embedding(question).tolist(),
        n_results=1,
        where={"plan": plan or ""},
    )
    if not res["ids"][0]:
        return None
    dist = res["distances"][0][0]
    meta = res["metadatas"][0][0]
    if dist > CACHE_MAX_DISTANCE:
        return None
    if meta["top_chunks"].split(",")[0] != hits[0]["chunk_id"]:
        return None  # similar words, different evidence — not the same question
    return {
        "answer": res["documents"][0][0],
        "cached_from": meta["question"],
        "model": meta["model"],
        "distance": dist,
    }


def _cache_store(question, plan, hits, answer_text, model_name):
    import hashlib
    key = hashlib.md5(f"{plan}|{question}".encode()).hexdigest()
    answer_cache.upsert(
        ids=[key],
        embeddings=_cache_key_embedding(question).tolist(),
        documents=[answer_text],
        metadatas=[{
            "question": question,
            "plan": plan or "",
            "top_chunks": ",".join(h["chunk_id"] for h in hits[:3]),
            "model": model_name,
        }],
    )

load_dotenv()  # reads .env and puts OPENROUTER_API_KEY into the environment
API_KEY = os.getenv("OPENROUTER_API_KEY")

# OpenRouter model ids to try, in order. Primary is a cheap, reliable PAID
# model (~$0.10 per MILLION input tokens — a full eval costs under a cent):
# no rate-limit stalls, consistent latency. The ':free' models below are
# emergency fallback only, so the app keeps working even without credit.
MODELS = [
    "google/gemini-2.5-flash-lite",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The rules the LLM must follow. This is where RAG gets its honesty:
# the model may ONLY use what retrieval found, never its own memory.
SYSTEM_PROMPT = """You are a careful assistant answering questions about \
Bajaj Allianz Health Guard insurance policies (Gold and Silver plans).

Rules:
1. Answer ONLY from the policy extracts provided by the user. Do not use \
any outside knowledge about insurance or this company.
2. After every fact, cite where it came from in brackets, like: \
[Gold, section "4. Road Ambulance", page 1].
3. If the extracts do not contain the answer, say exactly that — do not guess.
4. If the Gold and Silver plans differ, state each plan's answer separately.
5. Quote exact numbers, limits and time periods as written in the extracts."""


def call_llm(messages):
    """Send a chat request to OpenRouter, walking the fallback list until
    a model answers. Returns the raw response JSON."""
    errors = []
    for model in MODELS:
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0,  # no creativity wanted — deterministic answers
                # Cap the output length. Without this, OpenRouter reserves the
                # model's MAXIMUM possible output (65k tokens for Gemini) and
                # rejects the call unless the balance could afford that worst
                # case — so an unset max_tokens makes big-output models fail
                # on small balances even though real answers are ~300 tokens.
                "max_tokens": 2000,
            },
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        # OpenRouter sends the reason as JSON — remember it, try the next model
        detail = response.json().get("error", {}).get("message", response.text)
        errors.append(f"{model} -> {response.status_code}: {detail}")
        print(f"  ({model} unavailable, trying next...)", file=sys.stderr)

    raise RuntimeError(
        "All models failed:\n  " + "\n  ".join(errors) + "\n"
        "(402 means no credits — use ':free' model ids. "
        "429 means the free rate limit is hit — wait a minute and retry.)"
    )


# The faithfulness judge: a SECOND LLM call that audits the first one.
# It gets the same extracts plus the generated answer and must flag any
# claim that is not directly supported by the extracts. This is the
# standard "LLM-as-judge faithfulness" metric (as popularised by RAGAS):
# it measures exactly the thing we care most about — did the model stick
# to the document, or did it quietly add things it "knows" from training?
JUDGE_PROMPT = """You are auditing an insurance Q&A system for hallucinations.

You get: policy extracts, a question, and the system's answer.
Check EVERY factual claim in the answer against the extracts.

Reply with ONLY a JSON object, no other text:
{"faithful": true or false, "unsupported_claims": ["claim 1", ...]}

Rules:
- "faithful": true means every factual claim is directly supported by the extracts.
- Saying "the extracts do not contain this information" is always acceptable.
- Citation formatting, phrasing and summarisation are fine — only invented FACTS matter.
- If the answer adds numbers, conditions, or details not present in the extracts, it is not faithful."""


def judge_faithfulness(question, context_text, answer_text):
    """Return {'faithful': bool, 'unsupported_claims': [...]} or None if
    the judge's reply could not be parsed."""
    import json as _json
    import re as _re

    user_message = (
        "Policy extracts:\n\n" + context_text
        + f"\n\nQuestion: {question}\n\nSystem's answer:\n{answer_text}"
    )
    data = call_llm([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_message},
    ])
    reply = data["choices"][0]["message"]["content"]
    match = _re.search(r"\{.*\}", reply, _re.DOTALL)
    if not match:
        return None
    try:
        verdict = _json.loads(match.group(0))
        return {
            "faithful": bool(verdict.get("faithful")),
            "unsupported_claims": verdict.get("unsupported_claims", []),
        }
    except Exception:
        return None


def build_context(hits):
    """Paste the retrieved chunks into one text block the LLM can read.
    Each chunk is labelled so the model can cite it."""
    blocks = []
    for h in hits:
        blocks.append(
            f'--- Extract [{h["plan"]}, section "{h["section"]}", page {h["page"]}] ---\n'
            f'{h["text"]}'
        )
    return "\n\n".join(blocks)


def answer(question, plan=None, k=5, use_cache=True):
    """The whole RAG pipeline:
    1. retrieve the k most relevant chunks (optionally one plan only)
    2. check the semantic cache — a meaning-equivalent question with the
       SAME retrieved evidence reuses its answer, skipping the LLM
    3. otherwise: prompt = rules + chunks + question -> LLM -> cache it

    use_cache=False bypasses the cache entirely (evaluation must measure
    the live system, not yesterday's cached answers).
    """
    hits = search(question, plan=plan, k=k)

    if use_cache:
        cached = _cache_lookup(question, plan, hits)
        if cached:
            return {
                "answer": cached["answer"],
                "sources": hits,
                "model": cached["model"],
                "usage": {},
                "cached": True,
                "cached_from": cached["cached_from"],
            }

    user_message = (
        "Policy extracts:\n\n"
        + build_context(hits)
        + f"\n\nQuestion: {question}"
    )

    data = call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    answer_text = data["choices"][0]["message"]["content"]
    model_name = data.get("model", "?")
    if use_cache:
        _cache_store(question, plan, hits, answer_text, model_name)

    return {
        "answer": answer_text,
        "sources": hits,
        "model": model_name,
        "usage": data.get("usage", {}),
        "cached": False,
    }


if __name__ == "__main__":
    # Usage: python src/generate.py "your question" [gold|silver]
    if len(sys.argv) < 2:
        print('Usage: python src/generate.py "your question" [gold|silver]')
        sys.exit(1)

    question = sys.argv[1]
    plan = sys.argv[2].capitalize() if len(sys.argv) > 2 else None

    print(f"\nQuestion: {question}")
    print(f"Plan filter: {plan or 'none (searching both plans)'}")

    import time as _time
    t0 = _time.time()
    result = answer(question, plan=plan)
    took = _time.time() - t0

    src_label = ("SEMANTIC CACHE, no LLM call" if result.get("cached")
                 else result["model"])
    print(f"\n================ ANSWER ({src_label}, {took:.1f}s) ================\n")
    if result.get("cached"):
        print(f'(cached from: "{result["cached_from"]}")\n')
    print(result["answer"])

    print("\n================ SOURCES GIVEN TO THE MODEL ================")
    for h in result["sources"]:
        print(f'  {h["chunk_id"]}  {h["plan"]} | {h["section"]} | page {h["page"]}  (distance {h["distance"]:.3f})')

    u = result["usage"]
    if u:
        print(f'\nTokens: {u.get("prompt_tokens", "?")} in / {u.get("completion_tokens", "?")} out')
