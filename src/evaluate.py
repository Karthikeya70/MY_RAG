import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import re
import time

from retrieve import search

# Two separate report cards:
#
# 1. RETRIEVAL: did the right chunk appear in the top-k results?
#    Graded by checking that a known "evidence" text (a phrase we verified
#    exists in the policy) shows up in one of the retrieved chunks.
#    This needs NO LLM — it is free and instant, so run it after every change.
#
# 2. GENERATION (--full): is the final answer correct?
#    Graded by checking the answer contains the expected numbers/phrases,
#    or properly refuses when the document has no answer.
#    This calls the LLM once per question, so it is slower.

def normalize(text):
    """Make text comparable: lowercase, collapse whitespace, and turn
    typographic characters into plain ones. LLMs love pretty output —
    e.g. '1800‑103‑2529' with non-breaking hyphens (U+2011) looks identical
    to '1800-103-2529' but fails a naive string match."""
    text = text.lower()
    text = re.sub(r"[‐‑‒–—―−]", "-", text)  # fancy dashes -> -
    text = re.sub(r"[‘’]", "'", text)   # curly single quotes -> '
    text = re.sub(r"[“”]", '"', text)   # curly double quotes -> "
    text = text.replace("*", "").replace("`", "")  # markdown bold/code marks
    return re.sub(r"\s+", " ", text)


REFUSAL_PHRASES = [
    "cannot", "can't", "not contain", "no information", "not mentioned",
    "not provided", "unable to", "not specified", "not available",
    "do not include", "does not include", "not included in",
    "not list", "do not state", "does not state", "not stated",
    "do not specify", "does not specify", "no mention", "not mention",
]


def grade_retrieval(item, k, rerank=True, fuse=True, facts=True):
    """Return (hit, rank). hit=True if any evidence phrase appears in a
    retrieved chunk; rank = position of the first chunk that has it."""
    if not item["evidence"]:
        return None, None  # nothing to look for (refusal questions)

    hits = search(item["question"], plan=item["plan"], k=k,
                  rerank=rerank, fuse=fuse, facts=facts)
    for rank, h in enumerate(hits, start=1):
        chunk_text = normalize(h["text"])
        if any(normalize(ev) in chunk_text for ev in item["evidence"]):
            return True, rank
    return False, None


def grade_answer(item, answer_text):
    """Return True if the answer passes this question's checks."""
    ans = normalize(answer_text)

    if item["expect_refusal"]:
        # The only correct behavior is admitting the document doesn't say.
        return any(p in ans for p in REFUSAL_PHRASES)

    # Every keyword group must be satisfied by at least one of its variants.
    # Example: [["20000", "20,000"]] -> answer must contain 20000 OR 20,000.
    for group in item["expected_keywords"]:
        if not any(normalize(variant) in ans for variant in group):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="also call the LLM and grade the final answers")
    parser.add_argument("--regrade", action="store_true",
                        help="re-score the answers saved in results.json "
                             "without calling the LLM (for grader fixes)")
    parser.add_argument("--k", type=int, default=5,
                        help="how many chunks retrieval returns (default 5)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="skip the reranker stage (to compare against)")
    parser.add_argument("--no-fuse", action="store_true",
                        help="use the reranker's order directly, no fusion "
                             "with the vector order (to compare against)")
    parser.add_argument("--no-facts", action="store_true",
                        help="skip the fine-grained fact index (to compare)")
    parser.add_argument("--judge", action="store_true",
                        help="LLM-as-judge faithfulness check: audit every "
                             "answer for claims not supported by the extracts")
    args = parser.parse_args()

    with open("data/eval/questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    # Answers from the last LLM run. --regrade re-scores them; a plain
    # retrieval-only run carries them over untouched, so a quick retrieval
    # experiment never wipes the answer scores off the scoreboard.
    saved_answers = {}
    saved_faithful = {}
    try:
        with open("data/eval/results.json", encoding="utf-8") as f:
            for r in json.load(f):
                if r.get("answer"):
                    saved_answers[r["id"]] = r["answer"]
                if r.get("faithful") is not None:
                    saved_faithful[r["id"]] = {
                        "faithful": r["faithful"],
                        "unsupported_claims": r.get("unsupported_claims", []),
                    }
    except FileNotFoundError:
        pass

    if args.full:
        from generate import answer as generate_answer
    if args.judge:
        from generate import judge_faithfulness, build_context

    results = []
    print(f"\n{'id':<5} {'type':<15} {'retrieval':<12} {'answer':<8} question")
    print("-" * 90)

    for item in questions:
        hit, rank = grade_retrieval(item, args.k, rerank=not args.no_rerank,
                                    fuse=not args.no_fuse,
                                    facts=not args.no_facts)
        retrieval_str = "-" if hit is None else (f"hit @{rank}" if hit else "MISS")

        answer_ok = None
        answer_text = ""
        if not args.full:
            # reuse (and re-score) the answers from the last LLM run
            answer_text = saved_answers.get(item["id"], "")
            if answer_text:
                answer_ok = grade_answer(item, answer_text)
        elif args.full:
            try:
                # use_cache=False: the eval must measure the live system,
                # not replay yesterday's cached answers
                result = generate_answer(item["question"], plan=item["plan"],
                                         k=args.k, use_cache=False)
                answer_text = result["answer"]
                answer_ok = grade_answer(item, answer_text)
            except Exception as e:
                answer_text = f"ERROR: {e}"
                answer_ok = False
            time.sleep(2)  # be gentle with free-tier rate limits
        answer_str = "-" if answer_ok is None else ("PASS" if answer_ok else "FAIL")

        # Faithfulness audit: did the answer invent anything?
        # (carried over from the last --judge run unless re-judging now)
        verdict = saved_faithful.get(item["id"]) if not args.full else None
        if args.judge and answer_text and not answer_text.startswith("ERROR"):
            hits_for_judge = search(item["question"], plan=item["plan"], k=args.k)
            try:
                verdict = judge_faithfulness(
                    item["question"], build_context(hits_for_judge), answer_text)
            except Exception:
                verdict = None
            time.sleep(2)
        judge_str = ""
        if args.judge:
            judge_str = ("?" if verdict is None
                         else ("faithful" if verdict["faithful"] else "HALLUCINATED"))

        print(f"{item['id']:<5} {item['type']:<15} {retrieval_str:<12} {answer_str:<8} "
              f"{judge_str:<13} {item['question'][:40]}")

        results.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "plan": item["plan"],
            "retrieval_hit": hit,
            "retrieval_rank": rank,
            "answer_pass": answer_ok,
            "answer": answer_text,
            "faithful": verdict["faithful"] if verdict else None,
            "unsupported_claims": verdict["unsupported_claims"] if verdict else [],
        })

    # ── Summary ──────────────────────────────────────────────────────────
    scored = [r for r in results if r["retrieval_hit"] is not None]
    hits = [r for r in scored if r["retrieval_hit"]]
    print("\n================ SUMMARY ================")
    print(f"Retrieval hit@{args.k}: {len(hits)}/{len(scored)}"
          f"  ({100 * len(hits) // len(scored)}%)")

    # Standard IR metrics:
    # hit@1 — how often the correct chunk is the very FIRST result
    # MRR (Mean Reciprocal Rank) — average of 1/rank, 0 for misses;
    #   1.0 = always first, 0.5 = typically second, etc.
    hit1 = [r for r in hits if r["retrieval_rank"] == 1]
    print(f"Retrieval hit@1: {len(hit1)}/{len(scored)}"
          f"  ({100 * len(hit1) // len(scored)}%)")
    mrr = sum(1 / r["retrieval_rank"] for r in hits) / len(scored) if scored else 0
    print(f"MRR: {mrr:.3f}")

    graded_exists = any(r["answer_pass"] is not None for r in results)
    if graded_exists:
        graded = [r for r in results if r["answer_pass"] is not None]
        passed = [r for r in graded if r["answer_pass"]]
        print(f"Answers:   {len(passed)}/{len(graded)} correct  ({100 * len(passed) // len(graded)}%)")
        failed = [r for r in graded if not r["answer_pass"]]
        if failed:
            print("\nFailed answers:")
            for r in failed:
                print(f"  {r['id']} ({r['type']}): {r['question']}")

    judged = [r for r in results if r["faithful"] is not None]
    if judged:
        faithful = [r for r in judged if r["faithful"]]
        print(f"Faithfulness: {len(faithful)}/{len(judged)} answers fully "
              f"grounded in the extracts  ({100 * len(faithful) // len(judged)}%)")
        for r in judged:
            if not r["faithful"]:
                print(f"  {r['id']} HALLUCINATED: {r['unsupported_claims']}")

    with open("data/eval/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nDetailed results saved to data/eval/results.json")


if __name__ == "__main__":
    main()
