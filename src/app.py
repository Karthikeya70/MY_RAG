import sys
sys.stdout.reconfigure(encoding="utf-8")

import os

# Make sure our sibling modules (retrieve.py, generate.py) are importable
# no matter which folder the app is started from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ChromaDB lives in <project root>/.chroma — run from the project root.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import json

from flask import Flask, render_template, request, jsonify
from retrieve import search, collection

app = Flask(__name__)


def load_stats():
    """Live numbers for the scoreboard: chunk count straight from the
    vector database, scores from the most recent eval run. Rerun
    `python src/evaluate.py --full` and refresh the page to update."""
    stats = {"chunks": collection.count(), "docs": 2,
             "retrieval": "—", "answers": "—", "faithfulness": "—"}
    try:
        with open("data/eval/results.json", encoding="utf-8") as f:
            results = json.load(f)
        scored = [r for r in results if r.get("retrieval_hit") is not None]
        if scored:
            hit = sum(1 for r in scored if r["retrieval_hit"])
            stats["retrieval"] = f"{100 * hit // len(scored)}%"
        graded = [r for r in results if r.get("answer_pass") is not None]
        if graded:
            ok = sum(1 for r in graded if r["answer_pass"])
            stats["answers"] = f"{100 * ok // len(graded)}%"
        judged = [r for r in results if r.get("faithful") is not None]
        if judged:
            ok = sum(1 for r in judged if r["faithful"])
            stats["faithfulness"] = f"{100 * ok // len(judged)}%"
    except FileNotFoundError:
        pass  # no eval run yet — scoreboard shows dashes
    return stats


@app.route("/")
def home():
    return render_template("index.html", **load_stats())


@app.route("/api/search", methods=["POST"])
def api_search():
    """Retrieval only: question in -> top chunks out. No LLM, no cost."""
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    plan = data.get("plan") or None
    if not question:
        return jsonify({"error": "Please type a question."}), 400
    hits = search(question, plan=plan, k=int(data.get("k", 5)))
    return jsonify({"hits": hits})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Full pipeline: retrieval + LLM answer (uses OpenRouter free models)."""
    from generate import answer  # imported lazily: only needed for asking
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    plan = data.get("plan") or None
    if not question:
        return jsonify({"error": "Please type a question."}), 400
    try:
        result = answer(question, plan=plan, k=int(data.get("k", 5)))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
