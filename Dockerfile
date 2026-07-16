# Dockerfile for Hugging Face Spaces (Docker SDK) — and any other host.
#
# Key design points:
# - Only RUNTIME dependencies are installed. Docling (PDF parsing) is a
#   build-your-data tool, not an app dependency — the parsed chunks in
#   data/processed/ are committed, so the image never touches a PDF.
# - CPU-only PyTorch keeps the image several GB smaller than default torch.
# - The two ML models and the ChromaDB index are baked in AT BUILD TIME,
#   so the app boots in seconds instead of downloading models on start.
# - gunicorn imports src/app.py's `app` object directly; the Flask dev
#   server in the __main__ block is never used in production.
# - OPENROUTER_API_KEY is NOT in the image — set it as a Space secret
#   (Settings -> Variables and secrets); generate.py reads it from the
#   environment.

FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first (the big one), then the runtime deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    flask==3.1.3 \
    gunicorn \
    sentence-transformers==5.6.0 \
    chromadb==1.1.1 \
    requests \
    python-dotenv

# App code + the processed data it needs at runtime
COPY src/ src/
COPY data/processed/ data/processed/
COPY data/eval/ data/eval/

# Bake the embedding model + reranker into the image
ENV HF_HOME=/app/.hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Build the vector index (254 chunks + 1092 facts) into the image
RUN python src/embed.py

# Hugging Face Spaces routes traffic to port 7860.
# --timeout 300: worker boot loads two ML models (~30s) and LLM calls can
# take up to 2 minutes on slow days — don't let gunicorn kill the worker.
# -w 1: one worker; each worker holds its own copy of the models in RAM.
EXPOSE 7860
CMD ["gunicorn", "--chdir", "src", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "300", "app:app"]
