FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    "fastmcp>=2.0" \
    pydantic \
    anthropic \
    rank-bm25 \
    fastapi \
    "uvicorn[standard]"

# Copy source
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/seed_db.py ./scripts/seed_db.py
COPY evals/ ./evals/
COPY pyproject.toml ./

# Make src importable
ENV PYTHONPATH=/app/src

# Seed DB
RUN mkdir -p /data && FHIR_MCP_DB=/data/fhir.db python scripts/seed_db.py

ENV FHIR_MCP_DB=/data/fhir.db \
    FHIR_MCP_AUDIT_FILE=/data/audit.jsonl \
    FHIR_MCP_RAG_DISABLE_CHROMA=1 \
    PORT=8080

EXPOSE 8080

CMD python src/fhir_mcp/run_server.py
