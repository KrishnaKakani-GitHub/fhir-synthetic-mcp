# Clinical AI Governance Platform
# Multi-stage build: small runtime image, non-root user, baked-in seed.

# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir --upgrade pip

# Copy and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastmcp \
    pydantic \
    anthropic \
    rank-bm25 \
    fastapi \
    uvicorn \
    && pip install --no-cache-dir -e . --no-deps 2>/dev/null || true

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r clinical && useradd -r -g clinical clinical

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY src/ ./src/
COPY data/synthetic_patients.json ./data/synthetic_patients.json
COPY data/loinc_rules.json ./data/loinc_rules.json
COPY data/clinical_guidelines.json ./data/clinical_guidelines.json
COPY data/clinical_notes.json ./data/clinical_notes.json
COPY scripts/seed_db.py ./scripts/seed_db.py
COPY evals/ ./evals/

# Data and audit directories
RUN mkdir -p /data /audit && chown -R clinical:clinical /app /data /audit

USER clinical

# Seed the database at image build time
# (can be overridden at runtime by mounting /data and setting FHIR_MCP_DB)
ENV FHIR_MCP_DB=/data/fhir.db \
    FHIR_MCP_AUDIT_FILE=/audit/audit.jsonl \
    FHIR_MCP_RAG_DISABLE_CHROMA=1 \
    PORT=8080

RUN python scripts/seed_db.py

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "fhir_mcp.http_server:app", \
     "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
