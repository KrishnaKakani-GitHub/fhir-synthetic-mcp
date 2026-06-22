"""FastAPI HTTP server wrapping FastMCP over SSE transport.

Exposes the Clinical AI Governance MCP server over HTTPS so it can be
registered as a claude.ai web connector:
  Settings → Connectors → Add → https://your-deployment/mcp

Endpoints:
  GET  /health   — liveness probe (always 200 if server is up)
  GET  /ready    — readiness probe (checks DB connection)
  ALL  /mcp      — FastMCP SSE endpoint (the MCP transport layer)

Security:
  - CORS origins controlled by FHIR_MCP_ALLOWED_ORIGINS env var
  - X-Request-ID header injected on every response
  - Structured JSON access log per request
  - Auth enforcement is inside the MCP tools (auth.py), not at HTTP layer

Run locally:
    uvicorn fhir_mcp.http_server:app --host 0.0.0.0 --port 8080 --reload

PHI NOTE: The HTTP layer logs method, path, status, latency, and request IDs
only. It never logs request bodies, which may contain PHI.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_logger = logging.getLogger("fhir_mcp.http_server")

_PORT = int(os.environ.get("PORT", "8080"))
_HOST = os.environ.get("HOST", "0.0.0.0")
_ALLOWED_ORIGINS = [
    s.strip()
    for s in os.environ.get(
        "FHIR_MCP_ALLOWED_ORIGINS",
        "https://claude.ai,https://api.claude.ai",
    ).split(",")
    if s.strip()
]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='{"ts": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)s}',
    )


_setup_logging()


# Build FastAPI app + attach FastMCP SSE
def _build_app() -> FastAPI:
    from .server import mcp  # import here to defer DB init until startup

    # Mount MCP SSE handler onto FastAPI
    # FastMCP 3.x: get_asgi_app() returns an ASGI app for the /mcp route
    try:
        mcp_asgi = mcp.get_asgi_app()  # FastMCP 3.x
    except AttributeError:
        # Fallback: FastMCP may expose different interface
        mcp_asgi = mcp.asgi_app()

    fast_app = FastAPI(
        title="Clinical AI Governance Platform",
        description="Deterministic-gated FHIR MCP server with tamper-evident audit chain.",
        version="2.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # CORS — allow claude.ai and configured origins
    fast_app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Request ID + structured access logging middleware
    @fast_app.middleware("http")
    async def request_middleware(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        start = time.monotonic()
        response: Response = await call_next(request)
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        response.headers["X-Request-ID"] = request_id
        _logger.info(
            '"method": "%s", "path": "%s", "status": %d, """
            "latency_ms": %s, "request_id": "%s"',
            request.method, request.url.path,
            response.status_code, latency_ms, request_id,
        )
        return response

    # Health + readiness probes
    @fast_app.get("/health", include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok", "service": "clinical-ai-governance"}

    @fast_app.get("/ready", include_in_schema=False)
    async def ready() -> JSONResponse:
        """Readiness probe: checks that the database is accessible."""
        from .server import store
        try:
            store.get_patient_ids()
            return JSONResponse({"status": "ready", "db": "ok"})
        except Exception as e:
            return JSONResponse(
                {"status": "not_ready", "db_error": str(e)},
                status_code=503,
            )

    # Mount MCP SSE on /mcp
    fast_app.mount("/mcp", mcp_asgi)

    return fast_app


app = _build_app()


if __name__ == "__main__":
    uvicorn.run(
        "fhir_mcp.http_server:app",
        host=_HOST,
        port=_PORT,
        reload=False,
        log_config=None,  # use our structured logger
    )
