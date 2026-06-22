"""Production entrypoint: runs FastMCP over SSE transport.

Used by Railway/Docker instead of http_server.py to avoid FastAPI
wrapper complexity. FastMCP handles SSE transport natively.

Run:
    python -m fhir_mcp.run_server
"""
from __future__ import annotations

import os
import sys

port = int(os.environ.get("PORT", 8080))

# Import the MCP server (this initialises store, RAG, etc.)
from fhir_mcp.server import mcp  # noqa: E402

if __name__ == "__main__":
    print(f"Starting Clinical AI Governance MCP server on port {port}", flush=True)
    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=port,
    )
