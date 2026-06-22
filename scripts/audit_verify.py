#!/usr/bin/env python3
"""Verify the integrity of an audit chain file.

Usage:
    python scripts/audit_verify.py [path/to/audit.jsonl]

Exit codes:
    0 — chain intact, no tampering detected
    1 — chain broken, log may have been tampered with
    2 — file not found or argument error
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fhir_mcp.audit import verify_chain


def main() -> None:
    if len(sys.argv) > 2:
        print("Usage: audit_verify.py [path/to/audit.jsonl]", file=sys.stderr)
        sys.exit(2)

    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("audit.jsonl")
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    print(f"Verifying audit chain: {path}")
    ok = verify_chain(path)
    if ok:
        print("✓ Chain intact — no tampering detected.")
        sys.exit(0)
    else:
        print("✗ Chain broken — audit log may have been tampered with.")
        sys.exit(1)


if __name__ == "__main__":
    main()
