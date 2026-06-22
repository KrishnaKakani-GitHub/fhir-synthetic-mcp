"""Tamper-evident structured audit logging.

Each audit record is a JSON line appended to an append-only file.
A SHA-256 hash chain links records: each record's `prev_hash` field
contains the hash of the previous record (or 'GENESIS' for the first).
Tampering with any record breaks the chain; run `scripts/audit_verify.py`
to detect breaks.

Transport note: the MCP server uses stdio; stdout is the JSON-RPC channel.
Audit records go to FHIR_MCP_AUDIT_FILE or stderr — never stdout.

PHI NOTE: records log IDs and actions only — never contents. This is
PHI minimisation: the chain proves what happened without copying sensitive
payloads into a second location that also needs securing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_chain_lock = threading.Lock()
_logger = logging.getLogger("fhir_mcp.audit")

# Audit file: set FHIR_MCP_AUDIT_FILE to persist to disk.
# Unset → records go to stderr (suitable for stdio MCP transport).
_AUDIT_PATH: Path | None = (
    Path(os.environ["FHIR_MCP_AUDIT_FILE"])
    if "FHIR_MCP_AUDIT_FILE" in os.environ
    else None
)


def _hash_line(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _read_last_hash(path: Path) -> str:
    """Return the hash of the last non-empty line in the file, or 'GENESIS'."""
    if not path.exists():
        return "GENESIS"
    last: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                last = stripped
    return _hash_line(last) if last is not None else "GENESIS"


# Initialise the chain tip. Reads the last record from the file so the
# chain is continuous across process restarts.
_prev_hash: str = (
    _read_last_hash(_AUDIT_PATH)
    if _AUDIT_PATH is not None
    else "GENESIS"
)


def _ensure_stderr_handler() -> None:
    if not _logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(h)
        _logger.setLevel(logging.INFO)


_ensure_stderr_handler()


def audit(
    *,
    actor: str,
    action: str,
    reason: str,
    target_ids: list[str] | None = None,
    outcome: str = "ok",
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit one tamper-evident structured audit record.

    actor:      who initiated (agent id / human approver).
    action:     what operation ('propose_observation', 'approve_write', …).
    reason:     free-text caller justification, recorded verbatim.
    target_ids: affected resource IDs only — never record contents.
    outcome:    'ok' or 'error'.
    extra:      additional structured fields (write_id, error, …).
    """
    global _prev_hash

    with _chain_lock:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "reason": reason,
            "target_ids": target_ids or [],
            "outcome": outcome,
            "prev_hash": _prev_hash,
        }
        if extra:
            record.update(extra)

        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        current_hash = _hash_line(line)
        _prev_hash = current_hash

        if _AUDIT_PATH is not None:
            with _AUDIT_PATH.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            _logger.info(line)


# ------------------------------------------------------------------
# Chain verification (also used by scripts/audit_verify.py)
# ------------------------------------------------------------------


def verify_chain(path: Path) -> bool:
    """Verify the SHA-256 hash chain of an audit file.

    Returns True if the chain is intact (no tampering detected).
    Returns False and prints the offending line(s) to stderr if broken.
    """
    prev = "GENESIS"
    ok = True
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  ✗ line {lineno}: invalid JSON", file=sys.stderr)
                ok = False
                continue
            claimed = record.get("prev_hash", "")
            if claimed != prev:
                print(
                    f"  ✗ line {lineno}: prev_hash mismatch "
                    f"(expected …{prev[-12:]}, got …{claimed[-12:]})",
                    file=sys.stderr,
                )
                ok = False
            # Re-serialise with same settings as audit() to get a stable hash
            prev = _hash_line(
                json.dumps(record, separators=(",", ":"), sort_keys=True)
            )
    return ok
