"""Principal and approver verification.

Two env vars control access:
    FHIR_MCP_PRINCIPALS — comma-separated agent actor IDs allowed to call tools.
    FHIR_MCP_APPROVERS  — comma-separated human IDs allowed to approve/reject.

If either env var is unset or empty, the corresponding check is skipped
(development mode). In production, always set both.

The agent's actor ID (FHIR_MCP_ACTOR) is only as trustworthy as this layer:
without auth, any caller can claim any actor identity. Keep the two sets
disjoint — an agent principal must not appear in FHIR_MCP_APPROVERS.
"""
from __future__ import annotations

import os


class AuthError(Exception):
    """Raised when an actor is not in the authorised set."""


def _allowed_set(env_var: str) -> frozenset[str] | None:
    """Return the allowed set, or None if the env var is unset (dev mode)."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    return frozenset(v.strip() for v in raw.split(",") if v.strip())


def verify_agent_actor(actor: str) -> None:
    """Raise AuthError if actor is not an authorised agent principal.

    No-op when FHIR_MCP_PRINCIPALS is unset (development mode).
    """
    allowed = _allowed_set("FHIR_MCP_PRINCIPALS")
    if allowed is not None and actor not in allowed:
        raise AuthError(
            f"Actor '{actor}' is not an authorised principal. "
            "Set FHIR_MCP_PRINCIPALS to grant access."
        )


def verify_approver(approver: str) -> None:
    """Raise AuthError if approver is not an authorised human approver.

    No-op when FHIR_MCP_APPROVERS is unset (development mode).
    """
    allowed = _allowed_set("FHIR_MCP_APPROVERS")
    if allowed is not None and approver not in allowed:
        raise AuthError(
            f"Approver '{approver}' is not authorised. "
            "Set FHIR_MCP_APPROVERS to grant approval rights."
        )
