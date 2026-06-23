"""Principal and approver verification + PHI/DUA enforcement.

Environment variables:
    FHIR_MCP_PRINCIPALS  comma-separated agent actor IDs allowed to call tools
    FHIR_MCP_APPROVERS   comma-separated human IDs allowed to approve/reject
    FHIR_MCP_DUAS        comma-separated actor IDs with a signed Data Use Agreement
    FHIR_MCP_PHI_MODE    'strict' enables DUA gate on all PHI reads (default: off)

Development mode (env vars unset):
    All checks are skipped. Never run dev mode against real PHI.

Production with real PHI — set all four:
    FHIR_MCP_PRINCIPALS=agent:prod
    FHIR_MCP_APPROVERS=dr.smith,dr.jones
    FHIR_MCP_DUAS=agent:prod          # only actors that signed your institutional DUA
    FHIR_MCP_PHI_MODE=strict

DUA enforcement order (strict mode):
    1. Is actor in FHIR_MCP_PRINCIPALS?  (identity gate)
    2. Is actor in FHIR_MCP_DUAS?        (data use agreement gate)
    Both must pass before any PHI read is allowed.

Keep FHIR_MCP_PRINCIPALS and FHIR_MCP_APPROVERS disjoint:
    An agent principal must not appear in FHIR_MCP_APPROVERS.
"""
from __future__ import annotations

import os

_PHI_MODE = os.environ.get("FHIR_MCP_PHI_MODE", "").strip().lower()
PHI_MODE_STRICT: bool = _PHI_MODE == "strict"


class AuthError(Exception):
    """Raised when an actor fails any auth or DUA check."""


def _allowed_set(env_var: str) -> frozenset[str] | None:
    """Return the allowed set, or None if unset (dev mode)."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    return frozenset(v.strip() for v in raw.split(",") if v.strip())


def verify_agent_actor(actor: str) -> None:
    """Verify actor is an authorised principal.

    In strict PHI mode, also verifies DUA signature.
    No-op when FHIR_MCP_PRINCIPALS is unset (dev mode).
    """
    allowed = _allowed_set("FHIR_MCP_PRINCIPALS")
    if allowed is not None and actor not in allowed:
        raise AuthError(
            f"Actor '{actor}' is not an authorised principal. "
            "Set FHIR_MCP_PRINCIPALS to grant access."
        )
    if PHI_MODE_STRICT:
        verify_dua(actor)


def verify_approver(approver: str) -> None:
    """Verify approver is an authorised human approver.

    No-op when FHIR_MCP_APPROVERS is unset (dev mode).
    """
    allowed = _allowed_set("FHIR_MCP_APPROVERS")
    if allowed is not None and approver not in allowed:
        raise AuthError(
            f"Approver '{approver}' is not authorised. "
            "Set FHIR_MCP_APPROVERS to grant approval rights."
        )


def verify_dua(actor: str) -> None:
    """Verify the actor has a signed Data Use Agreement on file.

    Enforced when FHIR_MCP_PHI_MODE=strict OR called directly.
    No-op when FHIR_MCP_DUAS is unset (dev mode).

    In production: populate FHIR_MCP_DUAS with actor IDs whose
    institutional DUA paperwork has been completed and filed.

    PHI NOTE: This is a process gate, not a cryptographic one.
    It confirms the actor's ID appears on the approved list.
    Pair with mutual TLS or JWT verification at the network layer
    for cryptographic identity assurance.
    """
    dua_set = _allowed_set("FHIR_MCP_DUAS")
    if dua_set is None:
        return  # dev mode
    if actor not in dua_set:
        raise AuthError(
            f"Actor '{actor}' does not have a signed Data Use Agreement. "
            "Add actor to FHIR_MCP_DUAS after DUA paperwork is complete."
        )


def is_phi_mode_strict() -> bool:
    """Return True if FHIR_MCP_PHI_MODE=strict is set."""
    return PHI_MODE_STRICT
