"""FHIR MCP server — Clinical AI Governance Platform.

This is the file Claude Code and the Agent SDK launch. It defines the
tools Claude can call. Each tool:
  1. Verifies the caller's identity (auth layer)
  2. Delegates to the store
  3. Emits a tamper-evident audit record on both success and error paths

The write gate invariant is preserved:
  - Agents can READ and PROPOSE
  - Only a verified human approver can COMMIT via approve_write

Mental model: this server is a passive provider. Claude connects, asks
"what tools do you have?", and calls them. The server never calls Claude.

Run locally:   python -m fhir_mcp.server
Register:      claude mcp add clinical-governance -- python -m fhir_mcp.server
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from fastmcp import FastMCP

from .audit import audit
from .auth import AuthError, verify_agent_actor, verify_approver
from .models import ProposedObservation
from .store import FhirStore, StoreError

_DB_PATH = Path(
    os.environ.get(
        "FHIR_MCP_DB",
        Path(__file__).resolve().parents[2] / "data" / "fhir.db",
    )
)

mcp = FastMCP("clinical-ai-governance")
store = FhirStore(_DB_PATH)

# Agent actor identity. In production this comes from authenticated identity,
# NOT a constant — the audit 'actor' is only as trustworthy as auth.py.
_AGENT_ACTOR = os.environ.get("FHIR_MCP_ACTOR", "agent:dev")


# --- Read tools ---------------------------------------------------------------


@mcp.tool()
def list_patients(reason: str) -> list[str]:
    """List available patient IDs. `reason`: why you need this list."""
    try:
        verify_agent_actor(_AGENT_ACTOR)
    except AuthError as e:
        audit(actor=_AGENT_ACTOR, action="list_patients", reason=reason,
              outcome="error", extra={"error": str(e)})
        raise
    ids = store.get_patient_ids()
    audit(actor=_AGENT_ACTOR, action="list_patients", reason=reason, target_ids=ids)
    return ids


@mcp.tool()
def get_patient(patient_id: str, reason: str) -> dict:
    """Read one patient's demographics. `reason` is recorded in the audit trail."""
    try:
        verify_agent_actor(_AGENT_ACTOR)
        patient = store.get_patient(patient_id)
    except (AuthError, StoreError) as e:
        audit(actor=_AGENT_ACTOR, action="get_patient", reason=reason,
              target_ids=[patient_id], outcome="error", extra={"error": str(e)})
        raise
    audit(actor=_AGENT_ACTOR, action="get_patient", reason=reason,
          target_ids=[patient_id])
    return patient.model_dump(mode="json")


@mcp.tool()
def list_observations(patient_id: str, reason: str) -> list[dict]:
    """List a patient's observations. `reason` is audited."""
    try:
        verify_agent_actor(_AGENT_ACTOR)
        obs = store.list_observations(patient_id)
    except (AuthError, StoreError) as e:
        audit(actor=_AGENT_ACTOR, action="list_observations", reason=reason,
              target_ids=[patient_id], outcome="error", extra={"error": str(e)})
        raise
    audit(actor=_AGENT_ACTOR, action="list_observations", reason=reason,
          target_ids=[patient_id])
    return [o.model_dump(mode="json") for o in obs]


# --- Gated write tools --------------------------------------------------------
# The agent can PROPOSE and LIST pending, but it CANNOT approve.
# Approval is a separate tool driven by a human in the loop.


@mcp.tool()
def propose_observation(
    patient_id: str,
    code: str,
    display: str,
    value: float,
    unit: str,
    effective_date: str,
    reason: str,
) -> dict:
    """Propose a new observation. Stages it for human approval; does NOT write.

    Returns a pending-write ticket (write_id). A human must call
    `approve_write` before anything is committed to the database.
    `effective_date` is ISO format YYYY-MM-DD.
    """
    proposed = ProposedObservation(
        patient_id=patient_id,
        code=code,
        display=display,
        value=value,
        unit=unit,
        effective_date=date.fromisoformat(effective_date),
    )
    try:
        verify_agent_actor(_AGENT_ACTOR)
        pending = store.stage_write(proposed)
    except (AuthError, StoreError, ValueError) as e:
        audit(actor=_AGENT_ACTOR, action="propose_observation", reason=reason,
              target_ids=[patient_id], outcome="error", extra={"error": str(e)})
        raise
    audit(actor=_AGENT_ACTOR, action="propose_observation", reason=reason,
          target_ids=[patient_id],
          extra={"write_id": pending.write_id, "status": "pending"})
    return pending.model_dump(mode="json")


@mcp.tool()
def list_pending_writes(reason: str) -> list[dict]:
    """List writes awaiting human approval. For the reviewer's eyes."""
    try:
        verify_agent_actor(_AGENT_ACTOR)
    except AuthError as e:
        audit(actor=_AGENT_ACTOR, action="list_pending_writes", reason=reason,
              outcome="error", extra={"error": str(e)})
        raise
    pending = store.list_pending()
    audit(actor=_AGENT_ACTOR, action="list_pending_writes", reason=reason,
          target_ids=[p.write_id for p in pending])
    return [p.model_dump(mode="json") for p in pending]


@mcp.tool()
def approve_write(write_id: str, approver: str, reason: str) -> dict:
    """HUMAN-IN-THE-LOOP GATE. Commit a staged write.

    `approver` must identify the human authorising this. This is the only
    path that writes patient data to the database. Use deliberately.
    """
    try:
        verify_approver(approver)
        obs = store.approve_write(write_id, approver=approver)
    except (AuthError, StoreError) as e:
        audit(actor=approver, action="approve_write", reason=reason,
              target_ids=[write_id], outcome="error", extra={"error": str(e)})
        raise
    audit(actor=approver, action="approve_write", reason=reason,
          target_ids=[write_id, obs.id],
          extra={"committed_observation_id": obs.id})
    return obs.model_dump(mode="json")


@mcp.tool()
def reject_write(write_id: str, approver: str, reason: str) -> dict:
    """HUMAN-IN-THE-LOOP GATE. Reject a staged write (nothing is committed)."""
    try:
        verify_approver(approver)
        pending = store.reject_write(write_id, approver=approver)
    except (AuthError, StoreError) as e:
        audit(actor=approver, action="reject_write", reason=reason,
              target_ids=[write_id], outcome="error", extra={"error": str(e)})
        raise
    audit(actor=approver, action="reject_write", reason=reason,
          target_ids=[write_id], extra={"status": "rejected"})
    return pending.model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()
