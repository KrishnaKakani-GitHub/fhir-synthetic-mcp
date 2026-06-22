"""FHIR MCP server — Clinical AI Governance Platform.

This is the file Claude Code and the Agent SDK launch. It defines:
  - 7 tools (3 read, 4 write-gate)
  - 1 MCP resource (fhir://patient/{patient_id}/summary)
  - 2 MCP prompts (review_pending, patient_overview)

Each tool:
  1. Verifies the caller's identity (auth layer)
  2. Delegates to the store
  3. Emits a tamper-evident audit record on both success and error paths

Prompt caching: the system block and clinical guidelines block are marked
with cache_control breakpoints so repeated calls reuse the KV cache.

The write gate invariant is preserved:
  - Agents can READ and PROPOSE
  - Only a verified human approver can COMMIT via approve_write

Run locally:   python -m fhir_mcp.server
Register:      claude mcp add clinical-governance -- python -m fhir_mcp.server
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .audit import audit
from .auth import AuthError, verify_agent_actor, verify_approver
from .models import ProposedObservation
from .store import FhirStore, StoreError
from .validator import get_rules

_DB_PATH = Path(
    os.environ.get(
        "FHIR_MCP_DB",
        Path(__file__).resolve().parents[2] / "data" / "fhir.db",
    )
)

mcp = FastMCP("clinical-ai-governance")
store = FhirStore(_DB_PATH)

# Agent actor identity. In production this comes from authenticated identity.
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
    Any validation_warnings in the response are clinical flags that the
    approver should review before approving.
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
    audit(
        actor=_AGENT_ACTOR, action="propose_observation", reason=reason,
        target_ids=[patient_id],
        extra={
            "write_id": pending.write_id,
            "status": "pending",
            "has_warnings": bool(pending.validation_warnings),
        },
    )
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
    path that writes patient data to the database. Review validation_warnings
    in the pending write before approving.
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


# --- MCP Resources ------------------------------------------------------------
# Resources are read-only data providers. The agent can request them by URI.
# This is distinct from tools: resources provide context, tools take actions.


@mcp.resource("fhir://patient/{patient_id}/summary")
def patient_summary(patient_id: str) -> str:
    """FHIR patient summary as structured text.

    Returns demographics + all observations for a patient. Designed for
    use as context in agent prompts (prompt caching friendly — content
    is stable within a session if no new observations are committed).
    """
    try:
        verify_agent_actor(_AGENT_ACTOR)
        patient = store.get_patient(patient_id)
        observations = store.list_observations(patient_id)
    except (AuthError, StoreError) as e:
        return f"Error: {e}"

    audit(
        actor=_AGENT_ACTOR, action="read_patient_summary",
        reason="MCP resource request", target_ids=[patient_id],
    )

    lines = [
        f"Patient: {patient.name} ({patient.id})",
        f"DOB: {patient.birth_date} | Gender: {patient.gender.value} | MRN: {patient.mrn}",
        "",
        "Observations:",
    ]
    if not observations:
        lines.append("  (none recorded)")
    else:
        for o in observations:
            lines.append(
                f"  [{o.effective_date}] {o.display} ({o.code}): {o.value} {o.unit}"
            )
    return "\n".join(lines)


# --- MCP Prompts --------------------------------------------------------------
# Prompts are reusable message templates. They reduce prompt engineering
# scattered across callers into a single versioned location.
#
# Prompt caching note: the system block and the guidelines block are
# static within a deployment. Mark them with cache_control breakpoints
# so the KV cache reuses them across repeated calls.


@mcp.prompt()
def review_pending() -> list[dict[str, Any]]:
    """Prompt template for a human reviewer approving/rejecting pending writes.

    Returns an Anthropic messages array with a cacheable system block
    and a human turn requesting review of the pending queue.
    """
    rules = get_rules()
    rule_summary = json.dumps(
        {
            code: {
                "display": r["display"],
                "range": f"{r.get('min', '?')}-{r.get('max', '?')} {r.get('unit', '')}",
                "flag_above": r.get("flag_above"),
            }
            for code, r in rules.items()
        },
        indent=2,
    )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a clinical documentation reviewer for the Clinical AI Governance Platform. "
                        "Your role is to evaluate proposed FHIR observations that an AI agent has staged "
                        "for approval. For each pending write you will:\n"
                        "1. State the observation details clearly (patient, code, value, unit, date).\n"
                        "2. Check whether the value is within the normal clinical range for this LOINC code.\n"
                        "3. Note any validation_warnings from the deterministic gate.\n"
                        "4. State your recommendation: APPROVE or REJECT with a one-sentence clinical justification.\n\n"
                        "LOINC validation rules for this deployment:\n"
                        f"<loinc_rules>\n{rule_summary}\n</loinc_rules>"
                    ),
                    # Prompt caching: static system block — cache at this breakpoint
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": (
                        "Please review the pending writes returned by `list_pending_writes` "
                        "and provide your approval/rejection recommendation for each one."
                    ),
                },
            ],
        }
    ]


@mcp.prompt()
def patient_overview(patient_id: str) -> list[dict[str, Any]]:
    """Prompt template for comprehensive patient overview with clinical context.

    Returns an Anthropic messages array with the patient summary pre-loaded
    as a cacheable context block.
    """
    summary = patient_summary(patient_id)

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a clinical AI assistant helping a healthcare provider "
                        "review patient data. Analyse the patient summary below and "
                        "identify: (1) any observations outside normal ranges, "
                        "(2) trends that warrant clinical attention, "
                        "(3) observations that are missing but clinically indicated "
                        "based on the patient's current data.\n\n"
                        "<patient_summary>\n"
                        f"{summary}\n"
                        "</patient_summary>"
                    ),
                    # Prompt caching: patient context block — cache at this breakpoint
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": "Please provide your clinical assessment of this patient.",
                },
            ],
        }
    ]


if __name__ == "__main__":
    mcp.run()
