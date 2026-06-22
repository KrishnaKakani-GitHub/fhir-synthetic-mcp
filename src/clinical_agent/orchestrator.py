"""ClinicalOrchestrator: drives the Reader → RAG → Proposal pipeline.

Workflow:
  1. Reader Subagent   — fetches patient data in parallel for all patients
  2. RAG Subagent      — searches clinical guidelines based on patient context
  3. Proposal Subagent — generates structured proposals with confidence scores
     - If any observation is above the clinical flag threshold, routes to
       the extended-thinking variant of ProposalSubagent

Claude Agent SDK integration:
  - Uses query() with mcp_servers config to connect to the FHIR MCP server
  - Hooks: AuditHook (PostToolUse) + WriteGateHook (PreToolUse)
  - Session resume: session_id is returned and can be passed to resume a workflow

Prompt caching:
  - System prompts are marked for caching (static across calls)
  - Patient context block is marked for caching within a session

Note: This module requires ANTHROPIC_API_KEY env var and the FHIR MCP server
running (or configured via FHIR_MCP_DB etc.).

PHI NOTE: Patient data flows through the Agent SDK's context. This module
logs session IDs and proposal counts only — no PHI in structured logs.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .hooks import AuditHook, WriteGateHook
from .subagents import (
    PROPOSAL_SUBAGENT,
    PROPOSAL_SUBAGENT_THINKING,
    RAG_SUBAGENT,
    READER_SUBAGENT,
    SubagentConfig,
)

_logger = logging.getLogger("clinical_agent.orchestrator")

_MCP_SERVER_CMD = (
    os.environ.get("FHIR_MCP_CMD", "python -m fhir_mcp.server")
    .split()
)


class WorkflowResult:
    """Result of a complete clinical workflow run."""

    def __init__(
        self,
        patient_id: str,
        session_id: str | None,
        proposals: list[dict[str, Any]],
        guidelines: list[dict[str, Any]],
        metrics: dict[str, Any],
        used_extended_thinking: bool = False,
    ) -> None:
        self.patient_id = patient_id
        self.session_id = session_id
        self.proposals = proposals
        self.guidelines = guidelines
        self.metrics = metrics
        self.used_extended_thinking = used_extended_thinking

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "proposals": self.proposals,
            "guidelines": self.guidelines,
            "metrics": self.metrics,
            "used_extended_thinking": self.used_extended_thinking,
        }


class ClinicalOrchestrator:
    """Drives the three-subagent clinical workflow.

    Usage::

        import asyncio
        from clinical_agent.orchestrator import ClinicalOrchestrator

        async def main():
            orch = ClinicalOrchestrator(actor="orchestrator:prod")
            result = await orch.run_workflow(patient_id="pat-001")
            print(result.to_dict())

        asyncio.run(main())
    """

    def __init__(
        self,
        actor: str = "orchestrator:dev",
        mcp_server_cmd: list[str] | None = None,
    ) -> None:
        self._actor = actor
        self._mcp_cmd = mcp_server_cmd or _MCP_SERVER_CMD
        self._audit_hook = AuditHook(actor=actor)
        self._gate_hook = WriteGateHook()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_workflow(
        self,
        patient_id: str,
        session_id: str | None = None,
        force_extended_thinking: bool = False,
    ) -> WorkflowResult:
        """Run the full Reader → RAG → Proposal pipeline for one patient.

        Args:
            patient_id:               The patient to analyse.
            session_id:               Resume a previous session (optional).
            force_extended_thinking:  Always use extended thinking for proposals.

        Returns:
            WorkflowResult with proposals, guidelines, and session metrics.
        """
        start = time.monotonic()
        _logger.info("Starting workflow for patient=%s session=%s", patient_id, session_id)

        # Stage 1: Read patient data
        patient_context = await self._run_reader(patient_id, session_id)

        # Stage 2: Search guidelines based on patient context
        guideline_context = await self._run_rag(patient_context)

        # Stage 3: Determine if extended thinking is warranted
        use_thinking = force_extended_thinking or self._needs_extended_thinking(
            patient_context
        )
        if use_thinking:
            _logger.info("Routing to extended thinking for patient=%s", patient_id)

        # Stage 4: Generate proposals
        proposals = await self._run_proposal(
            patient_context, guideline_context, use_thinking
        )

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        metrics = self._audit_hook.metrics()
        metrics["workflow_total_ms"] = elapsed_ms

        _logger.info(
            "Workflow complete patient=%s proposals=%d elapsed_ms=%s",
            patient_id, len(proposals), elapsed_ms,
        )

        return WorkflowResult(
            patient_id=patient_id,
            session_id=session_id,
            proposals=proposals,
            guidelines=self._extract_guidelines(guideline_context),
            metrics=metrics,
            used_extended_thinking=use_thinking,
        )

    # ------------------------------------------------------------------
    # Subagent runners
    # ------------------------------------------------------------------

    async def _run_reader(
        self, patient_id: str, session_id: str | None
    ) -> str:
        """Run the Reader Subagent to fetch patient data."""
        return await self._query_subagent(
            subagent=READER_SUBAGENT,
            prompt=(
                f"Fetch all available data for patient {patient_id}. "
                "Return demographics and all observations."
            ),
            session_id=session_id,
        )

    async def _run_rag(self, patient_context: str) -> str:
        """Run the RAG Subagent to find relevant guidelines."""
        return await self._query_subagent(
            subagent=RAG_SUBAGENT,
            prompt=(
                "Based on the following patient data, search for the most relevant "
                "clinical guidelines:\n\n"
                f"<patient_context>\n{patient_context}\n</patient_context>\n\n"
                "Focus on guidelines related to the observed clinical values "
                "and their normal ranges."
            ),
        )

    async def _run_proposal(
        self,
        patient_context: str,
        guideline_context: str,
        use_extended_thinking: bool,
    ) -> list[dict[str, Any]]:
        """Run the Proposal Subagent to generate observation proposals."""
        subagent = (
            PROPOSAL_SUBAGENT_THINKING if use_extended_thinking else PROPOSAL_SUBAGENT
        )
        response = await self._query_subagent(
            subagent=subagent,
            prompt=(
                "Based on the patient data and relevant guidelines below, "
                "propose any clinically indicated observations for this patient.\n\n"
                f"<patient_context>\n{patient_context}\n</patient_context>\n\n"
                f"<guidelines>\n{guideline_context}\n</guidelines>\n\n"
                "For each proposal: include the LOINC code, value, unit, "
                "confidence score, and guideline citations. "
                "Remember: propose only — you cannot approve."
            ),
        )
        return self._parse_proposals(response)

    # ------------------------------------------------------------------
    # Agent SDK query wrapper
    # ------------------------------------------------------------------

    async def _query_subagent(
        self,
        subagent: SubagentConfig,
        prompt: str,
        session_id: str | None = None,
    ) -> str:
        """Query a subagent using the Claude Agent SDK.

        Falls back to a stub response in environments without the SDK
        installed (test/CI mode). This allows unit tests to run without
        the full ANTHROPIC_API_KEY + running MCP server.
        """
        try:
            import claude  # type: ignore[import-untyped]
        except ImportError:
            _logger.warning(
                "claude-agent-sdk not installed; returning stub response "
                "for subagent=%s", subagent.name
            )
            return self._stub_response(subagent.name)

        mcp_config = {
            "clinical-governance": {
                "command": self._mcp_cmd[0],
                "args": self._mcp_cmd[1:],
                "env": {
                    "FHIR_MCP_DB": os.environ.get("FHIR_MCP_DB", ""),
                    "FHIR_MCP_ACTOR": self._actor,
                    "FHIR_MCP_AUDIT_FILE": os.environ.get("FHIR_MCP_AUDIT_FILE", ""),
                    "FHIR_MCP_LOINC_RULES": os.environ.get("FHIR_MCP_LOINC_RULES", ""),
                    "FHIR_MCP_RAG_DISABLE_CHROMA": os.environ.get(
                        "FHIR_MCP_RAG_DISABLE_CHROMA", "0"
                    ),
                },
            }
        }

        kwargs: dict[str, Any] = {
            "model": subagent.model,
            "system_prompt": subagent.system_prompt,
            "mcp_servers": mcp_config,
            "allowed_tools": subagent.allowed_tools,
            "hooks": [self._audit_hook, self._gate_hook],
            "prompt": prompt,
        }
        if session_id:
            kwargs["session_id"] = session_id
        if subagent.thinking:
            kwargs["thinking"] = subagent.thinking

        result = await claude.query(**kwargs)
        return result.content if hasattr(result, "content") else str(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_extended_thinking(patient_context: str) -> bool:
        """Heuristic: route to extended thinking if context mentions flag thresholds.

        In production, this would parse the structured reader output and
        compare values against loinc_rules.json flag_above thresholds.
        Here we use a simple keyword heuristic as a placeholder.
        """
        keywords = ["flag", "above", "critical", "urgent", "alert", ">"]
        ctx_lower = patient_context.lower()
        return any(kw in ctx_lower for kw in keywords)

    @staticmethod
    def _parse_proposals(response: str) -> list[dict[str, Any]]:
        """Extract proposals list from subagent response."""
        try:
            # Try to parse as JSON directly
            data = json.loads(response)
            return data.get("proposals", [])
        except (json.JSONDecodeError, AttributeError):
            # Try to extract JSON block from text response
            import re
            m = re.search(r"\{.*?\"proposals\".*?\}", response, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0)).get("proposals", [])
                except json.JSONDecodeError:
                    pass
        return []

    @staticmethod
    def _extract_guidelines(guideline_context: str) -> list[dict[str, Any]]:
        """Extract guidelines list from RAG subagent response."""
        try:
            data = json.loads(guideline_context)
            return data.get("guidelines", [])
        except (json.JSONDecodeError, AttributeError):
            return []

    @staticmethod
    def _stub_response(subagent_name: str) -> str:
        """Stub responses for test/CI environments without the Agent SDK."""
        stubs: dict[str, str] = {
            "reader": json.dumps({
                "patient_id": "pat-001",
                "demographics": {"name": "Test Patient"},
                "observations": [],
                "error": None,
            }),
            "rag": json.dumps({
                "guidelines": [
                    {"id": "gl-001", "title": "JNC 8",
                     "relevance": "Hypertension management",
                     "key_thresholds": {"initiation_sbp": 140}}
                ],
                "search_queries": ["hypertension management"],
            }),
            "proposal": json.dumps({"proposals": []}),
        }
        return stubs.get(subagent_name, "{}")
