"""ClinicalOrchestrator — 5-agent pipeline with parallel Evidence and streaming.

Architecture (Day 11):

  Intake Agent     — Nemotron Parse → NLP → patient entities
  Evidence Agent   — RAG + ClinicalTrials.gov IN PARALLEL (asyncio.gather)
  Reasoning Agent  — Extended thinking, synthesizes patient + evidence
  Critic Agent     — Adversarial peer review of every proposal
  Compliance Agent — Deterministic LOINC gate + DUA + propose_observation
                           ↓
                      human gate (approve_write)

Key features:
  Parallel Evidence:  asyncio.gather() runs RAG + trials simultaneously,
                      eliminating the sequential bottleneck on the most
                      expensive retrieval stage.

  Streaming:          run_workflow_stream() is an async generator that yields
                      WorkflowEvent dicts at each stage transition. Clinicians
                      see real-time progress instead of blocking on completion.

  Critic Agent:       Every proposal is adversarially reviewed before the
                      human gate. Human sees proposal + critique together.
                      This is automated clinical peer review.

  Backward compat:    run_workflow() is unchanged. Streaming is additive.

PHI NOTE: session IDs and proposal counts are logged. No PHI in logs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator

from .hooks import AuditHook, WriteGateHook
from .subagents import (
    COMPLIANCE_SUBAGENT,
    CRITIC_SUBAGENT,
    EVIDENCE_RAG_SUBAGENT,
    EVIDENCE_TRIALS_SUBAGENT,
    INTAKE_SUBAGENT,
    REASONING_SUBAGENT,
    SubagentConfig,
)

_logger = logging.getLogger("clinical_agent.orchestrator")

_MCP_SERVER_CMD = (
    os.environ.get("FHIR_MCP_CMD", "python -m fhir_mcp.server")
    .split()
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class WorkflowEvent:
    """A streaming progress event emitted by run_workflow_stream()."""

    def __init__(
        self,
        stage: str,
        status: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.status = status  # starting | complete | error
        self.data = data or {}
        self.timestamp_ms = round(time.monotonic() * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "data": self.data,
            "timestamp_ms": self.timestamp_ms,
        }


class WorkflowResult:
    """Result of a complete 5-agent clinical workflow run."""

    def __init__(
        self,
        patient_id: str,
        session_id: str | None,
        proposals: list[dict[str, Any]],
        guidelines: list[dict[str, Any]],
        trials: list[dict[str, Any]],
        critique: dict[str, Any],
        metrics: dict[str, Any],
        used_extended_thinking: bool = False,
    ) -> None:
        self.patient_id = patient_id
        self.session_id = session_id
        self.proposals = proposals
        self.guidelines = guidelines
        self.trials = trials
        self.critique = critique
        self.metrics = metrics
        self.used_extended_thinking = used_extended_thinking

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "proposals": self.proposals,
            "guidelines": self.guidelines,
            "trials": self.trials,
            "critique": self.critique,
            "metrics": self.metrics,
            "used_extended_thinking": self.used_extended_thinking,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ClinicalOrchestrator:
    """5-agent clinical workflow orchestrator.

    Usage (blocking)::

        import asyncio
        from clinical_agent.orchestrator import ClinicalOrchestrator

        async def main():
            orch = ClinicalOrchestrator(actor="orchestrator:prod")
            result = await orch.run_workflow(patient_id="pat-001")
            print(result.to_dict())

        asyncio.run(main())

    Usage (streaming)::

        async def main():
            orch = ClinicalOrchestrator()
            async for event in orch.run_workflow_stream(patient_id="pat-001"):
                print(event.stage, event.status, event.data)
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
    # Public API: blocking
    # ------------------------------------------------------------------

    async def run_workflow(
        self,
        patient_id: str,
        session_id: str | None = None,
        document_source: str | None = None,
    ) -> WorkflowResult:
        """Run the full 5-agent pipeline for one patient.

        Args:
            patient_id:       The patient to analyse.
            session_id:       Resume a previous session (optional).
            document_source:  Path/URL to a clinical document to parse
                              via Nemotron Parse (optional).

        Returns:
            WorkflowResult with proposals, guidelines, trials, critique, metrics.
        """
        events: list[WorkflowEvent] = []
        result: WorkflowResult | None = None

        async for event in self.run_workflow_stream(
            patient_id=patient_id,
            session_id=session_id,
            document_source=document_source,
        ):
            events.append(event)
            if event.stage == "complete":
                result = event.data.get("result")

        if result is None:
            raise RuntimeError("Workflow did not complete successfully")
        return result

    # ------------------------------------------------------------------
    # Public API: streaming
    # ------------------------------------------------------------------

    async def run_workflow_stream(
        self,
        patient_id: str,
        session_id: str | None = None,
        document_source: str | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Streaming 5-agent pipeline. Yields WorkflowEvent at each stage.

        Stages emitted:
          intake → evidence (parallel RAG + trials) → reasoning → critic
          → compliance → complete

        Each stage yields two events: status=starting then status=complete.
        On error: status=error with error detail in data.
        """
        start = time.monotonic()
        metrics: dict[str, Any] = {}

        # --- Stage 1: Intake -------------------------------------------
        yield WorkflowEvent("intake", "starting", {"patient_id": patient_id})
        t0 = time.monotonic()
        patient_context = await self._run_intake(patient_id, session_id, document_source)
        metrics["intake_ms"] = round((time.monotonic() - t0) * 1000, 1)
        yield WorkflowEvent("intake", "complete", {"elapsed_ms": metrics["intake_ms"]})

        # --- Stage 2: Evidence (parallel RAG + trials) -----------------
        yield WorkflowEvent("evidence", "starting", {"parallel": True})
        t0 = time.monotonic()
        rag_context, trials_context = await asyncio.gather(
            self._run_rag(patient_context),
            self._run_trials(patient_context),
        )
        metrics["evidence_parallel_ms"] = round((time.monotonic() - t0) * 1000, 1)
        yield WorkflowEvent(
            "evidence", "complete",
            {
                "elapsed_ms": metrics["evidence_parallel_ms"],
                "parallel": True,
                "guideline_count": len(self._extract_guidelines(rag_context)),
                "trial_count": len(self._extract_trials(trials_context)),
            },
        )

        # --- Stage 3: Reasoning (extended thinking) --------------------
        yield WorkflowEvent("reasoning", "starting", {"extended_thinking": True})
        t0 = time.monotonic()
        reasoning_context = await self._run_reasoning(
            patient_context, rag_context, trials_context
        )
        metrics["reasoning_ms"] = round((time.monotonic() - t0) * 1000, 1)
        proposed = self._extract_proposed_observations(reasoning_context)
        yield WorkflowEvent(
            "reasoning", "complete",
            {"elapsed_ms": metrics["reasoning_ms"], "proposal_count": len(proposed)},
        )

        # --- Stage 4: Critic (adversarial peer review) -----------------
        yield WorkflowEvent("critic", "starting", {"proposal_count": len(proposed)})
        t0 = time.monotonic()
        critique_context = await self._run_critic(reasoning_context)
        metrics["critic_ms"] = round((time.monotonic() - t0) * 1000, 1)
        critique = self._parse_json(critique_context)
        verdict = critique.get("overall_verdict", "challenged")
        yield WorkflowEvent(
            "critic", "complete",
            {"elapsed_ms": metrics["critic_ms"], "verdict": verdict},
        )

        # --- Stage 5: Compliance (deterministic gate + propose) --------
        yield WorkflowEvent("compliance", "starting", {"critic_verdict": verdict})
        t0 = time.monotonic()
        compliance_context = await self._run_compliance(
            reasoning_context, critique_context
        )
        metrics["compliance_ms"] = round((time.monotonic() - t0) * 1000, 1)
        proposals = self._extract_staged_proposals(compliance_context)
        yield WorkflowEvent(
            "compliance", "complete",
            {"elapsed_ms": metrics["compliance_ms"], "staged_count": len(proposals)},
        )

        # --- Complete --------------------------------------------------
        metrics["workflow_total_ms"] = round((time.monotonic() - start) * 1000, 1)
        metrics.update(self._audit_hook.metrics())

        _logger.info(
            "Workflow complete patient=%s proposals=%d critic=%s total_ms=%s",
            patient_id, len(proposals), verdict, metrics["workflow_total_ms"],
        )

        result = WorkflowResult(
            patient_id=patient_id,
            session_id=session_id,
            proposals=proposals,
            guidelines=self._extract_guidelines(rag_context),
            trials=self._extract_trials(trials_context),
            critique=critique,
            metrics=metrics,
            used_extended_thinking=True,
        )
        yield WorkflowEvent("complete", "complete", {"result": result})

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    async def _run_intake(
        self,
        patient_id: str,
        session_id: str | None,
        document_source: str | None,
    ) -> str:
        prompt = f"Fetch all available data for patient {patient_id}."
        if document_source:
            prompt += (
                f" Also parse this clinical document: {document_source}. "
                "Use parse_clinical_document then cross-reference with the patient record."
            )
        prompt += " Return demographics and all observations as structured JSON."
        return await self._query_subagent(
            subagent=INTAKE_SUBAGENT,
            prompt=prompt,
            session_id=session_id,
        )

    async def _run_rag(self, patient_context: str) -> str:
        return await self._query_subagent(
            subagent=EVIDENCE_RAG_SUBAGENT,
            prompt=(
                "Find relevant clinical guidelines for this patient:\n\n"
                f"<patient_context>\n{patient_context}\n</patient_context>"
            ),
        )

    async def _run_trials(self, patient_context: str) -> str:
        return await self._query_subagent(
            subagent=EVIDENCE_TRIALS_SUBAGENT,
            prompt=(
                "Find recruiting clinical trials for conditions in this patient context:\n\n"
                f"<patient_context>\n{patient_context}\n</patient_context>"
            ),
        )

    async def _run_reasoning(
        self,
        patient_context: str,
        rag_context: str,
        trials_context: str,
    ) -> str:
        return await self._query_subagent(
            subagent=REASONING_SUBAGENT,
            prompt=(
                "Synthesize the following patient data and evidence into "
                "structured clinical reasoning and proposed observations.\n\n"
                f"<patient_context>\n{patient_context}\n</patient_context>\n\n"
                f"<guidelines>\n{rag_context}\n</guidelines>\n\n"
                f"<clinical_trials>\n{trials_context}\n</clinical_trials>"
            ),
        )

    async def _run_critic(self, reasoning_context: str) -> str:
        return await self._query_subagent(
            subagent=CRITIC_SUBAGENT,
            prompt=(
                "Perform adversarial peer review on these clinical proposals. "
                "Challenge each one across all five dimensions.\n\n"
                f"<reasoning_and_proposals>\n{reasoning_context}\n</reasoning_and_proposals>"
            ),
        )

    async def _run_compliance(
        self,
        reasoning_context: str,
        critique_context: str,
    ) -> str:
        return await self._query_subagent(
            subagent=COMPLIANCE_SUBAGENT,
            prompt=(
                "Stage the proposals that passed critic review for human approval.\n\n"
                "Rules:\n"
                "- Only stage proposals where critic verdict is approved or challenged.\n"
                "- Include the critique summary in every propose_observation reason field.\n"
                "- Do NOT stage rejected proposals.\n\n"
                f"<proposals>\n{reasoning_context}\n</proposals>\n\n"
                f"<peer_review>\n{critique_context}\n</peer_review>"
            ),
        )

    # ------------------------------------------------------------------
    # Agent SDK wrapper
    # ------------------------------------------------------------------

    async def _query_subagent(
        self,
        subagent: SubagentConfig,
        prompt: str,
        session_id: str | None = None,
    ) -> str:
        """Query a subagent via the Claude Agent SDK.

        Falls back to stub responses in test/CI environments where the SDK
        is not installed or ANTHROPIC_API_KEY is not set.
        """
        try:
            import claude  # type: ignore[import-untyped]
        except ImportError:
            _logger.warning(
                "claude-agent-sdk not installed; returning stub for subagent=%s",
                subagent.name,
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
            "mcp_servers": mcp_config if subagent.allowed_tools else {},
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
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {}

    def _extract_guidelines(self, rag_context: str) -> list[dict[str, Any]]:
        return self._parse_json(rag_context).get("guidelines", [])

    def _extract_trials(self, trials_context: str) -> list[dict[str, Any]]:
        return self._parse_json(trials_context).get("trials", [])

    def _extract_proposed_observations(
        self, reasoning_context: str
    ) -> list[dict[str, Any]]:
        return self._parse_json(reasoning_context).get("proposed_observations", [])

    def _extract_staged_proposals(
        self, compliance_context: str
    ) -> list[dict[str, Any]]:
        return self._parse_json(compliance_context).get("staged", [])

    # ------------------------------------------------------------------
    # Stubs for test/CI
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_response(subagent_name: str) -> str:
        stubs: dict[str, str] = {
            "intake": json.dumps({
                "patient_id": "pat-001",
                "demographics": {"name": "Test Patient"},
                "observations": [],
                "parsed_document": None,
                "entity_count": 0,
                "error": None,
            }),
            "evidence:rag": json.dumps({
                "guidelines": [
                    {"id": "gl-001", "title": "JNC 8",
                     "relevance": "Hypertension management",
                     "key_thresholds": {"initiation_sbp": 140}}
                ],
                "search_queries": ["hypertension"],
            }),
            "evidence:trials": json.dumps({"trials": []}),
            "reasoning": json.dumps({
                "clinical_summary": "Test patient with no observations.",
                "risk_level": "low",
                "proposed_observations": [],
                "reasoning_notes": "Stub reasoning.",
            }),
            "critic": json.dumps({
                "overall_verdict": "approved",
                "overall_rationale": "No proposals to critique.",
                "critiques": [],
                "peer_review_summary": "Nothing to review.",
            }),
            "compliance": json.dumps({"staged": [], "skipped": []}),
        }
        return stubs.get(subagent_name, "{}")
