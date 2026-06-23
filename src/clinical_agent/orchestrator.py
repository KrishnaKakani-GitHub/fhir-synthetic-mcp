"""ClinicalOrchestrator — dynamic 7-agent pipeline with checkpointing.

Architecture (Day 12):

  Stage 0  Planner      — reads task, writes WorkflowPlan (dynamic)
  Stage 1  Intake       — Nemotron Parse → NLP → patient entities
  Stage 2  Evidence     — RAG + ClinicalTrials.gov IN PARALLEL
  Stage 3  Reasoning    — Extended thinking, synthesizes patient + evidence
  Stage 3.5 Refuter     — Adversarial attack: breaks proposals before Critic
  Stage 4  Critic       — Thesis/antithesis → synthesis (Reasoning vs Refuter)
  Stage 5  Compliance   — Deterministic LOINC gate + DUA + propose_observation
                                ↓
                           human gate (approve_write)

PHI NOTE: session IDs and proposal counts are logged. No PHI in logs.
Checkpoint files follow the same access controls as FHIR_MCP_DB.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncGenerator

from .checkpoint import CheckpointStore
from .hooks import AuditHook, WriteGateHook
from .subagents import (
    COMPLIANCE_SUBAGENT,
    CRITIC_SUBAGENT,
    EVIDENCE_RAG_SUBAGENT,
    EVIDENCE_TRIALS_SUBAGENT,
    INTAKE_SUBAGENT,
    PLANNER_SUBAGENT,
    REASONING_SUBAGENT,
    REFUTER_SUBAGENT,
    SubagentConfig,
)

_logger = logging.getLogger("clinical_agent.orchestrator")

_MCP_SERVER_CMD = (
    os.environ.get("FHIR_MCP_CMD", "python -m fhir_mcp.server")
    .split()
)


# ---------------------------------------------------------------------------
# WorkflowPlan
# ---------------------------------------------------------------------------


class WorkflowPlan:
    """Parsed output of the Planner Agent — controls which stages run."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        stages = raw.get("stages", {})
        evidence = stages.get("evidence", {})
        reasoning = stages.get("reasoning", {})
        refuter = stages.get("refuter", {})
        critic = stages.get("critic", {})

        self.workflow_id: str = raw.get("workflow_id", str(uuid.uuid4()))
        self.task_type: str = raw.get("task_type", "full_workup")
        self.fast_path: bool = raw.get("fast_path", False)
        self.plan_rationale: str = raw.get("plan_rationale", "")

        self.run_intake: bool = stages.get("intake", True)
        self.run_rag: bool = evidence.get("rag", True)
        self.run_trials: bool = evidence.get("trials", True)
        self.run_reasoning: bool = reasoning.get("run", True) and not self.fast_path
        self.extended_thinking: bool = reasoning.get("extended_thinking", False)
        self.reasoning_budget: int = reasoning.get("budget_tokens", 8000)
        self.run_refuter: bool = refuter.get("run", True) and self.run_reasoning
        self.run_critic: bool = critic.get("run", True) and self.run_reasoning
        self.run_compliance: bool = stages.get("compliance", True)

    def to_dict(self) -> dict[str, Any]:
        return self._raw

    @classmethod
    def default(cls) -> "WorkflowPlan":
        """Full-workup plan used as fallback if Planner fails."""
        return cls({
            "workflow_id": str(uuid.uuid4()),
            "task_type": "full_workup",
            "stages": {
                "intake": True,
                "evidence": {"rag": True, "trials": True},
                "reasoning": {"run": True, "extended_thinking": True, "budget_tokens": 8000},
                "refuter": {"run": True},
                "critic": {"run": True},
                "compliance": True,
            },
            "fast_path": False,
            "plan_rationale": "Default full-workup plan (Planner fallback)",
        })


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class WorkflowEvent:
    """Streaming progress event emitted by run_workflow_stream()."""

    def __init__(
        self,
        stage: str,
        status: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.status = status
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
    """Result of a complete 7-agent dynamic clinical workflow run."""

    def __init__(
        self,
        patient_id: str,
        session_id: str | None,
        plan: WorkflowPlan,
        proposals: list[dict[str, Any]],
        guidelines: list[dict[str, Any]],
        trials: list[dict[str, Any]],
        critique: dict[str, Any],
        refuter_attacks: dict[str, Any],
        metrics: dict[str, Any],
        checkpoint_id: str | None = None,
    ) -> None:
        self.patient_id = patient_id
        self.session_id = session_id
        self.plan = plan
        self.proposals = proposals
        self.guidelines = guidelines
        self.trials = trials
        self.critique = critique
        self.refuter_attacks = refuter_attacks
        self.metrics = metrics
        self.checkpoint_id = checkpoint_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "plan": self.plan.to_dict(),
            "proposals": self.proposals,
            "guidelines": self.guidelines,
            "trials": self.trials,
            "critique": self.critique,
            "refuter_attacks": self.refuter_attacks,
            "metrics": self.metrics,
            "checkpoint_id": self.checkpoint_id,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ClinicalOrchestrator:
    """Dynamic 7-agent clinical workflow orchestrator with checkpointing.

    Usage (blocking)::

        import asyncio
        from clinical_agent.orchestrator import ClinicalOrchestrator

        async def main():
            orch = ClinicalOrchestrator(actor="orchestrator:prod")
            result = await orch.run_workflow(patient_id="pat-001")
            print(result.to_dict())

    Usage (resume)::

        result = await orch.run_workflow(
            patient_id="pat-001",
            checkpoint_id="<id from previous run>",
        )

    Usage (streaming)::

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
    # Public API
    # ------------------------------------------------------------------

    async def run_workflow(
        self,
        patient_id: str,
        session_id: str | None = None,
        document_source: str | None = None,
        checkpoint_id: str | None = None,
    ) -> WorkflowResult:
        result: WorkflowResult | None = None
        async for event in self.run_workflow_stream(
            patient_id=patient_id,
            session_id=session_id,
            document_source=document_source,
            checkpoint_id=checkpoint_id,
        ):
            if event.stage == "complete":
                result = event.data.get("result")
        if result is None:
            raise RuntimeError("Workflow did not complete successfully")
        return result

    async def run_workflow_stream(
        self,
        patient_id: str,
        session_id: str | None = None,
        document_source: str | None = None,
        checkpoint_id: str | None = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        start = time.monotonic()
        metrics: dict[str, Any] = {}

        ckpt = self._init_checkpoint(patient_id, checkpoint_id)
        completed = ckpt.completed_stages()

        # Stage 0: Plan
        if "plan" in completed:
            plan = WorkflowPlan(json.loads(completed["plan"]))
            yield WorkflowEvent("plan", "skipped", {"task_type": plan.task_type, "resumed": True})
        else:
            yield WorkflowEvent("plan", "starting", {"patient_id": patient_id})
            t0 = time.monotonic()
            plan = await self._run_planner(patient_id, document_source)
            metrics["plan_ms"] = round((time.monotonic() - t0) * 1000, 1)
            ckpt.save_plan(plan.to_dict())
            ckpt.save_stage("plan", json.dumps(plan.to_dict()))
            yield WorkflowEvent("plan", "complete", {
                "task_type": plan.task_type, "fast_path": plan.fast_path,
                "rationale": plan.plan_rationale, "elapsed_ms": metrics["plan_ms"],
            })

        # Stage 1: Intake
        if "intake" in completed:
            patient_context = completed["intake"]
            yield WorkflowEvent("intake", "skipped", {"resumed": True})
        else:
            yield WorkflowEvent("intake", "starting", {"patient_id": patient_id})
            t0 = time.monotonic()
            patient_context = await self._run_intake(patient_id, session_id, document_source)
            metrics["intake_ms"] = round((time.monotonic() - t0) * 1000, 1)
            ckpt.save_stage("intake", patient_context)
            yield WorkflowEvent("intake", "complete", {"elapsed_ms": metrics.get("intake_ms")})

        # Stage 2: Evidence (parallel)
        rag_context = completed.get("evidence_rag", "{}")
        trials_context = completed.get("evidence_trials", "{}")

        if "evidence_rag" in completed and "evidence_trials" in completed:
            yield WorkflowEvent("evidence", "skipped", {"resumed": True})
        elif plan.fast_path:
            yield WorkflowEvent("evidence", "skipped", {"reason": "fast_path"})
        else:
            yield WorkflowEvent("evidence", "starting", {
                "parallel": True, "rag": plan.run_rag, "trials": plan.run_trials
            })
            t0 = time.monotonic()
            tasks = []
            if plan.run_rag and "evidence_rag" not in completed:
                tasks.append(("rag", self._run_rag(patient_context)))
            if plan.run_trials and "evidence_trials" not in completed:
                tasks.append(("trials", self._run_trials(patient_context)))
            if tasks:
                names, coros = zip(*tasks)
                results = await asyncio.gather(*coros)
                for name, output in zip(names, results):
                    if name == "rag":
                        rag_context = output
                        ckpt.save_stage("evidence_rag", rag_context)
                    else:
                        trials_context = output
                        ckpt.save_stage("evidence_trials", trials_context)
            metrics["evidence_parallel_ms"] = round((time.monotonic() - t0) * 1000, 1)
            yield WorkflowEvent("evidence", "complete", {
                "elapsed_ms": metrics["evidence_parallel_ms"],
                "guideline_count": len(self._extract_guidelines(rag_context)),
                "trial_count": len(self._extract_trials(trials_context)),
            })

        # Stage 3: Reasoning
        if "reasoning" in completed:
            reasoning_context = completed["reasoning"]
            yield WorkflowEvent("reasoning", "skipped", {"resumed": True})
        elif not plan.run_reasoning:
            reasoning_context = "{\"proposed_observations\": [], \"clinical_summary\": \"fast-path\"}"
            yield WorkflowEvent("reasoning", "skipped", {"reason": "not required by plan"})
        else:
            reasoning_subagent = self._reasoning_subagent_for_plan(plan)
            yield WorkflowEvent("reasoning", "starting", {
                "extended_thinking": plan.extended_thinking,
                "budget_tokens": plan.reasoning_budget,
            })
            t0 = time.monotonic()
            reasoning_context = await self._run_reasoning(
                patient_context, rag_context, trials_context, subagent=reasoning_subagent,
            )
            metrics["reasoning_ms"] = round((time.monotonic() - t0) * 1000, 1)
            ckpt.save_stage("reasoning", reasoning_context)
            yield WorkflowEvent("reasoning", "complete", {
                "elapsed_ms": metrics["reasoning_ms"],
                "proposal_count": len(self._extract_proposed_observations(reasoning_context)),
            })

        # Stage 3.5: Refuter
        if "refuter" in completed:
            refuter_context = completed["refuter"]
            yield WorkflowEvent("refuter", "skipped", {"resumed": True})
        elif not plan.run_refuter:
            refuter_context = "{\"refuter_verdict\": \"all_survived\", \"attacks\": []}"
            yield WorkflowEvent("refuter", "skipped", {"reason": "not required by plan"})
        else:
            yield WorkflowEvent("refuter", "starting", {})
            t0 = time.monotonic()
            refuter_context = await self._run_refuter(
                reasoning_context, rag_context, patient_context
            )
            metrics["refuter_ms"] = round((time.monotonic() - t0) * 1000, 1)
            ckpt.save_stage("refuter", refuter_context)
            refuter_data = self._parse_json(refuter_context)
            yield WorkflowEvent("refuter", "complete", {
                "elapsed_ms": metrics["refuter_ms"],
                "verdict": refuter_data.get("refuter_verdict", "unknown"),
                "attack_count": len(refuter_data.get("attacks", [])),
            })

        # Stage 4: Critic
        if "critic" in completed:
            critique_context = completed["critic"]
            yield WorkflowEvent("critic", "skipped", {"resumed": True})
        elif not plan.run_critic:
            critique_context = "{\"overall_verdict\": \"approved\", \"critiques\": []}"
            yield WorkflowEvent("critic", "skipped", {"reason": "not required by plan"})
        else:
            yield WorkflowEvent("critic", "starting", {})
            t0 = time.monotonic()
            critique_context = await self._run_critic(reasoning_context, refuter_context)
            metrics["critic_ms"] = round((time.monotonic() - t0) * 1000, 1)
            ckpt.save_stage("critic", critique_context)
            critique_data = self._parse_json(critique_context)
            yield WorkflowEvent("critic", "complete", {
                "elapsed_ms": metrics["critic_ms"],
                "verdict": critique_data.get("overall_verdict", "unknown"),
            })

        # Stage 5: Compliance
        if "compliance" in completed:
            compliance_context = completed["compliance"]
            yield WorkflowEvent("compliance", "skipped", {"resumed": True})
        else:
            yield WorkflowEvent("compliance", "starting", {})
            t0 = time.monotonic()
            compliance_context = await self._run_compliance(
                reasoning_context, refuter_context, critique_context
            )
            metrics["compliance_ms"] = round((time.monotonic() - t0) * 1000, 1)
            ckpt.save_stage("compliance", compliance_context)
            yield WorkflowEvent("compliance", "complete", {
                "elapsed_ms": metrics["compliance_ms"],
                "staged_count": len(self._extract_staged_proposals(compliance_context)),
            })

        # Complete
        metrics["workflow_total_ms"] = round((time.monotonic() - start) * 1000, 1)
        metrics.update(self._audit_hook.metrics())

        critique = self._parse_json(critique_context)
        refuter_attacks = self._parse_json(refuter_context)
        proposals = self._extract_staged_proposals(compliance_context)

        _logger.info(
            "Workflow complete patient=%s task=%s proposals=%d critic=%s total_ms=%s",
            patient_id, plan.task_type, len(proposals),
            critique.get("overall_verdict"), metrics["workflow_total_ms"],
        )

        ckpt.delete()

        result = WorkflowResult(
            patient_id=patient_id,
            session_id=session_id,
            plan=plan,
            proposals=proposals,
            guidelines=self._extract_guidelines(rag_context),
            trials=self._extract_trials(trials_context),
            critique=critique,
            refuter_attacks=refuter_attacks,
            metrics=metrics,
            checkpoint_id=ckpt.checkpoint_id,
        )
        yield WorkflowEvent("complete", "complete", {"result": result})

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _init_checkpoint(
        self, patient_id: str, checkpoint_id: str | None
    ) -> CheckpointStore:
        if checkpoint_id:
            try:
                store = CheckpointStore.load(checkpoint_id)
                _logger.info("Resuming from checkpoint=%s", checkpoint_id)
                return store
            except FileNotFoundError:
                _logger.warning("Checkpoint %s not found; starting fresh", checkpoint_id)
        return CheckpointStore(patient_id=patient_id, actor=self._actor)

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    async def _run_planner(
        self, patient_id: str, document_source: str | None
    ) -> WorkflowPlan:
        task_description = f"Patient: {patient_id}."
        if document_source:
            task_description += f" Clinical document: {document_source}."
        try:
            response = await self._query_subagent(
                subagent=PLANNER_SUBAGENT,
                prompt=(
                    "Decide the optimal workflow for this clinical task.\n\n"
                    f"Task: {task_description}\n\n"
                    "Return the WorkflowPlan JSON only."
                ),
            )
            plan_data = self._parse_json(response)
            if not plan_data:
                raise ValueError("Planner returned empty JSON")
            return WorkflowPlan(plan_data)
        except Exception as exc:
            _logger.warning("Planner failed (%s); using default plan", exc)
            return WorkflowPlan.default()

    async def _run_intake(
        self, patient_id: str, session_id: str | None, document_source: str | None,
    ) -> str:
        prompt = f"Fetch all available data for patient {patient_id}."
        if document_source:
            prompt += (
                f" Also parse: {document_source}. "
                "Use parse_clinical_document then cross-reference with the patient record."
            )
        prompt += " Return demographics and all observations as structured JSON."
        return await self._query_subagent(
            subagent=INTAKE_SUBAGENT, prompt=prompt, session_id=session_id,
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
                "Find recruiting clinical trials for conditions in this patient:\n\n"
                f"<patient_context>\n{patient_context}\n</patient_context>"
            ),
        )

    async def _run_reasoning(
        self,
        patient_context: str,
        rag_context: str,
        trials_context: str,
        subagent: SubagentConfig | None = None,
    ) -> str:
        return await self._query_subagent(
            subagent=subagent or REASONING_SUBAGENT,
            prompt=(
                "Synthesize the following into structured clinical reasoning and proposals.\n\n"
                f"<patient_context>\n{patient_context}\n</patient_context>\n\n"
                f"<guidelines>\n{rag_context}\n</guidelines>\n\n"
                f"<clinical_trials>\n{trials_context}\n</clinical_trials>"
            ),
        )

    async def _run_refuter(
        self, reasoning_context: str, rag_context: str, patient_context: str,
    ) -> str:
        return await self._query_subagent(
            subagent=REFUTER_SUBAGENT,
            prompt=(
                "Attack the following clinical proposals. "
                "Find every reason they might be wrong.\n\n"
                f"<proposals_from_reasoning>\n{reasoning_context}\n</proposals_from_reasoning>\n\n"
                f"<raw_evidence>\n{rag_context}\n</raw_evidence>\n\n"
                f"<patient_context>\n{patient_context}\n</patient_context>"
            ),
        )

    async def _run_critic(
        self, reasoning_context: str, refuter_context: str,
    ) -> str:
        return await self._query_subagent(
            subagent=CRITIC_SUBAGENT,
            prompt=(
                "Resolve the dialectic. Weigh the clinical reasoning against "
                "the adversarial attacks and produce a synthesis verdict.\n\n"
                f"<reasoning_thesis>\n{reasoning_context}\n</reasoning_thesis>\n\n"
                f"<refuter_antithesis>\n{refuter_context}\n</refuter_antithesis>"
            ),
        )

    async def _run_compliance(
        self, reasoning_context: str, refuter_context: str, critique_context: str,
    ) -> str:
        return await self._query_subagent(
            subagent=COMPLIANCE_SUBAGENT,
            prompt=(
                "Stage the proposals that survived critic review for human approval.\n\n"
                "Rules:\n"
                "- Only stage proposals where Critic recommendation is approve or revise.\n"
                "- Include Critic verdict AND Refuter summary in every reason field.\n"
                "- Do NOT stage proposals Critic recommended reject.\n\n"
                f"<proposals>\n{reasoning_context}\n</proposals>\n\n"
                f"<refuter_attacks>\n{refuter_context}\n</refuter_attacks>\n\n"
                f"<critic_synthesis>\n{critique_context}\n</critic_synthesis>"
            ),
        )

    # ------------------------------------------------------------------
    # Plan-aware subagent builder
    # ------------------------------------------------------------------

    @staticmethod
    def _reasoning_subagent_for_plan(plan: WorkflowPlan) -> SubagentConfig:
        from .subagents import SubagentConfig, REASONING_SYSTEM_PROMPT
        thinking = (
            {"type": "enabled", "budget_tokens": plan.reasoning_budget}
            if plan.extended_thinking
            else None
        )
        return SubagentConfig(
            name="reasoning",
            allowed_tools=[],
            system_prompt=REASONING_SYSTEM_PROMPT,
            thinking=thinking,
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
        try:
            import claude  # type: ignore[import-untyped]
        except ImportError:
            _logger.warning("claude-agent-sdk not installed; stub for %s", subagent.name)
            return self._stub_response(subagent.name)

        mcp_config = (
            {
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
            if subagent.allowed_tools
            else {}
        )

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
    # Backward-compatible static helpers (used by test suite)
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_extended_thinking(patient_context: str) -> bool:
        """Heuristic: route to extended thinking if context mentions flag thresholds.

        Preserved for backward compatibility with test_orchestrator.py.
        In the 7-agent architecture the Planner decides this based on
        task description; this method is no longer called by the orchestrator.
        """
        keywords = ["flag", "above", "critical", "urgent", "alert", ">"]
        ctx_lower = patient_context.lower()
        return any(kw in ctx_lower for kw in keywords)

    @staticmethod
    def _parse_proposals(response: str) -> list[dict[str, Any]]:
        """Extract proposals list from a {\"proposals\": [...]} response.

        Preserved for backward compatibility with test_orchestrator.py.
        The 7-agent architecture uses _extract_proposed_observations() for
        Reasoning output and _extract_staged_proposals() for Compliance output.
        """
        try:
            data = json.loads(response)
            return data.get("proposals", [])
        except (json.JSONDecodeError, AttributeError):
            m = re.search(r"\{.*?\"proposals\".*?\}", response, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0)).get("proposals", [])
                except json.JSONDecodeError:
                    pass
        return []

    # ------------------------------------------------------------------
    # Stubs for test/CI
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_response(subagent_name: str) -> str:
        import uuid as _uuid
        stubs: dict[str, str] = {
            "planner": json.dumps({
                "workflow_id": str(_uuid.uuid4()),
                "task_type": "lab_recheck",
                "stages": {
                    "intake": True,
                    "evidence": {"rag": True, "trials": False},
                    "reasoning": {"run": True, "extended_thinking": False, "budget_tokens": 3000},
                    "refuter": {"run": True},
                    "critic": {"run": True},
                    "compliance": True,
                },
                "fast_path": False,
                "plan_rationale": "Stub plan for CI.",
            }),
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
                     "relevance": "Hypertension",
                     "key_thresholds": {"initiation_sbp": 140}}
                ],
                "search_queries": ["hypertension"],
            }),
            "evidence:trials": json.dumps({"trials": []}),
            "reasoning": json.dumps({
                "clinical_summary": "Test patient, no observations.",
                "risk_level": "low",
                "proposed_observations": [],
                "reasoning_notes": "Stub.",
            }),
            "refuter": json.dumps({
                "refuter_verdict": "all_survived",
                "attacks": [],
                "missed_proposals": [],
                "refuter_summary": "No proposals to attack.",
            }),
            "critic": json.dumps({
                "overall_verdict": "approved",
                "overall_rationale": "No proposals.",
                "critiques": [],
                "peer_review_summary": "Nothing to review.",
            }),
            "compliance": json.dumps({"staged": [], "skipped": []}),
        }
        return stubs.get(subagent_name, "{}")
