"""Clinical Agent SDK orchestration layer.

7-agent dynamic workflow:
  Planner → Intake → Evidence (parallel RAG + Trials)
  → Reasoning → Refuter → Critic → Compliance

Using claude-agent-sdk with PostToolUse hooks for audit + cost tracking.
Planner selects which stages run based on task type.
CheckpointStore persists stage outputs for resume-on-interrupt.
"""
from .orchestrator import ClinicalOrchestrator, WorkflowPlan, WorkflowResult
from .subagents import (
    COMPLIANCE_SUBAGENT,
    CRITIC_SUBAGENT,
    EVIDENCE_RAG_SUBAGENT,
    EVIDENCE_TRIALS_SUBAGENT,
    INTAKE_SUBAGENT,
    PLANNER_SUBAGENT,
    PROPOSAL_SUBAGENT,
    PROPOSAL_SUBAGENT_THINKING,
    RAG_SUBAGENT,
    READER_SUBAGENT,
    REASONING_SUBAGENT,
    REFUTER_SUBAGENT,
    make_proposal_subagent,
)

__all__ = [
    # Orchestrator
    "ClinicalOrchestrator",
    "WorkflowResult",
    "WorkflowPlan",
    # New subagents
    "PLANNER_SUBAGENT",
    "INTAKE_SUBAGENT",
    "EVIDENCE_RAG_SUBAGENT",
    "EVIDENCE_TRIALS_SUBAGENT",
    "REASONING_SUBAGENT",
    "REFUTER_SUBAGENT",
    "CRITIC_SUBAGENT",
    "COMPLIANCE_SUBAGENT",
    # Backward-compatible aliases
    "READER_SUBAGENT",
    "RAG_SUBAGENT",
    "PROPOSAL_SUBAGENT",
    "PROPOSAL_SUBAGENT_THINKING",
    "make_proposal_subagent",
]
