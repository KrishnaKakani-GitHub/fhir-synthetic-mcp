"""Clinical Agent SDK orchestration layer.

Three-subagent workflow:
  Reader → RAG → Proposal

Using claude-agent-sdk with PostToolUse hooks for audit + cost tracking.
Extended thinking routing for proposals with above-threshold values.
"""
from .orchestrator import ClinicalOrchestrator, WorkflowResult
from .subagents import (
    PROPOSAL_SUBAGENT,
    PROPOSAL_SUBAGENT_THINKING,
    RAG_SUBAGENT,
    READER_SUBAGENT,
    make_proposal_subagent,
)

__all__ = [
    "ClinicalOrchestrator",
    "WorkflowResult",
    "READER_SUBAGENT",
    "RAG_SUBAGENT",
    "PROPOSAL_SUBAGENT",
    "PROPOSAL_SUBAGENT_THINKING",
    "make_proposal_subagent",
]
