"""Tests for the Agent SDK orchestration layer.

The Agent SDK (claude-agent-sdk) requires ANTHROPIC_API_KEY and a running
MCP server, which are not available in CI. We test:
  1. The orchestrator falls back gracefully when the SDK is not installed
  2. Subagent configs are correctly structured
  3. Hooks accumulate metrics correctly
  4. Extended thinking routing logic
  5. Proposal + guideline parsing from structured responses

SDK integration tests are in tests/test_orchestrator_integration.py (manual).
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import pytest

# Ensure BM25/Chroma don't start in CI
os.environ.setdefault("FHIR_MCP_RAG_DISABLE_CHROMA", "1")

from clinical_agent.hooks import AuditHook, WriteGateHook
from clinical_agent.orchestrator import ClinicalOrchestrator, WorkflowResult
from clinical_agent.subagents import (
    PROPOSAL_SUBAGENT,
    PROPOSAL_SUBAGENT_THINKING,
    RAG_SUBAGENT,
    READER_SUBAGENT,
    make_proposal_subagent,
)


# --- Subagent config tests ----------------------------------------------------


def test_reader_subagent_tools() -> None:
    assert "list_patients" in READER_SUBAGENT.allowed_tools
    assert "get_patient" in READER_SUBAGENT.allowed_tools
    assert "list_observations" in READER_SUBAGENT.allowed_tools
    # Reader must not have write tools
    assert "approve_write" not in READER_SUBAGENT.allowed_tools
    assert "propose_observation" not in READER_SUBAGENT.allowed_tools


def test_rag_subagent_tools() -> None:
    assert READER_SUBAGENT.allowed_tools  # non-empty
    assert RAG_SUBAGENT.allowed_tools == ["search_guidelines"]


def test_proposal_subagent_no_approve() -> None:
    assert "propose_observation" in PROPOSAL_SUBAGENT.allowed_tools
    assert "approve_write" not in PROPOSAL_SUBAGENT.allowed_tools


def test_proposal_subagent_extended_thinking() -> None:
    assert PROPOSAL_SUBAGENT.thinking is None
    assert PROPOSAL_SUBAGENT_THINKING.thinking is not None
    assert PROPOSAL_SUBAGENT_THINKING.thinking["type"] == "enabled"
    assert PROPOSAL_SUBAGENT_THINKING.thinking["budget_tokens"] >= 1000


def test_make_proposal_subagent_toggle() -> None:
    with_thinking = make_proposal_subagent(use_extended_thinking=True)
    without_thinking = make_proposal_subagent(use_extended_thinking=False)
    assert with_thinking.thinking is not None
    assert without_thinking.thinking is None


# --- Hook tests ---------------------------------------------------------------


def test_audit_hook_accumulates_calls() -> None:
    hook = AuditHook(actor="test:actor")
    hook.pre_tool_use("list_patients", {"reason": "test"})
    hook.post_tool_use(
        "list_patients",
        {"reason": "test"},
        ["pat-001"],
        usage={"input_tokens": 100, "output_tokens": 50},
    )
    assert hook.total_tool_calls == 1
    assert hook.total_input_tokens == 100
    assert hook.total_output_tokens == 50
    assert hook.tool_error_count == 0


def test_audit_hook_counts_errors() -> None:
    hook = AuditHook(actor="test:actor")
    hook.pre_tool_use("propose_observation", {"reason": "test"})
    hook.post_tool_use(
        "propose_observation",
        {"reason": "test"},
        ValueError("Validation failed"),
    )
    assert hook.tool_error_count == 1


def test_audit_hook_metrics_structure() -> None:
    hook = AuditHook(actor="test:actor")
    metrics = hook.metrics()
    required_keys = [
        "total_tool_calls", "total_latency_ms", "tool_error_count",
        "total_input_tokens", "total_output_tokens", "estimated_cost_usd",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric key: {key}"


def test_write_gate_hook_blocks_empty_approver() -> None:
    hook = WriteGateHook()
    with pytest.raises(ValueError, match="cannot self-approve"):
        hook.pre_tool_use("approve_write", {"write_id": "pw-123", "approver": ""})


def test_write_gate_hook_allows_named_approver() -> None:
    hook = WriteGateHook()
    result = hook.pre_tool_use(
        "approve_write",
        {"write_id": "pw-123", "approver": "dr.smith"},
    )
    assert result is None  # pass-through


def test_write_gate_hook_passes_read_tools() -> None:
    hook = WriteGateHook()
    result = hook.pre_tool_use("list_patients", {"reason": "test"})
    assert result is None


# --- Orchestrator stub tests --------------------------------------------------


def test_orchestrator_stub_workflow() -> None:
    """Run the full workflow without the Agent SDK (stub mode)."""
    orch = ClinicalOrchestrator(actor="test:orchestrator")
    result = asyncio.get_event_loop().run_until_complete(
        orch.run_workflow(patient_id="pat-001")
    )
    assert isinstance(result, WorkflowResult)
    assert result.patient_id == "pat-001"
    assert isinstance(result.proposals, list)
    assert isinstance(result.guidelines, list)
    assert "total_tool_calls" in result.metrics


def test_orchestrator_workflow_result_serialises() -> None:
    orch = ClinicalOrchestrator()
    result = asyncio.get_event_loop().run_until_complete(
        orch.run_workflow(patient_id="pat-001")
    )
    d = result.to_dict()
    assert d["patient_id"] == "pat-001"
    # Should be JSON-serialisable
    json.dumps(d)


# --- Extended thinking routing ------------------------------------------------


def test_needs_extended_thinking_detects_critical() -> None:
    ctx_with_flag = "Patient heart rate 220 bpm — above flag threshold"
    ctx_normal = "Patient heart rate 75 bpm within normal range"
    assert ClinicalOrchestrator._needs_extended_thinking(ctx_with_flag)
    assert not ClinicalOrchestrator._needs_extended_thinking(ctx_normal)


# --- Proposal parsing ---------------------------------------------------------


def test_parse_proposals_valid_json() -> None:
    response = json.dumps({
        "proposals": [
            {
                "write_id": "pw-abc",
                "code": "8867-4",
                "display": "Heart rate",
                "value": 75,
                "unit": "/min",
                "confidence": 0.92,
                "confidence_rationale": "Value within guideline range",
                "guideline_citations": ["gl-003"],
            }
        ]
    })
    proposals = ClinicalOrchestrator._parse_proposals(response)
    assert len(proposals) == 1
    assert proposals[0]["confidence"] == 0.92


def test_parse_proposals_empty_response() -> None:
    proposals = ClinicalOrchestrator._parse_proposals("{}")
    assert proposals == []


def test_parse_proposals_malformed_returns_empty() -> None:
    proposals = ClinicalOrchestrator._parse_proposals("not json at all")
    assert proposals == []
