"""Tests for the principal / approver auth layer."""
from __future__ import annotations

import pytest

from fhir_mcp.auth import AuthError, verify_agent_actor, verify_approver


def test_dev_mode_allows_any_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FHIR_MCP_PRINCIPALS", raising=False)
    verify_agent_actor("any-random-actor")  # must not raise


def test_dev_mode_allows_any_approver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FHIR_MCP_APPROVERS", raising=False)
    verify_approver("anyone")  # must not raise


def test_rejects_unknown_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FHIR_MCP_PRINCIPALS", "agent:prod,agent:staging")
    with pytest.raises(AuthError, match="not an authorised principal"):
        verify_agent_actor("agent:evil")


def test_accepts_known_principal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FHIR_MCP_PRINCIPALS", "agent:prod,agent:staging")
    verify_agent_actor("agent:prod")     # no raise
    verify_agent_actor("agent:staging")  # no raise


def test_rejects_unknown_approver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FHIR_MCP_APPROVERS", "dr.smith,dr.jones")
    with pytest.raises(AuthError, match="not authorised"):
        verify_approver("dr.evil")


def test_accepts_known_approver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FHIR_MCP_APPROVERS", "dr.smith,dr.jones")
    verify_approver("dr.smith")  # no raise


def test_agent_cannot_self_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent principals must not appear in the approvers set."""
    monkeypatch.setenv("FHIR_MCP_PRINCIPALS", "agent:prod")
    monkeypatch.setenv("FHIR_MCP_APPROVERS", "dr.smith")
    verify_agent_actor("agent:prod")  # agent can call tools
    with pytest.raises(AuthError):    # but cannot approve
        verify_approver("agent:prod")


def test_whitespace_tolerant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FHIR_MCP_PRINCIPALS", " agent:prod , agent:staging ")
    verify_agent_actor("agent:prod")  # strips whitespace
