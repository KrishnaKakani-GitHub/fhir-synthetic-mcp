"""Agent SDK hooks for the Clinical AI Governance Platform.

Hooks intercept every tool call the agent makes. We use PostToolUse to:
  1. Append a tamper-evident audit record (who called what, when, outcome)
  2. Track per-call latency and estimated token cost
  3. Gate on write operations (assert approve_write requires an approver)

Hook API (claude-agent-sdk):
  - PreToolUseHook(tool_name, tool_input) -> modified input | None (pass-through)
  - PostToolUseHook(tool_name, tool_input, tool_output, usage) -> None

PHI NOTE: Hooks log tool names, patient IDs (from target_ids), and error
messages. They NEVER log full tool output, which may contain observation values.
"""
from __future__ import annotations

import time
from typing import Any

# Audit is imported from the MCP package so hooks share the same chain.
# In production these would be separate services; here we share the module
# for simplicity and because it's all the same process in demo mode.
try:
    from fhir_mcp.audit import audit
except ImportError:
    def audit(**kwargs: Any) -> None:  # type: ignore[misc]
        pass  # fallback if MCP package not installed


_WRITE_TOOLS = frozenset(["approve_write", "reject_write"])
_READ_TOOLS = frozenset(["list_patients", "get_patient", "list_observations",
                         "list_pending_writes"])
_PROPOSE_TOOLS = frozenset(["propose_observation", "search_guidelines"])


class AuditHook:
    """PostToolUse hook: tamper-evident audit record + latency tracking.

    Attach to AgentSDK via:
        agent = query(..., hooks=[AuditHook(actor="orchestrator:prod")])
    """

    def __init__(self, actor: str = "orchestrator:dev") -> None:
        self._actor = actor
        self._call_starts: dict[str, float] = {}  # tool_call_id -> start time
        # Metrics accumulate across the session
        self.total_tool_calls: int = 0
        self.total_latency_ms: float = 0.0
        self.tool_error_count: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def pre_tool_use(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        """Record call start time keyed by (tool_name, write_id or patient_id)."""
        key = self._call_key(tool_name, tool_input)
        self._call_starts[key] = time.monotonic()

    def post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Emit an audit record and update session metrics."""
        key = self._call_key(tool_name, tool_input)
        start = self._call_starts.pop(key, time.monotonic())
        latency_ms = round((time.monotonic() - start) * 1000, 1)

        self.total_tool_calls += 1
        self.total_latency_ms += latency_ms

        # Extract token counts from usage if available
        input_tokens = 0
        output_tokens = 0
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

        # Determine outcome and target IDs
        outcome = "ok"
        target_ids: list[str] = []
        error_msg: str | None = None

        if isinstance(tool_output, Exception):
            outcome = "error"
            error_msg = str(tool_output)
            self.tool_error_count += 1
        elif isinstance(tool_output, dict):
            if "write_id" in tool_output:
                target_ids.append(tool_output["write_id"])
            if "id" in tool_output:  # Observation id
                target_ids.append(tool_output["id"])

        if "patient_id" in tool_input:
            target_ids.insert(0, tool_input["patient_id"])
        if "write_id" in tool_input:
            target_ids.insert(0, tool_input["write_id"])

        audit(
            actor=self._actor,
            action=f"sdk_tool:{tool_name}",
            reason=tool_input.get("reason", ""),
            target_ids=target_ids,
            outcome=outcome,
            extra={
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                **(({"error": error_msg}) if error_msg else {}),
            },
        )

    def metrics(self) -> dict[str, Any]:
        """Return accumulated session metrics."""
        return {
            "total_tool_calls": self.total_tool_calls,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "tool_error_count": self.tool_error_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self._estimate_cost(),
        }

    def _estimate_cost(self) -> float:
        """Rough cost estimate using claude-sonnet-4 pricing (input $3/M, output $15/M)."""
        return round(
            (self.total_input_tokens / 1_000_000) * 3.0
            + (self.total_output_tokens / 1_000_000) * 15.0,
            6,
        )

    @staticmethod
    def _call_key(tool_name: str, tool_input: dict[str, Any]) -> str:
        patient = tool_input.get("patient_id", "")
        write = tool_input.get("write_id", "")
        return f"{tool_name}:{patient}:{write}:{time.time_ns()}"


class WriteGateHook:
    """PreToolUse hook: blocks approve_write if approver is not set.

    This is a defence-in-depth check — auth.py also enforces this.
    """

    def pre_tool_use(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> dict[str, Any] | None:
        if tool_name in ("approve_write", "reject_write"):
            approver = tool_input.get("approver", "").strip()
            if not approver:
                raise ValueError(
                    f"{tool_name} requires an 'approver' field identifying the "
                    "human making this decision. The agent cannot self-approve."
                )
        return None  # pass-through

    def post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
        usage: dict[str, Any] | None = None,
    ) -> None:
        pass  # no post-processing needed
