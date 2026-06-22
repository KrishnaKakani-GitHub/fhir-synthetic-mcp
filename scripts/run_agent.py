#!/usr/bin/env python3
"""CLI entry point for the Clinical AI Governance Platform workflow.

Usage:
    python scripts/run_agent.py --patient pat-001
    python scripts/run_agent.py --patient pat-001 --thinking
    python scripts/run_agent.py --patient pat-001 --session <session_id>

Requires:
    ANTHROPIC_API_KEY env var
    FHIR_MCP_DB pointing to a seeded SQLite database (run seed_db.py first)
    claude-agent-sdk installed (pip install claude-agent-sdk)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)

from clinical_agent import ClinicalOrchestrator


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Clinical AI Governance workflow for a patient."
    )
    parser.add_argument(
        "--patient", default="pat-001",
        help="Patient ID to analyse (default: pat-001)",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Force extended thinking for the proposal subagent",
    )
    parser.add_argument(
        "--session", default=None,
        help="Resume an existing session by ID",
    )
    parser.add_argument(
        "--actor", default="orchestrator:cli",
        help="Actor identity for audit records",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "WARNING: ANTHROPIC_API_KEY not set. "
            "Agent SDK calls will fail. Set it before running.",
            file=sys.stderr,
        )

    orch = ClinicalOrchestrator(actor=args.actor)
    result = await orch.run_workflow(
        patient_id=args.patient,
        session_id=args.session,
        force_extended_thinking=args.thinking,
    )

    print(json.dumps(result.to_dict(), indent=2))

    metrics = result.metrics
    print(f"\n--- Workflow metrics ---", file=sys.stderr)
    print(f"  Total tool calls:  {metrics['total_tool_calls']}", file=sys.stderr)
    print(f"  Total latency:     {metrics['total_latency_ms']} ms", file=sys.stderr)
    print(f"  Estimated cost:    ${metrics['estimated_cost_usd']}", file=sys.stderr)
    print(f"  Extended thinking: {result.used_extended_thinking}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
