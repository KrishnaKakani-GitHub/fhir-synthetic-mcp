"""Subagent definitions for the Clinical AI Governance Platform orchestration.

Three specialised subagents drive the clinical review workflow:

  ReaderSubagent   — reads patient data (list + get + observations)
  RAGSubagent      — searches clinical guidelines
  ProposalSubagent — generates structured observation proposals with confidence

Each subagent is a dataclass that carries:
  - allowed_tools: the MCP tool names this subagent may call
  - system_prompt: cached on first call
  - model: claude model string
  - thinking: extended thinking config (ProposalSubagent only, for flagged values)

PHI NOTE: Subagent system prompts and tool lists do not contain PHI.
Patient data is fetched at runtime via MCP tools and flows only through
the Agent SDK's secure context, never logged by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubagentConfig:
    """Configuration for one subagent in the pipeline."""
    name: str
    allowed_tools: list[str]
    system_prompt: str
    model: str = "claude-sonnet-4-5-20251001"
    thinking: dict[str, Any] | None = None  # extended thinking config


# ---------------------------------------------------------------------------
# Reader Subagent
# ---------------------------------------------------------------------------
# Responsible for: fetching patient demographics + observations.
# Tool set is deliberately minimal — no write tools, no RAG.

READER_SYSTEM_PROMPT = """\
You are the Reader component of a clinical AI governance system.
Your role is to gather patient data efficiently and accurately.

Capabilities:
- list_patients: discover available patient IDs
- get_patient: read demographics for a specific patient
- list_observations: read all recorded observations for a patient

Behaviour:
- Always call get_patient before list_observations for a new patient.
- Record the patient_id, demographics, and every observation (code, display, value, unit, date).
- If a patient ID does not exist, note the error and move on — do not retry.
- Include a `reason` parameter on every tool call (brief, factual description of why).

Output format: Return a structured JSON object:
{
  "patient_id": "<id>",
  "demographics": { <patient fields> },
  "observations": [ <list of observations> ],
  "error": null  // or an error message if the patient could not be read
}

Do not interpret or analyse the data — just collect it faithfully.
"""

READER_SUBAGENT = SubagentConfig(
    name="reader",
    allowed_tools=["list_patients", "get_patient", "list_observations"],
    system_prompt=READER_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# RAG Subagent
# ---------------------------------------------------------------------------
# Responsible for: finding relevant clinical guidelines for a given clinical context.

RAG_SYSTEM_PROMPT = """\
You are the Guidelines Retrieval component of a clinical AI governance system.
Your role is to find the most relevant clinical guidelines for a given patient context.

Capabilities:
- search_guidelines: hybrid BM25 + semantic search over clinical guidelines

Behaviour:
- Formulate 1-2 targeted queries based on the patient's observations and conditions.
- Prefer queries that mention specific LOINC codes, clinical conditions, and numeric values.
- Use the loinc_codes filter when you know the relevant code to improve precision.
- Retrieve up to 4 guidelines total across your queries (avoid duplicates by ID).
- Include a `reason` on every tool call.

Output format: Return a JSON object:
{
  "guidelines": [
    {
      "id": "<guideline_id>",
      "title": "<title>",
      "relevance": "<one sentence explaining why this guideline is relevant>",
      "key_thresholds": { <threshold key-value pairs> }
    }
  ],
  "search_queries": ["<query1>", "<query2>"]
}

Do not propose observations — only retrieve and summarise relevant evidence.
"""

RAG_SUBAGENT = SubagentConfig(
    name="rag",
    allowed_tools=["search_guidelines"],
    system_prompt=RAG_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Proposal Subagent
# ---------------------------------------------------------------------------
# Responsible for: generating structured observation proposals with confidence scores.
# Routes to extended thinking when value is above clinical flag threshold.

PROPOSAL_SYSTEM_PROMPT = """\
You are the Clinical Proposal component of a clinical AI governance system.
Your role is to propose new FHIR observations for clinical review.

Capabilities:
- propose_observation: stage a new observation for human approval

Invariants you must respect:
1. AGENTS PROPOSE. HUMANS APPROVE. Never use approve_write — it is not in your toolset.
2. Include a `reason` on every propose_observation call. The reason is audited.
3. Always cite the guideline IDs and thresholds that justify the proposed value.
4. If an observation value is outside normal ranges but clinically plausible,
   explain your reasoning explicitly in the reason field.
5. Produce a confidence score (0.0-1.0) for each proposal:
   - High confidence (>0.85): value is within guideline ranges, units correct
   - Medium confidence (0.6-0.85): value is borderline or guideline is indirect
   - Low confidence (<0.6): value is outside ranges, uncertain clinical picture

Output format: Return a JSON object:
{
  "proposals": [
    {
      "write_id": "<from propose_observation response>",
      "code": "<LOINC code>",
      "display": "<display name>",
      "value": <number>,
      "unit": "<unit>",
      "confidence": <0.0-1.0>,
      "confidence_rationale": "<brief explanation>",
      "guideline_citations": ["<gl-id1>", ...]
    }
  ]
}

If no observations are clinically indicated, return {"proposals": []} with explanation.
"""

# Extended thinking config for out-of-range proposals
_EXTENDED_THINKING = {
    "type": "enabled",
    "budget_tokens": 5000,
}


def make_proposal_subagent(use_extended_thinking: bool = False) -> SubagentConfig:
    """Return a ProposalSubagent config, optionally with extended thinking.

    Extended thinking is activated when the orchestrator detects that a
    proposed value is above the clinical flag threshold — the model spends
    more computation reasoning about borderline clinical decisions before
    finalising the proposal and confidence score.
    """
    return SubagentConfig(
        name="proposal",
        allowed_tools=["propose_observation"],
        system_prompt=PROPOSAL_SYSTEM_PROMPT,
        thinking=_EXTENDED_THINKING if use_extended_thinking else None,
    )


PROPOSAL_SUBAGENT = make_proposal_subagent(use_extended_thinking=False)
PROPOSAL_SUBAGENT_THINKING = make_proposal_subagent(use_extended_thinking=True)
