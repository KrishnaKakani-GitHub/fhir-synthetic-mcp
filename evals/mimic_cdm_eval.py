"""MIMIC-CDM eval for the main governance agent (src/clinical_agent/).

This module sits in evals/ (the main project's eval directory) and grades
the clinical governance agent on the MIMIC-CDM 4-axis rubric.

Bridges both sub-projects
--------------------------
  Input  : MIMIC-CDM cases (evidence_pipeline/datasets/mimic_cdm.py)
  Agent  : src/clinical_agent/orchestrator.py -- the LLM governance agent
  Scoring: evidence_pipeline.datasets.mimic_cdm.score_case (CDMScore)
  Codes  : evidence_pipeline.ontology.cui_mapper (crosswalk validation)

Directional coupling:
  evidence_pipeline -> main project  [OK by design]
  main project -> evidence_pipeline  [also OK in eval context]
  NEVER: production code in src/ imports from evidence_pipeline/

CDM eval axes (Hager et al. Nature Medicine 2024)
--------------------------------------------------
  1. Diagnosis accuracy    -- ICD-10 F1 vs gold diagnosis
  2. Treatment accuracy    -- RxNorm F1 vs gold treatment
  3. Lab ordering accuracy -- LOINC F1 vs gold lab orders
  4. Procedure accuracy    -- CPT F1 vs gold procedures

CI mode: uses mocked governance agent output (deterministic, no LLM tokens)
Prod mode: replace _run_agent() stub with real orchestrator call

PHI NOTE: All cases in CI use synthetic CDM cases (zero patient data).
For real MIMIC-CDM: log case_id only, never clinical presentation text.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from evidence_pipeline.datasets.mimic_cdm import (
    CDMCase,
    CDMReport,
    CDMScore,
    generate_synthetic_cdm_cases,
    score_case,
)
from evidence_pipeline.ontology.cui_mapper import search_by_name

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent interface
# ---------------------------------------------------------------------------

@dataclass
class AgentCDMResponse:
    """Parsed output from the governance agent for one CDM case."""
    case_id: str
    proposed_icd: list[str]
    proposed_rxnorm: list[str]
    proposed_loinc: list[str]
    proposed_cpt: list[str]
    raw_response: str = ""    # for debugging; never log in production
    is_mocked: bool = False


async def _run_agent_mock(case: CDMCase) -> AgentCDMResponse:
    """Mocked governance agent response for CI.

    Production replacement:
      from src.clinical_agent.orchestrator import ClinicalOrchestrator
      orchestrator = ClinicalOrchestrator()
      result = await orchestrator.process_clinical_query(case.presentation)
      return parse_agent_cdm_response(case.case_id, result)
    """
    mapping = search_by_name(case.primary_condition)
    return AgentCDMResponse(
        case_id=case.case_id,
        proposed_icd=case.diagnosis_icd,   # upper bound: gold labels
        proposed_rxnorm=mapping.rxnorm[:2] if mapping and mapping.rxnorm
                        else case.treatment_rxnorm,
        proposed_loinc=mapping.loinc[:3] if mapping and mapping.loinc
                       else case.labs_loinc,
        proposed_cpt=case.procedures_cpt,
        is_mocked=True,
    )


# ---------------------------------------------------------------------------
# Governance-agent-specific report
# ---------------------------------------------------------------------------

GOVERNANCE_CDM_THRESHOLD = 0.75


@dataclass
class GovernanceCDMReport:
    """CDM eval results for the governance agent."""
    scores: list[CDMScore] = field(default_factory=list)
    agent_responses: list[AgentCDMResponse] = field(default_factory=list)
    ci_threshold: float = GOVERNANCE_CDM_THRESHOLD

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean_composite(self) -> float:
        return sum(s.composite for s in self.scores) / self.n if self.n else 0.0

    @property
    def passed(self) -> bool:
        return self.mean_composite >= self.ci_threshold

    def axis_means(self) -> dict[str, float]:
        if not self.scores:
            return {}
        axes = ["diagnosis_score", "treatment_score",
                "lab_ordering_score", "procedure_score"]
        return {
            ax: round(sum(getattr(s, ax) for s in self.scores) / self.n, 4)
            for ax in axes
        }

    def summary(self) -> dict[str, Any]:
        return {
            "source": "governance_agent",
            "n": self.n,
            "mean_composite": round(self.mean_composite, 4),
            "passed": self.passed,
            "ci_threshold": self.ci_threshold,
            "axis_means": self.axis_means(),
            "is_mocked": all(r.is_mocked for r in self.agent_responses),
        }

    def print_summary(self) -> None:
        am = self.axis_means()
        mock_tag = " [MOCKED]" if all(r.is_mocked for r in self.agent_responses) else ""
        print(f"\nGovernance Agent CDM Eval{mock_tag} (Hager et al. 2024 rubric)")
        print(f"  cases             : {self.n}")
        print(f"  composite (mean)  : {self.mean_composite:.3f}")
        print(f"  CI gate           : {self.ci_threshold}  {'PASS' if self.passed else 'FAIL'}")
        for ax, score in am.items():
            bar = "✓" if score >= self.ci_threshold else "✗"
            print(f"  {bar} {ax:<24}: {score:.3f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_governance_cdm_eval_async(
    cases: list[CDMCase] | None = None,
    use_mock: bool = True,
) -> GovernanceCDMReport:
    """Async runner: evaluate governance agent on CDM cases.

    Set use_mock=False in production to call the real agent.
    """
    cases = cases or generate_synthetic_cdm_cases()
    report = GovernanceCDMReport()
    for case in cases:
        if use_mock:
            agent_resp = await _run_agent_mock(case)
        else:
            raise NotImplementedError(
                "Set use_mock=True for CI, or wire to ClinicalOrchestrator."
            )
        score = score_case(
            case,
            predicted_icd=agent_resp.proposed_icd,
            predicted_rxnorm=agent_resp.proposed_rxnorm,
            predicted_loinc=agent_resp.proposed_loinc,
            predicted_cpt=agent_resp.proposed_cpt,
        )
        report.scores.append(score)
        report.agent_responses.append(agent_resp)
        log.info(
            "[CDM] case_id=%s composite=%.3f mocked=%s",
            case.case_id, score.composite, agent_resp.is_mocked,
        )
    return report


def run_governance_cdm_eval(
    cases: list[CDMCase] | None = None,
    use_mock: bool = True,
) -> GovernanceCDMReport:
    """Sync wrapper for run_governance_cdm_eval_async."""
    return asyncio.run(run_governance_cdm_eval_async(cases, use_mock))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    report = run_governance_cdm_eval()
    report.print_summary()
    sys.exit(0 if report.passed else 1)
