"""MIMIC-CDM eval for the main governance agent (src/clinical_agent/).

This module sits in evals/ (the main project's eval directory) and grades
the clinical governance agent on the MIMIC-CDM 4-axis rubric.

Bridges both sub-projects
--------------------------
  Input  : MIMIC-CDM cases (evidence_pipeline/datasets/mimic_cdm.py)
  Agent  : evals/cdm_agent.py -- pluggable model backend (mock | meditron)
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

Backends (evals/cdm_agent.py)
  mock     : echoes gold labels -> 1.000 by construction. CI default.
  meditron : Meditron-7B via local Ollama. LOCAL-ONLY (never in CI).

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
    CDMScore,
    generate_synthetic_cdm_cases,
    score_case,
)
from evidence_pipeline.evals.clinical_decision import CDMReport
from evidence_pipeline.ontology.cui_mapper import search_by_name

log = logging.getLogger(__name__)


def _gold_lookup(cases: list[CDMCase]) -> dict[str, dict[str, list[str]]]:
    """Build case_id -> gold codes map for the mock backend."""
    return {
        c.case_id: {
            "icd": c.diagnosis_icd,
            "rxnorm": c.treatment_rxnorm,
            "loinc": c.labs_loinc,
            "cpt": c.procedures_cpt,
        }
        for c in cases
    }


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
    backend: str = "mock",
) -> GovernanceCDMReport:
    """Async runner: evaluate the CDM model backend on CDM cases.

    backend: 'mock' (CI default, deterministic) or 'meditron' (local Ollama).
    use_mock is kept for backward compat; use_mock=False + backend='meditron'
    runs the real model.
    """
    from evals.cdm_agent import make_backend

    cases = cases or generate_synthetic_cdm_cases()
    # Resolve backend: explicit non-mock backend overrides use_mock.
    resolved = "mock" if (use_mock and backend == "mock") else backend
    agent = make_backend(resolved, gold_lookup=_gold_lookup(cases))

    report = GovernanceCDMReport()
    for case in cases:
        resp = agent.propose(case.case_id, case.presentation)
        # Adapt cdm_agent.AgentCDMResponse -> local AgentCDMResponse shape.
        agent_resp = AgentCDMResponse(
            case_id=resp.case_id,
            proposed_icd=resp.proposed_icd,
            proposed_rxnorm=resp.proposed_rxnorm,
            proposed_loinc=resp.proposed_loinc,
            proposed_cpt=resp.proposed_cpt,
            is_mocked=resp.is_mocked,
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
            "[CDM] case_id=%s composite=%.3f backend=%s",
            case.case_id, score.composite, resolved,
        )
    return report


def run_governance_cdm_eval(
    cases: list[CDMCase] | None = None,
    use_mock: bool = True,
    backend: str = "mock",
) -> GovernanceCDMReport:
    """Sync wrapper for run_governance_cdm_eval_async."""
    return asyncio.run(run_governance_cdm_eval_async(cases, use_mock, backend))


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MIMIC-CDM governance eval")
    parser.add_argument(
        "--model", default="mock", choices=["mock", "meditron"],
        help="Backend: 'mock' (CI default) or 'meditron' (local Ollama).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    report = run_governance_cdm_eval(
        use_mock=(args.model == "mock"), backend=args.model,
    )
    report.print_summary()
    sys.exit(0 if report.passed else 1)
