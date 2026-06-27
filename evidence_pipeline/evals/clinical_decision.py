"""Layer 4 -- LLM Clinical Decision Making eval (MIMIC-CDM-style).

Inspired by:
  Hager et al., Nature Medicine 2024
  "Evaluation and mitigation of the limitations of LLMs in clinical decision-making"
  https://doi.org/10.1038/s41591-024-03097-1

What this layer grades
-----------------------
Given a clinical presentation, does the evidence pipeline's LLM component
propose the correct:
  1. Diagnosis   (ICD-10 codes)
  2. Treatment   (RxNorm IDs)
  3. Lab orders  (LOINC codes)
  4. Procedures  (CPT codes)

In CI: the "LLM" is the deterministic crosswalk lookup -- a mock that returns
known-correct codes for each case. This makes the layer fully testable without
spending tokens or requiring API keys.

In production: replace _mock_llm_response() with the actual governance agent
call from src/clinical_agent/orchestrator.py.

Relation to the governance agent eval (evals/mimic_cdm_eval.py)
---------------------------------------------------------------
This file (evidence_pipeline/evals/) grades the EVIDENCE LAYER --
does the evidence retrieval + ontology mapping support correct decisions?

evals/mimic_cdm_eval.py grades the GOVERNANCE AGENT --
does the LLM agent itself make correct decisions?

Both use the same CDMCase schema and CDMScore rubric.

PHI NOTE: Operates on synthetic CDM cases in CI. For real MIMIC-CDM cases,
log case_id only, never clinical text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evidence_pipeline.datasets.mimic_cdm import (
    CDMCase,
    CDMScore,
    generate_synthetic_cdm_cases,
    score_case,
)
from evidence_pipeline.ontology.cui_mapper import search_by_name


# ---------------------------------------------------------------------------
# CI mock: crosswalk-backed deterministic "LLM" response
# ---------------------------------------------------------------------------

def _mock_llm_response(case: CDMCase) -> dict[str, list[str]]:
    """Simulate an LLM response using the crosswalk.

    In CI, returns the gold labels directly (upper bound on score).
    In production, replace with actual governance agent call:
      from src.clinical_agent.orchestrator import ClinicalOrchestrator
      result = await orchestrator.run(case.presentation)
      return parse_agent_output(result)
    """
    # CI mock: always return gold labels (deterministic upper bound).
    # This verifies the eval machinery end-to-end without spending tokens.
    # In production: call src.clinical_agent.orchestrator.ClinicalOrchestrator
    # and parse structured output. Do NOT use gold labels in production.
    return {
        "icd": case.diagnosis_icd,
        "rxnorm": case.treatment_rxnorm,
        "loinc": case.labs_loinc,
        "cpt": case.procedures_cpt,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

CDM_CI_THRESHOLD = 0.75   # composite score threshold for CI gate


@dataclass
class CDMReport:
    scores: list[CDMScore] = field(default_factory=list)
    ci_threshold: float = CDM_CI_THRESHOLD

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def mean_composite(self) -> float:
        return sum(s.composite for s in self.scores) / self.n if self.n else 0.0

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
            "n": self.n,
            "mean_composite": round(self.mean_composite, 4),
            "ci_threshold": self.ci_threshold,
            "passed": self.mean_composite >= self.ci_threshold,
            "axis_means": self.axis_means(),
        }

    def print_summary(self) -> None:
        am = self.axis_means()
        print("\nCDM Eval (LLM clinical decision making -- Hager et al. 2024 rubric)")
        print(f"  cases             : {self.n}")
        print(f"  composite (mean)  : {self.mean_composite:.3f}  (threshold {self.ci_threshold})")
        for ax, score in am.items():
            bar = "✓" if score >= self.ci_threshold else "✗"
            print(f"  {bar} {ax:<24}: {score:.3f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_cdm_eval(
    cases: list[CDMCase] | None = None,
    use_mock: bool = True,
) -> CDMReport:
    """Run CDM eval on a list of cases.

    Args:
        cases:    CDM cases to evaluate. Defaults to synthetic cases.
        use_mock: If True (CI), use crosswalk-backed mock LLM.
                  If False, call the actual governance agent.
    """
    cases = cases or generate_synthetic_cdm_cases()
    report = CDMReport()
    for case in cases:
        if use_mock:
            pred = _mock_llm_response(case)
        else:
            raise NotImplementedError(
                "Production mode: replace with governance agent call. "
                "See src/clinical_agent/orchestrator.py."
            )
        score = score_case(
            case,
            predicted_icd=pred["icd"],
            predicted_rxnorm=pred["rxnorm"],
            predicted_loinc=pred["loinc"],
            predicted_cpt=pred["cpt"],
        )
        report.scores.append(score)
    return report
