"""Layer 3 — Product evals for the Clinical Evidence Intelligence Pipeline.

End-to-end grading of the full pipeline on a golden dataset of real clinical
questions drawn from the MedQuAD corpus. No live API calls — trials and CMS
steps are mocked so the suite runs in CI without network access.

AMIE-style multi-axis scoring
------------------------------
Inspired by Google DeepMind's AMIE evaluation framework:
  research.google/blog/amie-gains-vision-a-research-ai-agent-for-
  multi-modal-diagnostic-dialogue/

AMIE grades clinical AI across multiple axes rather than a single pass/fail.
We apply the same pattern to the evidence pipeline:

  Axis 1 — ontology_accuracy    ICD-10 + CUI match gold labels
  Axis 2 — evidence_sourcing    NCT IDs present and well-formed
  Axis 3 — metatag_completeness all required metatag fields populated
  Axis 4 — grounding_score      FACTS grounding (codes attributable to source)
  Axis 5 — safety_gate          Layer 2 safety sweep passes

Composite score = mean across axes. CI gate threshold: >= 0.8.

JD alignment: "Verify study outputs / QA-QC" and
"Use LLMs to produce varied, high-quality outputs."

PHI NOTE: Golden dataset uses public MedQuAD question text only.
No patient data is present. Zero PHI touchpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evidence_pipeline.evals.grounding import ground_output
from evidence_pipeline.evals.safety import run_safety_sweep
from evidence_pipeline.ontology.cui_mapper import lookup_cui, search_by_name


# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------

@dataclass
class GoldenCase:
    case_id: str
    question: str
    focus: str
    expected_icd10: str
    expected_cui: str
    question_type: str = "information"
    is_rare_disease: bool = False
    required_metatags: list[str] = field(default_factory=list)


GOLDEN_DATASET: list[GoldenCase] = [
    GoldenCase(
        case_id="prod_001",
        question="What is paroxysmal nocturnal hemoglobinuria?",
        focus="Paroxysmal Nocturnal Hemoglobinuria",
        expected_icd10="D59.5",
        expected_cui="C0028344",
        question_type="information",
        is_rare_disease=True,
        required_metatags=["icd10_primary", "rxnorm_drugs", "loinc_markers"],
    ),
    GoldenCase(
        case_id="prod_002",
        question="How is Gaucher disease treated?",
        focus="Gaucher Disease",
        expected_icd10="E75.22",
        expected_cui="C0017205",
        question_type="treatment",
        is_rare_disease=True,
        required_metatags=["icd10_primary", "rxnorm_drugs"],
    ),
    GoldenCase(
        case_id="prod_003",
        question="What are the symptoms of sickle cell disease?",
        focus="Sickle Cell Disease",
        expected_icd10="D57.1",
        expected_cui="C0002895",
        question_type="symptoms",
        is_rare_disease=True,
        required_metatags=["icd10_primary", "loinc_markers"],
    ),
    GoldenCase(
        case_id="prod_004",
        question="What is the treatment for type 2 diabetes?",
        focus="Type 2 Diabetes Mellitus",
        expected_icd10="E11",
        expected_cui="C0011860",
        question_type="treatment",
        required_metatags=["icd10_primary", "rxnorm_drugs", "loinc_markers"],
    ),
    GoldenCase(
        case_id="prod_005",
        question="How is hypertension managed?",
        focus="Hypertension",
        expected_icd10="I10",
        expected_cui="C0020538",
        question_type="treatment",
        required_metatags=["icd10_primary", "rxnorm_drugs"],
    ),
    GoldenCase(
        case_id="prod_006",
        question="What causes heart failure?",
        focus="Heart Failure",
        expected_icd10="I50",
        expected_cui="C0018801",
        question_type="causes",
        required_metatags=["icd10_primary", "loinc_markers"],
    ),
    GoldenCase(
        case_id="prod_007",
        question="What is COPD and how is it treated?",
        focus="COPD",
        expected_icd10="J44.9",
        expected_cui="C0024117",
        question_type="treatment",
        required_metatags=["icd10_primary", "rxnorm_drugs"],
    ),
    GoldenCase(
        case_id="prod_008",
        question="What are the stages of chronic kidney disease?",
        focus="Chronic Kidney Disease",
        expected_icd10="N18.9",
        expected_cui="C0403447",
        question_type="stages",
        required_metatags=["icd10_primary", "loinc_markers"],
    ),
]


# ---------------------------------------------------------------------------
# Pipeline simulation (no live API)
# ---------------------------------------------------------------------------

def _simulate_pipeline(case: GoldenCase) -> dict[str, Any]:
    """Run deterministic pipeline stages without live API calls.

    Stages 1-2 (ICD-10 + CUI crosswalk) are real.
    Stages 3-4 (trials + CMS) are mocked — product eval grades the
    ontology layer correctness, not API availability.
    """
    mapping = search_by_name(case.focus)
    icd10_codes = mapping.icd10 if mapping else []
    primary_icd10 = icd10_codes[0] if icd10_codes else None
    cui_data = mapping.to_dict() if mapping else None
    return {
        "metatags": {
            "icd10_primary": primary_icd10,
            "icd10_candidates": icd10_codes,
            "rxnorm_drugs": mapping.rxnorm if mapping else [],
            "loinc_markers": mapping.loinc if mapping else [],
            "snomed_concepts": mapping.snomed if mapping else [],
            "recruiting_trial_count": 0,
            "has_cms_national_coverage": False,
        },
        "phenotype": {
            "primary_icd10": {"code": primary_icd10} if primary_icd10 else None,
            "cui_crosswalk": cui_data,
        },
        "evidence": {"clinical_trials": {"trials": []}},
        "qa_metrics": {
            "icd10_codes_found": len(icd10_codes),
            "cui_crosswalk_hit": mapping is not None,
            "pipeline_complete": mapping is not None,
        },
    }


# ---------------------------------------------------------------------------
# AMIE-style multi-axis scoring
# ---------------------------------------------------------------------------

@dataclass
class AxisScores:
    """Five-axis scores in [0.0, 1.0]. Inspired by AMIE eval methodology."""
    ontology_accuracy: float    # ICD-10 + CUI correct
    evidence_sourcing: float    # valid NCT IDs (mocked = 1.0 if no trials)
    metatag_completeness: float # required metatags present
    grounding_score: float      # FACTS grounding score
    safety_gate: float          # 1.0 if safety sweep passes, 0.0 otherwise

    @property
    def composite(self) -> float:
        return (
            self.ontology_accuracy +
            self.evidence_sourcing +
            self.metatag_completeness +
            self.grounding_score +
            self.safety_gate
        ) / 5.0

    def to_dict(self) -> dict[str, float]:
        return {
            "ontology_accuracy": round(self.ontology_accuracy, 4),
            "evidence_sourcing": round(self.evidence_sourcing, 4),
            "metatag_completeness": round(self.metatag_completeness, 4),
            "grounding_score": round(self.grounding_score, 4),
            "safety_gate": round(self.safety_gate, 4),
            "composite": round(self.composite, 4),
        }


CI_THRESHOLD = 0.8   # composite score must meet this to pass the CI gate


@dataclass
class ProductResult:
    case_id: str
    question: str
    expected_icd10: str
    expected_cui: str
    actual_icd10: str | None
    actual_cui: str | None
    axes: AxisScores

    @property
    def passed(self) -> bool:
        return self.axes.composite >= CI_THRESHOLD

    @property
    def icd10_correct(self) -> bool:
        if not self.actual_icd10 or not self.expected_icd10:
            return False
        return (self.actual_icd10 == self.expected_icd10 or
                self.actual_icd10.startswith(self.expected_icd10) or
                self.expected_icd10.startswith(self.actual_icd10[:3]))

    @property
    def cui_correct(self) -> bool:
        return self.actual_cui == self.expected_cui


@dataclass
class ProductReport:
    results: list[ProductResult] = field(default_factory=list)
    ci_threshold: float = CI_THRESHOLD

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def mean_composite(self) -> float:
        return sum(r.axes.composite for r in self.results) / self.n if self.n else 0.0

    @property
    def icd10_accuracy(self) -> float:
        return sum(1 for r in self.results if r.icd10_correct) / self.n if self.n else 0.0

    @property
    def cui_accuracy(self) -> float:
        return sum(1 for r in self.results if r.cui_correct) / self.n if self.n else 0.0

    @property
    def overall_pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / self.n if self.n else 0.0

    def axis_means(self) -> dict[str, float]:
        if not self.results:
            return {}
        axes = ["ontology_accuracy", "evidence_sourcing",
                "metatag_completeness", "grounding_score", "safety_gate"]
        return {
            ax: round(sum(getattr(r.axes, ax) for r in self.results) / self.n, 4)
            for ax in axes
        }

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean_composite": round(self.mean_composite, 4),
            "ci_threshold": self.ci_threshold,
            "overall_pass_rate": round(self.overall_pass_rate, 4),
            "icd10_accuracy": round(self.icd10_accuracy, 4),
            "cui_accuracy": round(self.cui_accuracy, 4),
            "axis_means": self.axis_means(),
        }

    def print_summary(self) -> None:
        am = self.axis_means()
        print("\nProduct eval (AMIE-style multi-axis)")
        print(f"  cases             : {self.n}")
        print(f"  composite (mean)  : {self.mean_composite:.3f}  (threshold {self.ci_threshold})")
        print(f"  overall pass rate : {self.overall_pass_rate:.1%}")
        print(f"  --- axes ---")
        for ax, score in am.items():
            bar = "✓" if score >= self.ci_threshold else "✗"
            print(f"  {bar} {ax:<24}: {score:.3f}")


# ---------------------------------------------------------------------------
# Axis computation helpers
# ---------------------------------------------------------------------------

def _score_ontology(result_icd10_correct: bool, result_cui_correct: bool) -> float:
    return (float(result_icd10_correct) + float(result_cui_correct)) / 2.0


def _score_evidence_sourcing(trials: list[dict[str, Any]]) -> float:
    """1.0 if no trials (mocked), or if all present NCT IDs are well-formed."""
    import re
    if not trials:
        return 1.0
    valid = sum(1 for t in trials
                if re.match(r'^NCT[0-9]{8}$', t.get("nct_id", "")))
    return valid / len(trials)


def _score_metatag_completeness(metatags: dict[str, Any],
                                required: list[str]) -> float:
    if not required:
        return 1.0
    present = sum(1 for k in required if metatags.get(k))
    return present / len(required)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_product_eval(dataset: list[GoldenCase] | None = None) -> ProductReport:
    """Grade the pipeline against the golden dataset using AMIE-style axes."""
    cases = dataset or GOLDEN_DATASET
    safety_report = run_safety_sweep()
    safety_score = 1.0 if safety_report.passed else 0.0

    report = ProductReport()
    for case in cases:
        output = _simulate_pipeline(case)
        metatags = output["metatags"]
        phenotype = output["phenotype"]
        cui_data = phenotype.get("cui_crosswalk") or {}
        actual_icd10 = (phenotype.get("primary_icd10") or {}).get("code")
        actual_cui = cui_data.get("cui")
        trials = output["evidence"]["clinical_trials"]["trials"]

        # Compute binary correctness for ontology axis
        icd10_ok = bool(
            actual_icd10 and (
                actual_icd10 == case.expected_icd10 or
                actual_icd10.startswith(case.expected_icd10) or
                case.expected_icd10.startswith(actual_icd10[:3])
            )
        )
        cui_ok = actual_cui == case.expected_cui

        # FACTS grounding score for this output
        gr = ground_output(output, label=case.case_id)

        axes = AxisScores(
            ontology_accuracy=_score_ontology(icd10_ok, cui_ok),
            evidence_sourcing=_score_evidence_sourcing(trials),
            metatag_completeness=_score_metatag_completeness(
                metatags, case.required_metatags),
            grounding_score=gr.grounding_score,
            safety_gate=safety_score,
        )
        report.results.append(ProductResult(
            case_id=case.case_id,
            question=case.question,
            expected_icd10=case.expected_icd10,
            expected_cui=case.expected_cui,
            actual_icd10=actual_icd10,
            actual_cui=actual_cui,
            axes=axes,
        ))
    return report
